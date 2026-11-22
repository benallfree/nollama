// nollama's Metal kernels. Compiled at runtime from this source (see gpu.go), so builds need no Xcode.
#include <metal_stdlib>
using namespace metal;

// Decodes the 6-bit scale and min of sub-block j from a Q4_K scales array (as scaleMinK4 in quant.go).
static inline uchar2 scale_min_k4(int j, device const uchar *q) {
	if (j < 4) {
		return uchar2(q[j] & 63, q[j + 4] & 63);
	}
	return uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// scale_min_k4x2 is the 6-bit scales and mins of Q4_K sub-blocks k and k+1 (k even) from a block's first 16 bytes
// h (d, dmin, scales[12]): (scale k, scale k+1, min k, min k+1). Masks on registers, both bytes at once (k and k+1
// are both below 4 or both not).
static inline float4 scale_min_k4x2(uint k, uint4 h) {
	uint sh = 8 * (k % 4), y = h.y >> sh, z = h.z >> sh;
	uint sc, mn;
	if (k < 4) {
		sc = y & 0x3f3f, mn = z & 0x3f3f;
	} else {
		uint w = h.w >> sh;
		sc = (w & 0x0f0f) | ((y >> 2) & 0x3030), mn = ((w >> 4) & 0x0f0f) | ((z >> 2) & 0x3030);
	}
	return float4(sc & 0xff, (sc >> 8) & 0xff, mn & 0xff, (mn >> 8) & 0xff);
}

struct MatvecArgs {
	uint rows;     // output rows
	uint nb;       // 256-element blocks per row
	uint rowBytes; // bytes per weight row
	uint cols;     // input length (nb*256)
};

// MV_ROWS: rows per simdgroup in the matvecs. Each lane loads its activations once per block and uses
// them for all MV_ROWS rows: re-reading them per row, not the weights, was what limited these kernels.
#define MV_ROWS 8

// Pair mode (gate/up/SwiGLU in one dispatch): a simdgroup's MV_ROWS virtual rows are MV_ROWS/2 rows of w and the same
// rows of w2 (same type and shape), and it writes silu(w row . x) * (w2 row . x), as swiglu_f32. Grid then covers
// rows / (MV_ROWS/2 * simdgroups).
#define MV_OUT (PAIR ? MV_ROWS / 2 : MV_ROWS)

// mv_row is virtual row r's weights for a simdgroup whose first output row is r0.
template <bool PAIR>
static inline device const uchar *mv_row(device const uchar *w, device const uchar *w2, uint r0, int r, uint rows, uint rowBytes) {
	if (PAIR) {
		return (r < MV_ROWS / 2 ? w : w2) + (ulong)min(r0 + (uint)(r % (MV_ROWS / 2)), rows - 1) * rowBytes;
	}
	return w + (ulong)min(r0 + (uint)r, rows - 1) * rowBytes;
}

// mv_store writes a simdgroup's sums (simd_sum'ed in every lane) for its nr output rows.
template <bool PAIR>
static inline void mv_store(thread float *sum, device float *y, ulong base, uint nr, uint lane) {
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		sum[r] = simd_sum(sum[r]);
	}
	if (lane != 0) {
		return;
	}
#pragma unroll
	for (int r = 0; r < MV_OUT; r++) {
		if (r < (int)nr) {
			if (PAIR) {
				float g = sum[r];
				y[base + r] = g / (1 + exp(-g)) * sum[r + MV_ROWS / 2];
			} else {
				y[base + r] = sum[r];
			}
		}
	}
}

// y[t*rows + r] = dot(W row r, x[t*cols:]) for Q4_K weights and float activations.
// A simdgroup computes MV_ROWS rows. Eight lanes share a block (lane groups take blocks g, g+4, ...); lane i of a
// group covers 32 elements: sub-blocks 2j and 2j+1 (j = i/2), quant bytes 16*(i%2).. of chunk j (one 16-byte
// load). Grid: (rows / (MV_OUT * simdgroups), tokens).
template <bool PAIR>
kernel void matvec_q4k(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           device const uchar *w2 [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_OUT;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_OUT, a.rows - r0);
	device const float *xt = x + (ulong)tg.y * a.cols;
	uint il = lane % 8, j = il / 2, off = 32 * j + (il % 2) * 16;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 8; b < a.nb; b += 4) {
		device const float4 *x1 = (device const float4 *)(xt + b * 256 + 64 * j + (il % 2) * 16);
		float4 a1[4], a2[4];
		float t1 = 0, t2 = 0;
		for (int i = 0; i < 4; i++) {
			a1[i] = x1[i];
			a2[i] = x1[i + 8]; // +32 elements
			t1 += a1[i].x + a1[i].y + a1[i].z + a1[i].w;
			t2 += a2[i].x + a2[i].y + a2[i].z + a2[i].w;
		}
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) { // rows past the end repeat the last row (not written)
			device const uchar *blk = mv_row<PAIR>(w, w2, r0, r, a.rows, a.rowBytes) + b * 144;
			uint4 hd = *(device const uint4 *)blk; // d, dmin, scales[12]: one 16-byte load (blocks are 16-aligned)
			float2 dm = float2(as_type<half2>(hd.x));
			float d = dm.x, dmin = dm.y;
			float4 sm = scale_min_k4x2(2 * j, hd);
			uint4 q = *(device const uint4 *)(blk + 16 + off);
			float s1 = 0, s2 = 0;
			for (int i = 0; i < 4; i++) {
				uint v = q[i];
				s1 += dot(float4(as_type<uchar4>(v & 0x0F0F0F0F)), a1[i]);
				s2 += dot(float4(as_type<uchar4>((v >> 4) & 0x0F0F0F0F)), a2[i]);
			}
			sum[r] += d * (sm.x * s1 + sm.y * s2) - dmin * (sm.z * t1 + sm.w * t2);
		}
	}
	mv_store<PAIR>(sum, y, (ulong)tg.y * a.rows + r0, nr, lane);
}
template [[host_name("matvec_q4k_f32")]] kernel decltype(matvec_q4k<false>) matvec_q4k<false>;
template [[host_name("matvec_q4k_pair")]] kernel decltype(matvec_q4k<true>) matvec_q4k<true>;

// Same contract for Q6_K weights (210-byte blocks: ql[128], qh[64], scales[16] int8, d f16). Eight lanes share a
// block; lane i takes half h = i/4 and l = 8*(i%4)..+8 of it, which with l+32, l+64, l+96 is 32 elements.
template <bool PAIR>
kernel void matvec_q6k(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           device const uchar *w2 [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_OUT;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_OUT, a.rows - r0);
	device const float *xt = x + (ulong)tg.y * a.cols;
	uint il = lane % 8, h = il / 4, l0 = 8 * (il % 4), is = l0 / 16;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 8; b < a.nb; b += 4) {
		device const float4 *xh = (device const float4 *)(xt + b * 256 + 128 * h + l0);
		float4 xv[4][2]; // quarters 0..3 (elements +0, +32, +64, +96), 8 values each
		for (int q = 0; q < 4; q++) {
			xv[q][0] = xh[8 * q];
			xv[q][1] = xh[8 * q + 1];
		}
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) { // rows past the end repeat the last row (not written)
			device const uchar *blk = mv_row<PAIR>(w, w2, r0, r, a.rows, a.rowBytes) + b * 210;
			device const uchar *ql = blk + 64 * h + l0, *qh = blk + 128 + 32 * h + l0;
			device const char *sc = (device const char *)(blk + 192 + 8 * h);
			float s[4] = {0, 0, 0, 0};
			device const ushort *ql2 = (device const ushort *)ql, *qh2 = (device const ushort *)qh; // blocks are 2-aligned
			for (int k = 0; k < 2; k++) { // four bytes of each at once
				uint lo = as_type<uint>(ushort2(ql2[2 * k], ql2[2 * k + 1]));
				uint hi = as_type<uint>(ushort2(ql2[16 + 2 * k], ql2[17 + 2 * k]));
				uint hb = as_type<uint>(ushort2(qh2[2 * k], qh2[2 * k + 1]));
				float4 q0 = float4(as_type<uchar4>((lo & 0x0F0F0F0F) | ((hb & 0x03030303) << 4))) - 32;
				float4 q1 = float4(as_type<uchar4>((hi & 0x0F0F0F0F) | (((hb >> 2) & 0x03030303) << 4))) - 32;
				float4 q2 = float4(as_type<uchar4>(((lo >> 4) & 0x0F0F0F0F) | (((hb >> 4) & 0x03030303) << 4))) - 32;
				float4 q3 = float4(as_type<uchar4>(((hi >> 4) & 0x0F0F0F0F) | (((hb >> 6) & 0x03030303) << 4))) - 32;
				s[0] += dot(q0, xv[0][k]);
				s[1] += dot(q1, xv[1][k]);
				s[2] += dot(q2, xv[2][k]);
				s[3] += dot(q3, xv[3][k]);
			}
			sum[r] += float(*(device const half *)(blk + 208)) *
			          (float(sc[is]) * s[0] + float(sc[is + 2]) * s[1] + float(sc[is + 4]) * s[2] + float(sc[is + 6]) * s[3]);
		}
	}
	mv_store<PAIR>(sum, y, (ulong)tg.y * a.rows + r0, nr, lane);
}
template [[host_name("matvec_q6k_f32")]] kernel decltype(matvec_q6k<false>) matvec_q6k<false>;
template [[host_name("matvec_q6k_pair")]] kernel decltype(matvec_q6k<true>) matvec_q6k<true>;

// Same contract for F16 weights; a.nb counts 256-element steps. A simdgroup computes MV_ROWS rows; lane l takes
// elements 8l..8l+7 of each 256 (one 16-byte load per row), reusing its activations across the rows.
template <bool PAIR>
kernel void matvec_f16(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           device const uchar *w2 [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_OUT;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_OUT, a.rows - r0);
	device const float *xt = x + (ulong)tg.y * a.cols;
	float sum[MV_ROWS] = {};
	for (uint b = 0; b < a.nb; b++) {
		device const float4 *xb = (device const float4 *)(xt + b * 256 + 8 * lane);
		float4 x0 = xb[0], x1 = xb[1];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			device const half4 *wr = (device const half4 *)(mv_row<PAIR>(w, w2, r0, r, a.rows, a.rowBytes) + (b * 256 + 8 * lane) * 2);
			sum[r] += dot(float4(wr[0]), x0) + dot(float4(wr[1]), x1);
		}
	}
	mv_store<PAIR>(sum, y, (ulong)tg.y * a.rows + r0, nr, lane);
}
template [[host_name("matvec_f16_f32")]] kernel decltype(matvec_f16<false>) matvec_f16<false>;
template [[host_name("matvec_f16_pair")]] kernel decltype(matvec_f16<true>) matvec_f16<true>;

// Same contract for Q8_0 weights (34-byte blocks: d f16, qs[32] int8); a.nb counts 32-element blocks. A simdgroup
// computes MV_ROWS rows; four lanes share a block (lane groups take blocks g, g+8, ...), lane i of a group covering
// elements 8i..8i+7.
template <bool PAIR>
kernel void matvec_q8_0(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant MatvecArgs &a [[buffer(3)]],
                           device const uchar *w2 [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_OUT;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_OUT, a.rows - r0);
	device const float *xt = x + (ulong)tg.y * a.cols;
	uint e = (lane % 4) * 8;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 4; b < a.nb; b += 8) {
		device const float4 *xb = (device const float4 *)(xt + b * 32 + e);
		float4 x0 = xb[0], x1 = xb[1];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			device const uchar *blk = mv_row<PAIR>(w, w2, r0, r, a.rows, a.rowBytes) + b * 34;
			device const char2 *q = (device const char2 *)(blk + 2 + e); // blocks are 34 bytes: 2-byte aligned
			char2 q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
			sum[r] += float(*(device const half *)blk) *
			          (dot(float4(q0.x, q0.y, q1.x, q1.y), x0) + dot(float4(q2.x, q2.y, q3.x, q3.y), x1));
		}
	}
	mv_store<PAIR>(sum, y, (ulong)tg.y * a.rows + r0, nr, lane);
}
template [[host_name("matvec_q8_0_f32")]] kernel decltype(matvec_q8_0<false>) matvec_q8_0<false>;
template [[host_name("matvec_q8_0_pair")]] kernel decltype(matvec_q8_0<true>) matvec_q8_0<true>;

// In place is fine (out == x).
struct NormArgs {
	uint n;
	float eps;
};

kernel void rmsnorm_f32(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant NormArgs &a [[buffer(3)]],
                        threadgroup float *part [[threadgroup(0)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
	device const float *xr = x + (ulong)row * a.n;
	device float *orow = out + (ulong)row * a.n;
	float ss = 0;
	for (uint i = tid; i < a.n; i += nt) {
		ss += xr[i] * xr[i];
	}
	ss = simd_sum(ss);
	if (lane == 0) {
		part[sg] = ss;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (sg == 0) {
		float v = lane < nsg ? part[lane] : 0;
		v = simd_sum(v);
		if (lane == 0) {
			part[0] = v;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float scale = rsqrt(part[0] / float(a.n) + a.eps);
	for (uint i = tid; i < a.n; i += nt) {
		orow[i] = xr[i] * scale * w[i];
	}
}

// norm_rope_f32 is rmsnorm over each head (one threadgroup of headDim/2 threads per [token][head] row of x, in place)
// followed by rope_neox: two dispatches in one.
// With store set it also writes the result into a transposed f32 key cache kT ([heads*headDim][ctxP]) at position
// pos + token: the keys' normRope and storeKV in one dispatch.
struct NormRopeArgs {
	uint heads;
	uint headDim;
	float eps;
	uint store;
	uint ctxP;
	uint pos;
};

kernel void norm_rope_f32(device float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device const float *cs [[buffer(2)]],
                          device const float *sn [[buffer(3)]],
                          constant NormRopeArgs &a [[buffer(4)]],
                          device float *kT [[buffer(5)]],
                          threadgroup float *part [[threadgroup(0)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint i [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
	uint half_ = a.headDim / 2, t = row / a.heads;
	device float *v = x + (ulong)row * a.headDim;
	float x0 = v[i], x1 = v[i + half_];
	float ss = simd_sum(x0 * x0 + x1 * x1);
	if (lane == 0) {
		part[sg] = ss;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	ss = 0;
	for (uint k = 0; k < nsg; k++) {
		ss += part[k];
	}
	float scale = rsqrt(ss / float(a.headDim) + a.eps);
	x0 *= scale * w[i];
	x1 *= scale * w[i + half_];
	float c = cs[t * half_ + i], s = sn[t * half_ + i];
	float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
	v[i] = y0;
	v[i + half_] = y1;
	if (a.store) {
		device float *kr = kT + (ulong)(row % a.heads) * a.headDim * a.ctxP + a.pos + t;
		kr[(ulong)i * a.ctxP] = y0;
		kr[(ulong)(i + half_) * a.ctxP] = y1;
	}
}

// add_norm_f32 is x += y then out = rmsnorm(x) * w, per row of n (one threadgroup per row): add_f32 and rmsnorm_f32 in
// one dispatch.
kernel void add_norm_f32(device float *x [[buffer(0)]],
                         device const float *y [[buffer(1)]],
                         device const float *w [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         constant NormArgs &a [[buffer(4)]],
                         threadgroup float *part [[threadgroup(0)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
	device float *xr = x + (ulong)row * a.n;
	device const float *yr = y + (ulong)row * a.n;
	device float *orow = out + (ulong)row * a.n;
	float ss = 0;
	for (uint i = tid; i < a.n; i += nt) {
		float v = xr[i] + yr[i];
		xr[i] = v;
		ss += v * v;
	}
	ss = simd_sum(ss);
	if (lane == 0) {
		part[sg] = ss;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	ss = 0;
	for (uint k = 0; k < nsg; k++) {
		ss += part[k];
	}
	float scale = rsqrt(ss / float(a.n) + a.eps);
	for (uint i = tid; i < a.n; i += nt) {
		orow[i] = xr[i] * scale * w[i];
	}
}

// rope_neox rotates pairs (i, i+half) of every head in place: x is [tokens][heads][headDim],
// cs/sn are [tokens][half] (cos and sin of position*freq, computed on the CPU as forward does).
struct RopeArgs {
	uint heads;
	uint headDim;
};

kernel void rope_neox_f32(device float *x [[buffer(0)]],
                          device const float *cs [[buffer(1)]],
                          device const float *sn [[buffer(2)]],
                          constant RopeArgs &a [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]]) {
	uint half_ = a.headDim / 2;
	uint i = gid.x, h = gid.y, t = gid.z;
	if (i >= half_ || h >= a.heads) {
		return;
	}
	device float *v = x + ((ulong)t * a.heads + h) * a.headDim;
	float c = cs[t * half_ + i], s = sn[t * half_ + i];
	float x0 = v[i], x1 = v[i + half_];
	v[i] = x0 * c - x1 * s;
	v[i + half_] = x0 * s + x1 * c;
}

// swiglu: out[i] = silu(gate[i]) * up[i].
kernel void swiglu_f32(device const float *gate [[buffer(0)]],
                       device const float *up [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant uint &n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
	if (i < n) {
		float g = gate[i];
		out[i] = g / (1 + exp(-g)) * up[i];
	}
}

// add: x[i] += y[i] (the residual stream).
kernel void add_f32(device float *x [[buffer(0)]],
                    device const float *y [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
	if (i < n) {
		x[i] += y[i];
	}
}

#include <metal_simdgroup_matrix>

// matmul_q8_0_f32: y[t*rows + r] = dot(W row r, x[t]) for a batch of tokens (prompt reading).
// Each threadgroup computes a 64-row x 64-token tile with 8 simdgroups (32x16 each, as 4x2 8x8
// simdgroup matrices). The K loop steps one Q8_0 block (32) at a time: weights are dequantized and
// activations copied into threadgroup memory, then multiplied with simdgroup_multiply_accumulate.
// Grid: (rows/64, ceil(tokens/64)). a.nb counts 32-element steps (any weight type: dequant8); a.rows here is also the y stride.
struct MatmulArgs {
	uint rows;
	uint nb;
	uint rowBytes;
	uint cols;
	uint tokens;
};

// dequant8 writes 8 consecutive weights of a row, starting at element k (a multiple of 8), to dst as halves.
// typ: 0 Q8_0, 1 Q4_K, 2 Q6_K, 3 F16.
static inline void dequant8(device const uchar *row, uint k, threadgroup half *dst, const int typ) {
	if (typ == 0) {
		device const uchar *blk = row + (k / 32) * 34;
		float d = float(*(device const half *)blk);
		device const char *qs = (device const char *)(blk + 2 + k % 32);
		for (int i = 0; i < 8; i++) {
			dst[i] = half(d * float(qs[i]));
		}
	} else if (typ == 1) {
		device const uchar *blk = row + (k / 256) * 144;
		uint e = k % 256, j = e / 64, hi = (e % 64) / 32, l = e % 32;
		uchar2 sm = scale_min_k4(2 * j + hi, blk + 4);
		float d = float(*(device const half *)blk) * sm.x, m = float(*(device const half *)(blk + 2)) * sm.y;
		device const uchar *qs = blk + 16 + 32 * j + l;
		for (int i = 0; i < 8; i++) {
			dst[i] = half(d * float(hi ? qs[i] >> 4 : qs[i] & 0xF) - m);
		}
	} else if (typ == 3) {
		device const half *p = (device const half *)row + k;
		for (int i = 0; i < 8; i++) {
			dst[i] = p[i];
		}
	} else {
		device const uchar *blk = row + (k / 256) * 210;
		uint e = k % 256, h = e / 128, quarter = (e % 128) / 32, l = e % 32;
		device const uchar *ql = blk + 64 * h + l + (quarter & 1) * 32, *qh = blk + 128 + 32 * h + l;
		float d = float(*(device const half *)(blk + 208)) * float(((device const char *)(blk + 192 + 8 * h))[l / 16 + 2 * quarter]);
		uint shift = quarter >= 2 ? 4 : 0;
		for (int i = 0; i < 8; i++) {
			dst[i] = half(d * float(int(((ql[i] >> shift) & 0xF) | (((qh[i] >> (2 * quarter)) & 3) << 4)) - 32));
		}
	}
}

// Threadgroup row strides of the tiled matmul: the weight tile's 32 k and the activation tile's 64 tokens, padded so
// a fragment's rows use other banks.
#define MM_LD 36
#define MM_XLD 68

// matmul_tile: the tiles are halves (half the threadgroup memory, and an M1 Max measured faster with them; sums stay
// float), and the activations are staged transposed ([k][token]) so their fragments load without a transpose, which
// measured 4-5% faster on a 512-token prompt than f32 [token][k] tiles. Fragments are stored straight to y.
static inline void matmul_tile(device const uchar *w, device const float *x, device float *y, constant MatmulArgs &a,
                               threadgroup float *shm, uint2 tg, uint tid, uint sg, uint lane, const int typ) {
	threadgroup half *sw = (threadgroup half *)shm; // [64 rows][MM_LD]
	threadgroup half *sx = sw + 64 * MM_LD;         // [32 k][MM_XLD tokens]
	uint r0 = tg.x * 64, t0 = tg.y * 64;
	uint sr = (sg % 2) * 32, st = (sg / 2) * 16; // this simdgroup's 32x16 sub-tile (8 simdgroups)

	simdgroup_float8x8 c[4][2];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++) {
			c[i][j] = simdgroup_float8x8(0);
		}
	}
	// 256 threads: thread tid loads 8 consecutive k of row/token tid/4.
	uint lr = tid / 4, lk = (tid % 4) * 8;
	for (uint b = 0; b < a.nb; b++) {
		uint r = r0 + lr, t = t0 + lr;
		threadgroup half *dst = sw + lr * MM_LD + lk;
		if (r < a.rows) {
			dequant8(w + (ulong)r * a.rowBytes, b * 32 + lk, dst, typ);
		} else {
			for (int i = 0; i < 8; i++) {
				dst[i] = 0;
			}
		}
		threadgroup half *dx = sx + lk * MM_XLD + lr;
		if (t < a.tokens) {
			device const float4 *src = (device const float4 *)(x + (ulong)t * a.cols + b * 32 + lk);
			float4 x0 = src[0], x1 = src[1];
			for (int i = 0; i < 4; i++) {
				dx[i * MM_XLD] = half(x0[i]);
				dx[(4 + i) * MM_XLD] = half(x1[i]);
			}
		} else {
			for (int i = 0; i < 8; i++) {
				dx[i * MM_XLD] = 0;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint kk = 0; kk < 32; kk += 8) {
			simdgroup_half8x8 ma[4], mb[2];
			for (int i = 0; i < 4; i++) {
				simdgroup_load(ma[i], sw + (sr + 8 * i) * MM_LD + kk, MM_LD);
			}
			for (int j = 0; j < 2; j++) {
				simdgroup_load(mb[j], sx + kk * MM_XLD + st + 8 * j, MM_XLD);
			}
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 2; j++) {
					simdgroup_multiply_accumulate(c[i][j], ma[i], mb[j], c[i][j]);
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	// c[i][j] is [rows][tokens]: store each fragment transposed into y ([token][row]); edge fragments go through 64
	// floats per simdgroup and are written in bounds.
	threadgroup float *f = shm + sg * 64;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++) {
			uint r = r0 + sr + 8 * i, t = t0 + st + 8 * j;
			if (r + 8 <= a.rows && t + 8 <= a.tokens) {
				simdgroup_store(c[i][j], y + (ulong)t * a.rows + r, a.rows, ulong2(0, 0), true);
				continue;
			}
			simdgroup_store(c[i][j], f, 8, ulong2(0, 0), true); // f[token][row]
			simdgroup_barrier(mem_flags::mem_threadgroup);
			for (uint e = lane; e < 64; e += 32) {
				if (t + e / 8 < a.tokens && r + e % 8 < a.rows) {
					y[(ulong)(t + e / 8) * a.rows + r + e % 8] = f[e];
				}
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
		}
	}
}

#define MATMUL_KERNEL(name, typ)                                                                                     \
	kernel void name(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],                   \
	                 device float *y [[buffer(2)]], constant MatmulArgs &a [[buffer(3)]],                        \
	                 threadgroup float *shm [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],       \
	                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],       \
	                 uint lane [[thread_index_in_simdgroup]]) {                                                  \
		matmul_tile(w, x, y, a, shm, tg, tid, sg, lane, typ);                                                    \
	}
MATMUL_KERNEL(matmul_q8_0_f32, 0)
MATMUL_KERNEL(matmul_q4k_f32, 1)
MATMUL_KERNEL(matmul_q6k_f32, 2)
MATMUL_KERNEL(matmul_f16_f32, 3)

// attention_f32: causal attention over the f32 KV cache, one threadgroup per (token, query head).
// q and out are [tokens][heads][headDim]; kT is one layer's keys transposed, [kvHeads][headDim][ctxP];
// v is [ctxP][kvHeads*headDim]. Token t sits at position pos+t and sees positions [0, pos+t], minus the
// masked range [mask[t].x, mask[t].y) (packed sequences hide each other; {0,0} hides nothing).
// sc is scratch for the scores, [tokens*heads][ctxP]. Threadgroup memory: headDim + 2*threads floats.
struct AttnArgs {
	uint heads;
	uint kvHeads;
	uint headDim;
	uint ctxP;
	uint pos;
	float scale;
	uint stageStart; // block KV caches: positions from here are in the stage (KVQ)
};

// KV cache accessors for the attention kernels: key(j, d) and val(j, d) are channel d of KV head kvh at position j.
// KVF32 is the f32 cache (keys transposed [kvHeads*headDim][ctxP], values by position [ctxP][kvd]); KVF16 is the
// f16 cache in the CPU's compact layout (keys by position [ctxP][kvd], values by channel [kvd][ctxP]).
struct KVF32 {
	typedef float T;
	device const float *k, *v, *stage;
	uint ctxP, kvd, hd, kvh, stageStart;
	static constant constexpr bool byChannel = false; // values are stored by position
	float key(uint j, uint d) const { return k[(ulong)(kvh * hd + d) * ctxP + j]; }
	float4 key4(uint j, uint d) const { return float4(key(j, d), key(j, d + 1), key(j, d + 2), key(j, d + 3)); }
	float val(uint j, uint d) const { return v[(ulong)j * kvd + kvh * hd + d]; }
};
struct KVF16 {
	typedef half T;
	device const half *k, *v;
	device const float *stage;
	uint ctxP, kvd, hd, kvh, stageStart;
	static constant constexpr bool byChannel = true; // values are stored by channel
	float key(uint j, uint d) const { return k[(ulong)j * kvd + kvh * hd + d]; }
	float4 key4(uint j, uint d) const { return float4(*(device const half4 *)(k + (ulong)j * kvd + kvh * hd + d)); } // d % 4 == 0
	float val(uint j, uint d) const { return v[(ulong)(kvh * hd + d) * ctxP + j]; }
};
// KVQ is a block cache in the CPU's compact layout (kvcache.go): keys by position, each KV head's headDim numbers in
// q8_0 blocks of 32; values by channel, positions in blocks of 32 (VB 8: q8_0, 34 bytes; VB 4: q4_0, 18 bytes); the
// newest block's values at full precision in stage ([kvd][32] floats, positions from stageStart).
template <int VB>
struct KVQ {
	typedef uchar T;
	device const uchar *k, *v;
	device const float *stage;
	uint ctxP, kvd, hd, kvh, stageStart;
	static constant constexpr bool byChannel = true;
	// a position's row is kvd/32 blocks (hd is a multiple of 32): no runtime division per key
	device const uchar *kblk(uint j, uint d) const { return k + (ulong)j * (kvd / 32 * 34) + (kvh * hd + d) / 32 * 34; }
	float key(uint j, uint d) const {
		device const uchar *b = kblk(j, d);
		return float(*(device const half *)b) * float(((device const char *)b)[2 + d % 32]);
	}
	float4 key4(uint j, uint d) const { // d % 4 == 0; quants are 2-byte aligned
		device const uchar *b = kblk(j, d);
		device const char2 *q = (device const char2 *)(b + 2 + d % 32);
		char2 x = q[0], y = q[1];
		return float(*(device const half *)b) * float4(x.x, x.y, y.x, y.y);
	}
	float val(uint j, uint d) const {
		uint ch = kvh * hd + d;
		if (j >= stageStart) {
			return stage[ch * 32 + j - stageStart];
		}
		const uint bb = VB == 8 ? 34 : 18;
		device const uchar *b = v + (ulong)ch * (ctxP / 32 * bb) + j / 32 * bb;
		float sc = float(*(device const half *)b);
		uint jj = j % 32;
		if (VB == 8) {
			return sc * float(((device const char *)b)[2 + jj]);
		}
		uint q = b[2 + jj % 16];
		return sc * (float(jj < 16 ? q & 0xF : q >> 4) - 8);
	}
};
typedef KVQ<8> KVQ8;
typedef KVQ<4> KVQ4;


// attn_scores writes the scaled scores of positions [c0, c1) against the head's query qs into ss[j - c0], -INFINITY for
// masked positions. Lanes take consecutive positions (coalesced key loads), simdgroups split the head's channels, and
// red (nsg*32 floats) sums their parts: short dependent chains instead of one thread looping over every channel, which
// is what short contexts need (the one-pass kernel). The split kernel's chunks keep a position per thread: measured
// faster there.
template <typename P, typename KV>
static inline void attn_scores(threadgroup const float *qs, KV kv, uint hd, uint c0, uint c1,
                               uint2 m, float scale, P ss, threadgroup float *red, uint sg, uint nsg, uint lane) {
	uint dper = (hd + nsg - 1) / nsg, d0 = sg * dper, d1 = min(hd, d0 + dper);
	for (uint jb = c0; jb < c1; jb += 32) {
		uint j = jb + lane;
		float x = 0;
		if (j < c1) {
			for (uint d = d0; d < d1; d++) {
				x += qs[d] * kv.key(j, d);
			}
		}
		red[sg * 32 + lane] = x;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (sg == 0 && j < c1) {
			float tot = 0;
			for (uint i = 0; i < nsg; i++) {
				tot += red[i * 32 + lane];
			}
			ss[j - c0] = (j < m.x || j >= m.y) ? tot * scale : -INFINITY;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
	}
}

template <typename KV>
kernel void attention_kv(device const float *q [[buffer(0)]],
                          device const typename KV::T *kT [[buffer(1)]],
                          device const typename KV::T *v [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          device float *sc [[buffer(4)]],
                          device const uint2 *mask [[buffer(5)]],
                          constant AttnArgs &a [[buffer(6)]],
                          device const float *stage [[buffer(7)]],
                          threadgroup float *shm [[threadgroup(0)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 ntg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
	uint nt = ntg.x;
	uint h = tg.x, t = tg.y;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint ctx = a.pos + t + 1;
	uint2 m = mask[t];
	threadgroup float *qs = shm;           // [headDim]
	threadgroup float *red = shm + hd;     // [threads] reduction scratch
	device float *s = sc + ((ulong)t * a.heads + h) * a.ctxP;
	device const float *qh = q + ((ulong)t * a.heads + h) * hd;
	for (uint d = tid; d < hd; d += nt) {
		qs[d] = qh[d];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	// scores and their max
	KV kv{kT, v, stage, a.ctxP, kvd, hd, kvh, a.stageStart};
	attn_scores(qs, kv, hd, 0, ctx, m, a.scale, s, red, sg, nsg, lane);
	float mx = -INFINITY;
	for (uint j = tid; j < ctx; j += nt) {
		mx = max(mx, s[j]);
	}
	mx = simd_max(mx);
	if (lane == 0) {
		red[sg] = mx;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	mx = red[0];
	for (uint i = 1; i < nsg; i++) {
		mx = max(mx, red[i]);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	// softmax numerators and their sum
	float sum = 0;
	for (uint j = tid; j < ctx; j += nt) {
		float e = exp(s[j] - mx);
		s[j] = e;
		sum += e;
	}
	sum = simd_sum(sum);
	if (lane == 0) {
		red[sg] = sum;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	sum = 0;
	for (uint i = 0; i < nsg; i++) {
		sum += red[i];
	}
	float inv = 1 / sum;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	if (KV::byChannel) { // values by channel: a simdgroup per channel, lanes over positions (as the split kernel)
		for (uint d = sg; d < hd; d += nsg) {
			float o = 0;
			for (uint j = lane; j < ctx; j += 32) {
				o += s[j] * kv.val(j, d);
			}
			o = simd_sum(o);
			if (lane == 0) {
				out[((ulong)t * a.heads + h) * hd + d] = o * inv;
			}
		}
		return;
	}
	// weighted sum of values: thread (d, part) sums positions part, part+parts, ... for channel d
	uint parts = nt / hd, d = tid % hd, part = tid / hd;
	float o = 0;
	if (part < parts) {
		for (uint j = part; j < ctx; j += parts) {
			o += s[j] * kv.val(j, d);
		}
	}
	red[tid] = o;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < hd) {
		float acc = 0;
		for (uint p = 0; p < parts; p++) {
			acc += red[p * hd + tid];
		}
		out[((ulong)t * a.heads + h) * hd + tid] = acc * inv;
	}
}

// kv_store_f32 writes n tokens' keys and values into one layer's f32 cache at positions pos..:
// keys transposed (kT[i*ctxP + p], i = kvHead*headDim + d), values by row (vc[p*kvd + i]).
struct KVStoreArgs {
	uint kvd;
	uint ctxP;
	uint pos;
};

kernel void kv_store_f32(device const float *k [[buffer(0)]],
                         device const float *v [[buffer(1)]],
                         device float *kT [[buffer(2)]],
                         device float *vc [[buffer(3)]],
                         constant KVStoreArgs &a [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
	uint i = gid.x, t = gid.y;
	if (i >= a.kvd) {
		return;
	}
	kT[(ulong)i * a.ctxP + a.pos + t] = k[(ulong)t * a.kvd + i];
	vc[((ulong)a.pos + t) * a.kvd + i] = v[(ulong)t * a.kvd + i];
}

// copy: dst[i] = src[i].
kernel void copy_f32(device float *dst [[buffer(0)]],
                     device const float *src [[buffer(1)]],
                     constant uint &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
	if (i < n) {
		dst[i] = src[i];
	}
}

// attention_split_f32 is attention for a few tokens over a long context ("flash decoding"): threadgroup
// (head, token, split) covers positions [split*chunk, split*chunk+chunk) and writes a partial result,
// part[((t*heads + h)*splits + split)*(headDim+2)] = {max score, sum of exp, unnormalized output[headDim]}.
// attention_combine_f32 merges the splits. Threadgroup memory: headDim + chunk + simdgroups*headDim floats.
struct AttnSplitArgs {
	uint heads;
	uint kvHeads;
	uint headDim;
	uint ctxP;
	uint pos;
	float scale;
	uint chunk;
	uint splits;
	uint stageStart; // block KV caches: positions from here are in the stage (KVQ)
};

template <typename KV>
kernel void attention_split_kv(device const float *q [[buffer(0)]],
                                device const typename KV::T *kT [[buffer(1)]],
                                device const typename KV::T *v [[buffer(2)]],
                                device float *part [[buffer(3)]],
                                device const uint2 *mask [[buffer(4)]],
                                constant AttnSplitArgs &a [[buffer(5)]],
                                device const float *stage [[buffer(6)]],
                                threadgroup float *shm [[threadgroup(0)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint3 ntg [[threads_per_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint nsg [[simdgroups_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
	uint nt = ntg.x;
	uint h = tg.x, t = tg.y, sp = tg.z;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint ctx = a.pos + t + 1, c0 = sp * a.chunk, c1 = min(ctx, c0 + a.chunk);
	uint2 m = mask[t];
	device float *P = part + (((ulong)t * a.heads + h) * a.splits + sp) * (hd + 2);
	threadgroup float *qs = shm, *ss = shm + hd, *red = shm + hd + a.chunk;
	if (c0 >= ctx) { // past this token's context
		for (uint d = tid; d < hd + 2; d += nt) {
			P[d] = d == 0 ? -INFINITY : 0;
		}
		return;
	}
	device const float *qh = q + ((ulong)t * a.heads + h) * hd;
	for (uint d = tid; d < hd; d += nt) {
		qs[d] = qh[d];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	KV kv{kT, v, stage, a.ctxP, kvd, hd, kvh, a.stageStart};
	float mx = -INFINITY;
	for (uint j = c0 + tid; j < c1; j += nt) {
		float x = -INFINITY;
		if (j < m.x || j >= m.y) {
			x = 0;
			for (uint d = 0; d < hd; d += 4) { // headDim % 4 == 0
				x += dot(*(threadgroup const float4 *)(qs + d), kv.key4(j, d));
			}
			x *= a.scale;
		}
		ss[j - c0] = x;
		mx = max(mx, x);
	}
	mx = simd_max(mx);
	if (lane == 0) {
		red[sg] = mx;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	mx = red[0];
	for (uint i = 1; i < nsg; i++) {
		mx = max(mx, red[i]);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (mx == -INFINITY) { // every position here is masked
		for (uint d = tid; d < hd + 2; d += nt) {
			P[d] = d == 0 ? -INFINITY : 0;
		}
		return;
	}
	float sum = 0;
	for (uint j = c0 + tid; j < c1; j += nt) {
		float e = exp(ss[j - c0] - mx);
		ss[j - c0] = e;
		sum += e;
	}
	sum = simd_sum(sum);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (lane == 0) {
		red[sg] = sum;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	sum = 0;
	for (uint i = 0; i < nsg; i++) {
		sum += red[i];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	if (KV::byChannel) {
		// values by channel: simdgroup sg takes channels sg, sg+nsg, ...; its lanes walk the channel's row (consecutive
		// positions, coalesced: one block per step for block caches), and simd_sum adds them. Lanes over channels
		// instead read 32 rows per step, several times slower over a long context.
		for (uint d = sg; d < hd; d += nsg) {
			float o = 0;
			for (uint j = c0 + lane; j < c1; j += 32) {
				o += ss[j - c0] * kv.val(j, d);
			}
			o = simd_sum(o);
			if (lane == 0) {
				P[2 + d] = o;
			}
		}
	} else {
		// values by position: simdgroup sg sums positions c0+sg, c0+sg+nsg, ...; lane covers channels lane, lane+32, ...
		float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // headDim <= 256
		for (uint j = c0 + sg; j < c1; j += nsg) {
			float w = ss[j - c0];
			for (uint i = 0, d = lane; d < hd; i++, d += 32) {
				acc[i] += w * kv.val(j, d);
			}
		}
		for (uint i = 0, d = lane; d < hd; i++, d += 32) {
			red[sg * hd + d] = acc[i];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint d = tid; d < hd; d += nt) {
			float o = 0;
			for (uint i = 0; i < nsg; i++) {
				o += red[i * hd + d];
			}
			P[2 + d] = o;
		}
	}
	if (tid == 0) {
		P[0] = mx;
		P[1] = sum;
	}
}

// attention_combine_f32 merges attention_split_f32's partial results; one threadgroup (headDim threads) per
// (head, token).
kernel void attention_combine_f32(device const float *part [[buffer(0)]],
                                  device float *out [[buffer(1)]],
                                  constant AttnSplitArgs &a [[buffer(2)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
	uint h = tg.x, t = tg.y, hd = a.headDim;
	if (tid >= hd) {
		return;
	}
	device const float *P = part + ((ulong)t * a.heads + h) * a.splits * (hd + 2);
	float M = -INFINITY;
	for (uint s = 0; s < a.splits; s++) {
		M = max(M, P[s * (hd + 2)]);
	}
	float num = 0, den = 0;
	for (uint s = 0; s < a.splits; s++) {
		device const float *p = P + s * (hd + 2);
		if (p[0] != -INFINITY) {
			float w = exp(p[0] - M);
			den += w * p[1];
			num += w * p[2 + tid];
		}
	}
	out[((ulong)t * a.heads + h) * hd + tid] = num / den;
}

// attention_tile_f32: causal attention for prompts, tiled ("flash attention", f32). A threadgroup of 256 threads
// takes one query head and ATT_QT consecutive tokens and walks the keys in chunks of ATT_KC positions, staging each
// chunk's keys and values in threadgroup memory once for all its tokens, with an online softmax. Same layouts as
// attention_f32 (no scratch); headDim <= 128. Grid (heads, ceil(tokens/ATT_QT)).
#define ATT_QT 16
#define ATT_KC 16
struct AttnTileArgs {
	uint heads;
	uint kvHeads;
	uint headDim;
	uint ctxP;
	uint pos;
	float scale;
	uint tokens;
	uint stageStart; // block KV caches: positions from here are in the stage (KVQ)
};

template <typename KV>
kernel void attention_tile_kv(device const float *q [[buffer(0)]],
                               device const typename KV::T *kT [[buffer(1)]],
                               device const typename KV::T *v [[buffer(2)]],
                               device float *out [[buffer(3)]],
                               device const uint2 *mask [[buffer(4)]],
                               constant AttnTileArgs &a [[buffer(5)]],
                               device const float *stage [[buffer(6)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
	threadgroup float qs[ATT_QT][128], ks[128][ATT_KC + 1], vs[ATT_KC][128], ps[ATT_QT][ATT_KC + 1];
	threadgroup float alpha[ATT_QT], lsum[ATT_QT], mrow[ATT_QT];
	uint h = tg.x, t0 = tg.y * ATT_QT;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint nt = min((uint)ATT_QT, a.tokens - t0);
	for (uint i = tid; i < ATT_QT * hd; i += 256) {
		uint t = i / hd, d = i % hd;
		qs[t][d] = t < nt ? q[((ulong)(t0 + t) * a.heads + h) * hd + d] * a.scale : 0;
	}
	if (tid < ATT_QT) {
		mrow[tid] = -INFINITY;
		lsum[tid] = 0;
	}
	uint orow = tid / 16, ocol = tid % 16; // this thread's output: row orow, channels ocol + 16*i
	float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	uint ctxEnd = a.pos + t0 + nt;
	KV kv{kT, v, stage, a.ctxP, kvd, hd, kvh, a.stageStart};
	for (uint c0 = 0; c0 < ctxEnd; c0 += ATT_KC) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint i = tid; i < hd * ATT_KC; i += 256) {
			uint d = i / ATT_KC, j = i % ATT_KC;
			ks[d][j] = c0 + j < ctxEnd ? kv.key(c0 + j, d) : 0;
		}
		for (uint i = tid; i < ATT_KC * hd; i += 256) {
			uint j = i / hd, d = i % hd;
			vs[j][d] = c0 + j < ctxEnd ? kv.val(c0 + j, d) : 0;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		{ // scores: 256 = ATT_QT * ATT_KC, one per thread
			uint t = tid / ATT_KC, j = tid % ATT_KC, p = c0 + j;
			uint2 m = mask[t0 + min(t, nt - 1)];
			float s = -INFINITY;
			if (t < nt && p <= a.pos + t0 + t && (p < m.x || p >= m.y)) {
				s = 0;
				for (uint d = 0; d < hd; d++) {
					s += qs[t][d] * ks[d][j];
				}
			}
			ps[t][j] = s;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid < ATT_QT) { // online softmax, one thread per token
			float mx = mrow[tid];
			for (int j = 0; j < ATT_KC; j++) {
				mx = max(mx, ps[tid][j]);
			}
			float al = mx == -INFINITY ? 1 : exp(mrow[tid] - mx), l = lsum[tid] * al;
			for (int j = 0; j < ATT_KC; j++) {
				float e = mx == -INFINITY ? 0 : exp(ps[tid][j] - mx);
				ps[tid][j] = e;
				l += e;
			}
			alpha[tid] = al;
			mrow[tid] = mx;
			lsum[tid] = l;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		float al = alpha[orow];
		for (int i = 0; i < 8; i++) {
			o[i] *= al;
		}
		for (int j = 0; j < ATT_KC; j++) {
			float pj = ps[orow][j];
			for (int i = 0; i < 8; i++) {
				uint d = ocol + 16 * i;
				if (d < hd) {
					o[i] += pj * vs[j][d];
				}
			}
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (orow < nt) {
		float inv = 1 / lsum[orow];
		for (int i = 0; i < 8; i++) {
			uint d = ocol + 16 * i;
			if (d < hd) {
				out[((ulong)(t0 + orow) * a.heads + h) * hd + d] = o[i] * inv;
			}
		}
	}
}

#define FA_HD 128
// attention_flash: prompt attention on the matrix units (headDim 128). A threadgroup of 8 simdgroups takes 64 tokens
// of one head; each simdgroup keeps its 8 query rows and their output accumulator as 16 8x8 fragments each, and walks
// the context 32 positions at a time: S = Q K^T, an online softmax over S's rows (four lanes per row; the scale goes
// on the scores), O = diag(alpha) O, then O += P V. K and V stage through threadgroup memory as halves; Q's fragments
// load straight from q and the output is stored straight to out (edge tiles go through threadgroup memory, so
// partial tiles never touch past the buffers). Against the earlier 32-token, 16-position, all-f32 kernel this measured
// 1.5x (512 tokens from position 0) to 1.8x (at position 1536) faster on an M1 Max. Grid: (heads, tokens/64), 256
// threads.
#define FB_KC 32
template <typename KV, uint NSG = 8>
kernel void attention_flash_kv(device const float *q [[buffer(0)]],
                                 device const typename KV::T *kT [[buffer(1)]],
                                 device const typename KV::T *v [[buffer(2)]],
                                 device float *out [[buffer(3)]],
                                 device const uint2 *mask [[buffer(4)]],
                                 constant AttnTileArgs &a [[buffer(5)]],
                                 device const float *stage [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
	threadgroup half ks[FA_HD * FB_KC]; // [128][32]
	threadgroup half vs[FB_KC * FA_HD]; // [32][128]
	threadgroup float ss[NSG][8][FB_KC];  // each simdgroup's scores, then probabilities; also edge staging
	threadgroup float dg[NSG][8][8];      // each simdgroup's diagonal row factors
	const uint QT = 8 * NSG, NT = 32 * NSG; // tokens and threads per threadgroup
	uint h = tg.x, t0 = tg.y * QT;
	uint kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * FA_HD;
	uint nt = min(QT, a.tokens - t0);
	uint qs = a.heads * FA_HD; // row stride of q and out
	uint row0 = t0 + sg * 8;   // this simdgroup's first token
	bool full = row0 + 8 <= a.tokens;
	threadgroup float *sc = &ss[sg][0][0];
	for (uint i = lane; i < 64; i += 32) {
		((threadgroup float *)dg[sg])[i] = 0;
	}
	simdgroup_float8x8 qf[16], of[16];
	for (int k = 0; k < 16; k++) {
		if (full) {
			simdgroup_load(qf[k], q + (ulong)row0 * qs + h * FA_HD + k * 8, qs);
		} else {
			for (uint e = lane; e < 64; e += 32) {
				uint t = row0 + e / 8;
				sc[e] = t < a.tokens ? q[(ulong)t * qs + h * FA_HD + k * 8 + e % 8] : 0;
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
			simdgroup_load(qf[k], sc, 8);
			simdgroup_barrier(mem_flags::mem_threadgroup);
		}
		of[k] = simdgroup_float8x8(0);
	}

	uint r = lane / 4, cq = (lane % 4) * 8; // this lane's row of the simdgroup's 8 and its 8 of the 32 columns
	uint trow = sg * 8 + r;
	bool valid = trow < nt;
	uint tpos = a.pos + t0 + trow;
	uint2 m = mask[t0 + min(trow, nt - 1)];
	float mrun = -INFINITY, lrun = 0;
	uint ctxEnd = a.pos + t0 + nt;
	KV cache{kT, v, stage, a.ctxP, kvd, FA_HD, kvh, a.stageStart};
	for (uint c0 = 0; c0 < ctxEnd; c0 += FB_KC) {
		// stage in the order the cache stores (coalesced reads): the f32 cache has keys by channel and values by
		// position, the compact caches the other way round
		for (uint i = tid; i < FA_HD * FB_KC; i += NT) {
			uint d = KV::byChannel ? i % FA_HD : i / FB_KC, j = KV::byChannel ? i / FA_HD : i % FB_KC;
			ks[d * FB_KC + j] = half(c0 + j < ctxEnd ? cache.key(c0 + j, d) : 0);
		}
		for (uint i = tid; i < FB_KC * FA_HD; i += NT) {
			uint j = KV::byChannel ? i % FB_KC : i / FA_HD, d = KV::byChannel ? i / FB_KC : i % FA_HD;
			vs[j * FA_HD + d] = half(c0 + j < ctxEnd ? cache.val(c0 + j, d) : 0);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0)};
		for (int k = 0; k < 16; k++) {
			for (int c = 0; c < 4; c++) {
				simdgroup_half8x8 b;
				simdgroup_load(b, ks + k * 8 * FB_KC + 8 * c, FB_KC);
				simdgroup_multiply_accumulate(s[c], qf[k], b, s[c]);
			}
		}
		for (int c = 0; c < 4; c++) {
			simdgroup_store(s[c], sc + 8 * c, FB_KC);
		}
		simdgroup_barrier(mem_flags::mem_threadgroup);

		float sv[8], mx = -INFINITY;
		for (int i = 0; i < 8; i++) {
			uint p = c0 + cq + i;
			bool ok = valid && p <= tpos && (p < m.x || p >= m.y);
			sv[i] = ok ? sc[r * FB_KC + cq + i] * a.scale : -INFINITY;
			mx = max(mx, sv[i]);
		}
		mx = max(mx, simd_shuffle_xor(mx, 1));
		mx = max(mx, simd_shuffle_xor(mx, 2));
		float mnew = max(mrun, mx);
		float al = mnew == -INFINITY ? 1 : exp(mrun - mnew), ls = 0;
		for (int i = 0; i < 8; i++) {
			float e = mnew == -INFINITY ? 0 : exp(sv[i] - mnew);
			sc[r * FB_KC + cq + i] = e;
			ls += e;
		}
		ls += simd_shuffle_xor(ls, 1);
		ls += simd_shuffle_xor(ls, 2);
		lrun = lrun * al + ls;
		mrun = mnew;
		if (lane % 4 == 0) {
			dg[sg][r][r] = al;
		}
		simdgroup_barrier(mem_flags::mem_threadgroup);

		simdgroup_float8x8 D, pf[4];
		simdgroup_load(D, &dg[sg][0][0], 8);
		for (int c = 0; c < 4; c++) {
			simdgroup_load(pf[c], sc + 8 * c, FB_KC);
		}
		for (int k = 0; k < 16; k++) {
			simdgroup_float8x8 t;
			simdgroup_multiply(t, D, of[k]);
			for (int c = 0; c < 4; c++) {
				simdgroup_half8x8 vf;
				simdgroup_load(vf, vs + 8 * c * FA_HD + k * 8, FA_HD);
				simdgroup_multiply_accumulate(t, pf[c], vf, t);
			}
			of[k] = t;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if (lane % 4 == 0) {
		dg[sg][r][r] = lrun > 0 ? 1 / lrun : 0;
	}
	simdgroup_barrier(mem_flags::mem_threadgroup);
	simdgroup_float8x8 D;
	simdgroup_load(D, &dg[sg][0][0], 8);
	for (int k = 0; k < 16; k++) {
		simdgroup_float8x8 t;
		simdgroup_multiply(t, D, of[k]);
		if (full) {
			simdgroup_store(t, out + (ulong)row0 * qs + h * FA_HD + k * 8, qs);
			continue;
		}
		simdgroup_store(t, sc, 8);
		simdgroup_barrier(mem_flags::mem_threadgroup);
		for (uint e = lane; e < 64; e += 32) {
			uint tt = row0 + e / 8;
			if (tt < a.tokens) {
				out[(ulong)tt * qs + h * FA_HD + k * 8 + e % 8] = sc[e];
			}
		}
		simdgroup_barrier(mem_flags::mem_threadgroup);
	}
}

// Attention kernels for each KV cache type.
template [[host_name("attention_f32")]] kernel decltype(attention_kv<KVF32>) attention_kv<KVF32>;
template [[host_name("attention_f16")]] kernel decltype(attention_kv<KVF16>) attention_kv<KVF16>;
template [[host_name("attention_split_f32")]] kernel decltype(attention_split_kv<KVF32>) attention_split_kv<KVF32>;
template [[host_name("attention_split_f16")]] kernel decltype(attention_split_kv<KVF16>) attention_split_kv<KVF16>;
template [[host_name("attention_tile_f32")]] kernel decltype(attention_tile_kv<KVF32>) attention_tile_kv<KVF32>;
template [[host_name("attention_tile_f16")]] kernel decltype(attention_tile_kv<KVF16>) attention_tile_kv<KVF16>;
template [[host_name("attention_flash_f32")]] kernel decltype(attention_flash_kv<KVF32>) attention_flash_kv<KVF32>;
template [[host_name("attention_flash_f16")]] kernel decltype(attention_flash_kv<KVF16>) attention_flash_kv<KVF16>;
template [[host_name("attention_q8")]] kernel decltype(attention_kv<KVQ8>) attention_kv<KVQ8>;
template [[host_name("attention_q4")]] kernel decltype(attention_kv<KVQ4>) attention_kv<KVQ4>;
template [[host_name("attention_split_q8")]] kernel decltype(attention_split_kv<KVQ8>) attention_split_kv<KVQ8>;
template [[host_name("attention_split_q4")]] kernel decltype(attention_split_kv<KVQ4>) attention_split_kv<KVQ4>;
template [[host_name("attention_tile_q8")]] kernel decltype(attention_tile_kv<KVQ8>) attention_tile_kv<KVQ8>;
template [[host_name("attention_tile_q4")]] kernel decltype(attention_tile_kv<KVQ4>) attention_tile_kv<KVQ4>;
template [[host_name("attention_flash_q8")]] kernel decltype(attention_flash_kv<KVQ8>) attention_flash_kv<KVQ8>;
template [[host_name("attention_flash_q4")]] kernel decltype(attention_flash_kv<KVQ4>) attention_flash_kv<KVQ4>;

// kv_store_f16 writes n tokens' keys and values ([n][kvd] floats) into a layer's f16 cache at positions pos..: keys
// by position (kc[p*kvd + i]), values by channel (vc[i*ctxP + p]). Grid (kvd/256, tokens), 256 threads.
kernel void kv_store_f16(device const float *k [[buffer(0)]],
                         device const float *v [[buffer(1)]],
                         device half *kc [[buffer(2)]],
                         device half *vc [[buffer(3)]],
                         constant KVStoreArgs &a [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
	uint i = gid.x, t = gid.y;
	if (i >= a.kvd) {
		return;
	}
	kc[((ulong)a.pos + t) * a.kvd + i] = half(clamp(k[(ulong)t * a.kvd + i], -65504.f, 65504.f));
	vc[(ulong)i * a.ctxP + a.pos + t] = half(clamp(v[(ulong)t * a.kvd + i], -65504.f, 65504.f));
}

// Block KV caches (q8_0 keys; q8_0 or q4_0 values): kv_store_qk quantizes n tokens' keys into their position rows,
// kv_store_qv puts their values into the channel rows' blocks and the stage, as the CPU's storeKey/storeValue do.
struct KVQStoreArgs {
	uint kvd;
	uint ctxP;
	uint pos;
	uint n;
	uint oldStageBlk; // the block the stage held before this call
	uint vb;          // 8: q8_0 values, 4: q4_0
};

// One simdgroup per 32 keys (a q8_0 block): d = max|x|/127, quants rounded half to even. Grid: n*kvd/32 simdgroups,
// 8 per threadgroup.
kernel void kv_store_qk(device const float *k [[buffer(0)]],
                        device uchar *kc [[buffer(1)]],
                        constant KVQStoreArgs &a [[buffer(2)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
	uint item = tg * 8 + sg, nbk = a.kvd / 32;
	if (item >= a.n * nbk) {
		return;
	}
	uint t = item / nbk, bi = item % nbk;
	float x = k[(ulong)t * a.kvd + bi * 32 + lane];
	float d = simd_max(abs(x)) / 127, id = d != 0 ? 1 / d : 0;
	device uchar *b = kc + ((ulong)a.pos + t) * (nbk * 34) + bi * 34;
	if (lane == 0) {
		*(device half *)b = half(d);
	}
	((device char *)b)[2 + lane] = char(rint(x * id));
}

// quant_block quantizes the 32 values held one per lane into block b (q8_0, or q4_0 as ggml: the scale is the value
// of largest magnitude over -8, the first such value as the CPU's loop finds it).
static inline void quant_block(float x, device uchar *b, uint vb, uint lane) {
	if (vb == 8) {
		float d = simd_max(abs(x)) / 127, id = d != 0 ? 1 / d : 0;
		if (lane == 0) {
			*(device half *)b = half(d);
		}
		((device char *)b)[2 + lane] = char(rint(x * id));
		return;
	}
	float amax = simd_max(abs(x));
	uint first = simd_min(abs(x) == amax ? lane : 32u);
	float mx = simd_shuffle(x, first);
	float d = mx / -8, id = d != 0 ? 1 / d : 0;
	int q = clamp(int(x * id + 8.5f), 0, 15);
	int hi = simd_shuffle_down(q, 16);
	if (lane == 0) {
		*(device half *)b = half(d);
	}
	if (lane < 16) {
		b[2 + lane] = uchar(q | (hi << 4));
	}
}

// One simdgroup per channel, walking the blocks this call touches in order (so the old stage is read before the new
// one is written): lane j holds position blk*32+j, from the stage or the quantized block (positions before pos), or the
// new values. Full blocks are quantized; the last block's values go to the stage. Grid: kvd simdgroups, 8 per
// threadgroup.
kernel void kv_store_qv(device const float *v [[buffer(0)]],
                        device uchar *vc [[buffer(1)]],
                        device float *stage [[buffer(2)]],
                        constant KVQStoreArgs &a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
	uint ch = tg * 8 + sg;
	if (ch >= a.kvd) {
		return;
	}
	uint bb = a.vb == 8 ? 34 : 18, end = a.pos + a.n;
	device uchar *row = vc + (ulong)ch * (a.ctxP / 32 * bb);
	for (uint blk = a.pos / 32; blk * 32 < end; blk++) {
		uint p = blk * 32 + lane;
		float x = 0;
		if (p < a.pos) { // earlier positions of the first block
			if (blk == a.oldStageBlk) {
				x = stage[ch * 32 + lane];
			} else { // a rewind into a block that was already quantized
				device const uchar *b = row + blk * bb;
				float sc = float(*(device const half *)b);
				if (a.vb == 8) {
					x = sc * float(((device const char *)b)[2 + lane]);
				} else {
					uint q = b[2 + lane % 16];
					x = sc * (float(lane < 16 ? q & 0xF : q >> 4) - 8);
				}
			}
		} else if (p < end) {
			x = v[(ulong)(p - a.pos) * a.kvd + ch];
		}
		if (blk * 32 + 31 < end) {
			quant_block(x, row + blk * bb, a.vb, lane);
		}
		if ((blk + 1) * 32 >= end) { // the newest block: the stage
			stage[ch * 32 + lane] = x;
		}
	}
}

// copy_rows_u8: dst[r*len + c] = src[r*stride + off + c] for rows a.x, stride a.y, off a.z, len a.w (bytes).
kernel void copy_rows_u8(device const uchar *src [[buffer(0)]],
                         device uchar *dst [[buffer(1)]],
                         constant uint4 &a [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
	if (i >= a.x * a.w) {
		return;
	}
	uint r = i / a.w, c = i % a.w;
	dst[i] = src[(ulong)r * a.y + a.z + c];
}
