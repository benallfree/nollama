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

// scale_min_k4r is scale_min_k4 on the scale bytes held in h.yzw (a Q4_K block's first 16 bytes).
static inline float2 scale_min_k4r(uint j, uint4 h) {
	uint sc[3] = {h.y, h.z, h.w};
#define SCB(k) ((sc[(k) / 4] >> (8 * ((k) % 4))) & 0xFF)
	if (j < 4) {
		return float2(SCB(j) & 63, SCB(j + 4) & 63);
	}
	return float2((SCB(j + 4) & 0xF) | ((SCB(j - 4) >> 6) << 4), (SCB(j + 4) >> 4) | ((SCB(j) >> 6) << 4));
#undef SCB
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

// y[t*rows + r] = dot(W row r, x[t*cols:]) for Q4_K weights and float activations.
// A simdgroup computes MV_ROWS rows. Eight lanes share a block (lane groups take blocks g, g+4, ...); lane i of a
// group covers 32 elements: sub-blocks 2j and 2j+1 (j = i/2), quant bytes 16*(i%2).. of chunk j (one 16-byte
// load). Grid: (rows / (MV_ROWS * simdgroups), tokens).
kernel void matvec_q4k_f32(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_ROWS;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_ROWS, a.rows - r0);
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
			device const uchar *blk = w + (ulong)min(r0 + r, a.rows - 1) * a.rowBytes + b * 144;
			uint4 hd = *(device const uint4 *)blk; // d, dmin, scales[12]: one 16-byte load (blocks are 16-aligned)
			float2 dm = float2(as_type<half2>(hd.x));
			float d = dm.x, dmin = dm.y;
			float2 sm1 = scale_min_k4r(2 * j, hd);
			float2 sm2 = scale_min_k4r(2 * j + 1, hd);
			uint4 q = *(device const uint4 *)(blk + 16 + off);
			float s1 = 0, s2 = 0;
			for (int i = 0; i < 4; i++) {
				uint v = q[i];
				s1 += dot(float4(v & 0xF, (v >> 8) & 0xF, (v >> 16) & 0xF, (v >> 24) & 0xF), a1[i]);
				s2 += dot(float4((v >> 4) & 0xF, (v >> 12) & 0xF, (v >> 20) & 0xF, v >> 28), a2[i]);
			}
			sum[r] += d * (sm1.x * s1 + sm2.x * s2) - dmin * (sm1.y * t1 + sm2.y * t2);
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = simd_sum(sum[r]);
		if (lane == 0 && r < (int)nr) {
			y[(ulong)tg.y * a.rows + r0 + r] = v;
		}
	}
}

// Same contract for Q6_K weights (210-byte blocks: ql[128], qh[64], scales[16] int8, d f16). Eight lanes share a
// block; lane i takes half h = i/4 and l = 8*(i%4)..+8 of it, which with l+32, l+64, l+96 is 32 elements.
kernel void matvec_q6k_f32(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_ROWS;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_ROWS, a.rows - r0);
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
			device const uchar *blk = w + (ulong)min(r0 + r, a.rows - 1) * a.rowBytes + b * 210;
			device const uchar *ql = blk + 64 * h + l0, *qh = blk + 128 + 32 * h + l0;
			device const char *sc = (device const char *)(blk + 192 + 8 * h);
			float s[4] = {0, 0, 0, 0};
			for (int k = 0; k < 2; k++) {
				float4 q0, q1, q2, q3;
				for (int i = 0; i < 4; i++) {
					uint lo = ql[4 * k + i], hi = ql[4 * k + i + 32], hb = qh[4 * k + i];
					q0[i] = float(int((lo & 0xF) | ((hb & 3) << 4)) - 32);
					q1[i] = float(int((hi & 0xF) | (((hb >> 2) & 3) << 4)) - 32);
					q2[i] = float(int((lo >> 4) | (((hb >> 4) & 3) << 4)) - 32);
					q3[i] = float(int((hi >> 4) | (((hb >> 6) & 3) << 4)) - 32);
				}
				s[0] += dot(q0, xv[0][k]);
				s[1] += dot(q1, xv[1][k]);
				s[2] += dot(q2, xv[2][k]);
				s[3] += dot(q3, xv[3][k]);
			}
			sum[r] += float(*(device const half *)(blk + 208)) *
			          (float(sc[is]) * s[0] + float(sc[is + 2]) * s[1] + float(sc[is + 4]) * s[2] + float(sc[is + 6]) * s[3]);
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = simd_sum(sum[r]);
		if (lane == 0 && r < (int)nr) {
			y[(ulong)tg.y * a.rows + r0 + r] = v;
		}
	}
}

// Same contract for F16 weights; a.nb counts 256-element steps. A simdgroup computes MV_ROWS rows; lane l takes
// elements 8l..8l+7 of each 256 (one 16-byte load per row), reusing its activations across the rows.
kernel void matvec_f16_f32(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_ROWS;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_ROWS, a.rows - r0);
	device const float *xt = x + (ulong)tg.y * a.cols;
	float sum[MV_ROWS] = {};
	for (uint b = 0; b < a.nb; b++) {
		device const float4 *xb = (device const float4 *)(xt + b * 256 + 8 * lane);
		float4 x0 = xb[0], x1 = xb[1];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			device const half4 *wr = (device const half4 *)(w + (ulong)min(r0 + r, a.rows - 1) * a.rowBytes + (b * 256 + 8 * lane) * 2);
			sum[r] += dot(float4(wr[0]), x0) + dot(float4(wr[1]), x1);
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = simd_sum(sum[r]);
		if (lane == 0 && r < (int)nr) {
			y[(ulong)tg.y * a.rows + r0 + r] = v;
		}
	}
}

// Same contract for Q8_0 weights (34-byte blocks: d f16, qs[32] int8); a.nb counts 32-element blocks. A simdgroup
// computes MV_ROWS rows; four lanes share a block (lane groups take blocks g, g+8, ...), lane i of a group covering
// elements 8i..8i+7.
kernel void matvec_q8_0_f32(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant MatvecArgs &a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
	uint r0 = (tg.x * nsg + sg) * MV_ROWS;
	if (r0 >= a.rows) {
		return;
	}
	uint nr = min((uint)MV_ROWS, a.rows - r0);
	device const float *xt = x + (ulong)tg.y * a.cols;
	uint e = (lane % 4) * 8;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 4; b < a.nb; b += 8) {
		device const float4 *xb = (device const float4 *)(xt + b * 32 + e);
		float4 x0 = xb[0], x1 = xb[1];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			device const uchar *blk = w + (ulong)min(r0 + r, a.rows - 1) * a.rowBytes + b * 34;
			device const char2 *q = (device const char2 *)(blk + 2 + e); // blocks are 34 bytes: 2-byte aligned
			char2 q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
			sum[r] += float(*(device const half *)blk) *
			          (dot(float4(q0.x, q0.y, q1.x, q1.y), x0) + dot(float4(q2.x, q2.y, q3.x, q3.y), x1));
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = simd_sum(sum[r]);
		if (lane == 0 && r < (int)nr) {
			y[(ulong)tg.y * a.rows + r0 + r] = v;
		}
	}
}

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
struct NormRopeArgs {
	uint heads;
	uint headDim;
	float eps;
};

kernel void norm_rope_f32(device float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device const float *cs [[buffer(2)]],
                          device const float *sn [[buffer(3)]],
                          constant NormRopeArgs &a [[buffer(4)]],
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
	v[i] = x0 * c - x1 * s;
	v[i + half_] = x0 * s + x1 * c;
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

// dequant8 writes 8 consecutive weights of a row, starting at element k (a multiple of 8), to dst.
// typ: 0 Q8_0, 1 Q4_K, 2 Q6_K, 3 F16.
template <typename T> static inline void dequant8(device const uchar *row, uint k, threadgroup T *dst, const int typ) {
	if (typ == 0) {
		device const uchar *blk = row + (k / 32) * 34;
		float d = float(*(device const half *)blk);
		device const char *qs = (device const char *)(blk + 2 + k % 32);
		for (int i = 0; i < 8; i++) {
			dst[i] = T(d * float(qs[i]));
		}
	} else if (typ == 1) {
		device const uchar *blk = row + (k / 256) * 144;
		uint e = k % 256, j = e / 64, hi = (e % 64) / 32, l = e % 32;
		uchar2 sm = scale_min_k4(2 * j + hi, blk + 4);
		float d = float(*(device const half *)blk) * sm.x, m = float(*(device const half *)(blk + 2)) * sm.y;
		device const uchar *qs = blk + 16 + 32 * j + l;
		for (int i = 0; i < 8; i++) {
			dst[i] = T(d * float(hi ? qs[i] >> 4 : qs[i] & 0xF) - m);
		}
	} else if (typ == 3) {
		device const half *p = (device const half *)row + k;
		for (int i = 0; i < 8; i++) {
			dst[i] = T(float(p[i]));
		}
	} else {
		device const uchar *blk = row + (k / 256) * 210;
		uint e = k % 256, h = e / 128, quarter = (e % 128) / 32, l = e % 32;
		device const uchar *ql = blk + 64 * h + l + (quarter & 1) * 32, *qh = blk + 128 + 32 * h + l;
		float d = float(*(device const half *)(blk + 208)) * float(((device const char *)(blk + 192 + 8 * h))[l / 16 + 2 * quarter]);
		uint shift = quarter >= 2 ? 4 : 0;
		for (int i = 0; i < 8; i++) {
			dst[i] = T(d * float(int(((ql[i] >> shift) & 0xF) | (((qh[i] >> (2 * quarter)) & 3) << 4)) - 32));
		}
	}
}

// T is the weight tile's type in threadgroup memory: float, or half (half the memory traffic; the dequantized
// weights round to half, about 4e-5 of a row's magnitude).
template <typename T>
static inline void matmul_tile(device const uchar *w, device const float *x, device float *y, constant MatmulArgs &a,
                               threadgroup float *shm, uint2 tg, uint tid, uint sg, const int typ) {
	threadgroup T *sw = (threadgroup T *)shm;    // [64 rows][32 k]
	threadgroup float *sx = shm + 64 * 32;       // [64 tokens][32 k]
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
		{
			uint r = r0 + lr;
			threadgroup T *dst = sw + lr * 32 + lk;
			if (r < a.rows) {
				dequant8(w + (ulong)r * a.rowBytes, b * 32 + lk, dst, typ);
			} else {
				for (int i = 0; i < 8; i++) {
					dst[i] = 0;
				}
			}
			uint t = t0 + lr;
			threadgroup float *dx = sx + lr * 32 + lk;
			if (t < a.tokens) {
				device const float4 *src = (device const float4 *)(x + (ulong)t * a.cols + b * 32 + lk);
				((threadgroup float4 *)dx)[0] = src[0];
				((threadgroup float4 *)dx)[1] = src[1];
			} else {
				for (int i = 0; i < 8; i++) {
					dx[i] = 0;
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint kk = 0; kk < 32; kk += 8) {
			simdgroup_matrix<T, 8, 8> ma[4];
			simdgroup_float8x8 mb[2];
			for (int i = 0; i < 4; i++) {
				simdgroup_load(ma[i], sw + (sr + 8 * i) * 32 + kk, 32);
			}
			for (int j = 0; j < 2; j++) {
				simdgroup_load(mb[j], sx + (st + 8 * j) * 32 + kk, 32, ulong2(0, 0), true); // X^T: [k][token]
			}
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 2; j++) {
					simdgroup_multiply_accumulate(c[i][j], ma[i], mb[j], c[i][j]);
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	// c[i][j] is [rows][tokens]; stage the 64x64 tile transposed ([token][row]) and write in bounds.
	threadgroup float *out = shm; // reuse: [64 tokens][64 rows]
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++) {
			simdgroup_store(c[i][j], out + (st + 8 * j) * 64 + sr + 8 * i, 64, ulong2(0, 0), true);
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint i = tid; i < 64 * 64; i += 256) {
		uint t = t0 + i / 64, r = r0 + i % 64;
		if (t < a.tokens && r < a.rows) {
			y[(ulong)t * a.rows + r] = out[i];
		}
	}
}

#define MATMUL_KERNEL(name, typ)                                                                                     \
	kernel void name(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],                   \
	                 device float *y [[buffer(2)]], constant MatmulArgs &a [[buffer(3)]],                        \
	                 threadgroup float *shm [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],       \
	                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {     \
		matmul_tile<float>(w, x, y, a, shm, tg, tid, sg, typ);                                                   \
	}                                                                                                            \
	kernel void name##_h(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],               \
	                     device float *y [[buffer(2)]], constant MatmulArgs &a [[buffer(3)]],                    \
	                     threadgroup float *shm [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],   \
	                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
		matmul_tile<half>(w, x, y, a, shm, tg, tid, sg, typ);                                                    \
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
};

kernel void attention_f32(device const float *q [[buffer(0)]],
                          device const float *kT [[buffer(1)]],
                          device const float *v [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          device float *sc [[buffer(4)]],
                          device const uint2 *mask [[buffer(5)]],
                          constant AttnArgs &a [[buffer(6)]],
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
	device const float *kh = kT + (ulong)kvh * hd * a.ctxP;
	float mx = -INFINITY;
	for (uint j = tid; j < ctx; j += nt) {
		float x = -INFINITY;
		if (j < m.x || j >= m.y) {
			x = 0;
			for (uint d = 0; d < hd; d++) {
				x += qs[d] * kh[(ulong)d * a.ctxP + j];
			}
			x *= a.scale;
		}
		s[j] = x;
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

	// weighted sum of values: thread (d, part) sums positions part, part+parts, ... for channel d
	uint parts = nt / hd, d = tid % hd, part = tid / hd;
	float o = 0;
	if (part < parts) {
		device const float *vh = v + kvh * hd + d;
		for (uint j = part; j < ctx; j += parts) {
			o += s[j] * vh[(ulong)j * kvd];
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
};

kernel void attention_split_f32(device const float *q [[buffer(0)]],
                                device const float *kT [[buffer(1)]],
                                device const float *v [[buffer(2)]],
                                device float *part [[buffer(3)]],
                                device const uint2 *mask [[buffer(4)]],
                                constant AttnSplitArgs &a [[buffer(5)]],
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

	device const float *kh = kT + (ulong)kvh * hd * a.ctxP;
	float mx = -INFINITY;
	for (uint j = c0 + tid; j < c1; j += nt) {
		float x = -INFINITY;
		if (j < m.x || j >= m.y) {
			x = 0;
			for (uint d = 0; d < hd; d++) {
				x += qs[d] * kh[(ulong)d * a.ctxP + j];
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

	// values: simdgroup sg sums positions c0+sg, c0+sg+nsg, ...; lane covers channels lane, lane+32, ...
	float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // headDim <= 256
	device const float *vh = v + kvh * hd;
	for (uint j = c0 + sg; j < c1; j += nsg) {
		float w = ss[j - c0];
		device const float *vr = vh + (ulong)j * kvd;
		for (uint i = 0, d = lane; d < hd; i++, d += 32) {
			acc[i] += w * vr[d];
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
};

kernel void attention_tile_f32(device const float *q [[buffer(0)]],
                               device const float *kT [[buffer(1)]],
                               device const float *v [[buffer(2)]],
                               device float *out [[buffer(3)]],
                               device const uint2 *mask [[buffer(4)]],
                               constant AttnTileArgs &a [[buffer(5)]],
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
	device const float *kh = kT + (ulong)kvh * hd * a.ctxP;
	for (uint c0 = 0; c0 < ctxEnd; c0 += ATT_KC) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint i = tid; i < hd * ATT_KC; i += 256) {
			uint d = i / ATT_KC, j = i % ATT_KC;
			ks[d][j] = c0 + j < ctxEnd ? kh[(ulong)d * a.ctxP + c0 + j] : 0;
		}
		for (uint i = tid; i < ATT_KC * hd; i += 256) {
			uint j = i / hd, d = i % hd;
			vs[j][d] = c0 + j < ctxEnd ? v[(ulong)(c0 + j) * kvd + kvh * hd + d] : 0;
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
