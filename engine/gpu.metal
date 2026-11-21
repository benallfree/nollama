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

struct MatvecArgs {
	uint rows;     // output rows
	uint nb;       // 256-element blocks per row
	uint rowBytes; // bytes per weight row
	uint cols;     // input length (nb*256)
};

// y[t*rows + r] = dot(W row r, x[t*cols:]) for Q4_K weights and float activations.
// One simdgroup per row; lane l handles elements 64j+l and 64j+32+l of every block, so
// both the weight bytes and the activations are read coalesced. Grid: (row groups, tokens).
kernel void matvec_q4k_f32(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r = tg.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	device const uchar *row = w + (ulong)r * a.rowBytes;
	device const float *xt = x + (ulong)tg.y * a.cols;
	float sum = 0;
	for (uint b = 0; b < a.nb; b++) {
		device const uchar *blk = row + b * 144;
		float d = float(*(device const half *)blk);
		float dmin = float(*(device const half *)(blk + 2));
		device const uchar *sc = blk + 4;
		device const uchar *qs = blk + 16;
		device const float *xb = xt + b * 256;
		for (int j = 0; j < 4; j++) {
			uchar2 sm1 = scale_min_k4(2 * j, sc);
			uchar2 sm2 = scale_min_k4(2 * j + 1, sc);
			uchar q = qs[32 * j + lane];
			float x1 = xb[64 * j + lane];
			float x2 = xb[64 * j + 32 + lane];
			sum += (d * sm1.x * float(q & 0xF) - dmin * sm1.y) * x1;
			sum += (d * sm2.x * float(q >> 4) - dmin * sm2.y) * x2;
		}
	}
	sum = simd_sum(sum);
	if (lane == 0) {
		y[(ulong)tg.y * a.rows + r] = sum;
	}
}

// Same contract for Q8_0 weights (34-byte blocks: d f16, qs[32] int8); a.nb counts 32-element blocks.
// Lanes split into 4 groups of 8; group g takes blocks g, g+4, ...; each lane does 4 elements of its block.
kernel void matvec_q8_0_f32(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant MatvecArgs &a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
	uint r = tg.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	device const uchar *row = w + (ulong)r * a.rowBytes;
	device const float *xt = x + (ulong)tg.y * a.cols;
	uint e = (lane % 8) * 4;
	float sum = 0;
	for (uint b = lane / 8; b < a.nb; b += 4) {
		device const uchar *blk = row + b * 34;
		device const char *qs = (device const char *)(blk + 2 + e);
		device const float *xb = xt + b * 32 + e;
		float s = float(qs[0]) * xb[0] + float(qs[1]) * xb[1] + float(qs[2]) * xb[2] + float(qs[3]) * xb[3];
		sum += float(*(device const half *)blk) * s;
	}
	sum = simd_sum(sum);
	if (lane == 0) {
		y[(ulong)tg.y * a.rows + r] = sum;
	}
}

// rmsnorm: out[r] = x[r] * w / sqrt(mean(x[r]^2) + eps), one threadgroup per row of n floats.
// Rows can be whole hidden states (w length n) or per-head slices for QK-norm (n = headDim).
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
// Each threadgroup computes a 32-row x 32-token tile with 4 simdgroups (16x16 each, as 2x2 8x8
// simdgroup matrices). The K loop steps one Q8_0 block (32) at a time: weights are dequantized and
// activations copied into threadgroup memory, then multiplied with simdgroup_multiply_accumulate.
// Grid: (rows/32, ceil(tokens/32)). a.nb counts 32-element blocks; a.rows here is also the y stride.
struct MatmulArgs {
	uint rows;
	uint nb;
	uint rowBytes;
	uint cols;
	uint tokens;
};

kernel void matmul_q8_0_f32(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant MatmulArgs &a [[buffer(3)]],
                            threadgroup float *shm [[threadgroup(0)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
	threadgroup float *sw = shm;        // [32 rows][32 k]
	threadgroup float *sx = shm + 1024; // [32 tokens][32 k]
	uint r0 = tg.x * 32, t0 = tg.y * 32;
	uint sr = (sg / 2) * 16, st = (sg % 2) * 16; // this simdgroup's sub-tile

	simdgroup_float8x8 c[2][2];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			c[i][j] = simdgroup_float8x8(0);
		}
	}
	// 128 threads: thread tid loads 8 consecutive k of row/token tid/4.
	uint lr = tid / 4, lk = (tid % 4) * 8;
	for (uint b = 0; b < a.nb; b++) {
		{
			uint r = r0 + lr;
			threadgroup float *dst = sw + lr * 32 + lk;
			if (r < a.rows) {
				device const uchar *blk = w + (ulong)r * a.rowBytes + b * 34;
				float d = float(*(device const half *)blk);
				device const char *qs = (device const char *)(blk + 2 + lk);
				for (int i = 0; i < 8; i++) {
					dst[i] = d * float(qs[i]);
				}
			} else {
				for (int i = 0; i < 8; i++) {
					dst[i] = 0;
				}
			}
			uint t = t0 + lr;
			threadgroup float *dx = sx + lr * 32 + lk;
			if (t < a.tokens) {
				device const float *src = x + (ulong)t * a.cols + b * 32 + lk;
				for (int i = 0; i < 8; i++) {
					dx[i] = src[i];
				}
			} else {
				for (int i = 0; i < 8; i++) {
					dx[i] = 0;
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint kk = 0; kk < 32; kk += 8) {
			simdgroup_float8x8 ma[2], mb[2];
			for (int i = 0; i < 2; i++) {
				simdgroup_load(ma[i], sw + (sr + 8 * i) * 32 + kk, 32);
				simdgroup_load(mb[i], sx + (st + 8 * i) * 32 + kk, 32, ulong2(0, 0), true); // X^T: [k][token]
			}
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					simdgroup_multiply_accumulate(c[i][j], ma[i], mb[j], c[i][j]);
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	// c[i][j] is [rows][tokens]; stage the 32x32 tile transposed ([token][row]) and write in bounds.
	threadgroup float *out = shm; // reuse: [32 tokens][32 rows]
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			simdgroup_store(c[i][j], out + (st + 8 * j) * 32 + sr + 8 * i, 32, ulong2(0, 0), true);
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint i = tid; i < 1024; i += 128) {
		uint t = t0 + i / 32, r = r0 + i % 32;
		if (t < a.tokens && r < a.rows) {
			y[(ulong)t * a.rows + r] = out[i];
		}
	}
}
