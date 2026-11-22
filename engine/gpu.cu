// nollama's CUDA kernels: the same operations, arguments and layouts as gpu.metal.
// gpu_cuda.ptx is generated from this file (go generate ./engine, see gpu_cuda.go); the NVIDIA driver compiles the
// PTX for the GPU at load time, so running nollama needs no CUDA toolkit.
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ __forceinline__ float f16(const uchar *p) {
	return __half2float(__ushort_as_half(*(const unsigned short *)p));
}

__device__ __forceinline__ float warp_sum(float v) {
	for (int o = 16; o > 0; o >>= 1) {
		v += __shfl_xor_sync(0xffffffff, v, o);
	}
	return v;
}

__device__ __forceinline__ float warp_max(float v) {
	for (int o = 16; o > 0; o >>= 1) {
		v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
	}
	return v;
}

// Block-wide sum or max over blockDim.x threads (a multiple of 32, at most 1024); red holds 32 floats.
__device__ float block_sum(float v, float *red) {
	uint lane = threadIdx.x % 32, w = threadIdx.x / 32, nw = blockDim.x / 32;
	v = warp_sum(v);
	__syncthreads();
	if (lane == 0) red[w] = v;
	__syncthreads();
	v = lane < nw ? red[lane] : 0;
	v = warp_sum(v);
	return v;
}

__device__ float block_max(float v, float *red) {
	uint lane = threadIdx.x % 32, w = threadIdx.x / 32, nw = blockDim.x / 32;
	v = warp_max(v);
	__syncthreads();
	if (lane == 0) red[w] = v;
	__syncthreads();
	v = lane < nw ? red[lane] : -INFINITY;
	v = warp_max(v);
	return v;
}

struct MatvecArgs {
	uint rows, nb, rowBytes, cols;
};

// Q4_K scale and min of sub-block j (scaleMinK4 in quant.go).
__device__ __forceinline__ void scale_min_k4(int j, const uchar *q, float *sc, float *mn) {
	if (j < 4) {
		*sc = q[j] & 63;
		*mn = q[j + 4] & 63;
	} else {
		*sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
	}
}

// MV_ROWS: rows per warp in the matvecs. Each lane loads its slice of the activations once per block and uses it
// for every row: re-reading the activations per row, not the weights, limits a one-row-per-warp kernel.
#define MV_ROWS 8

// y[t*rows + r] = W row r · x[t] for Q4_K. A warp computes MV_ROWS rows; eight lanes share a block (lane groups
// take blocks g, g+4, ...), lane i covering sub-blocks 2j, 2j+1 (j = i/2) through quant bytes 16*(i%2).. of
// chunk j. 4 warps per block; grid (rows / (4*MV_ROWS), tokens).
extern "C" __global__ void matvec_q4k_f32(const uchar *w, const float *x, float *y, MatvecArgs a) {
	uint r0 = (blockIdx.x * 4 + threadIdx.x / 32) * MV_ROWS, lane = threadIdx.x % 32;
	if (r0 >= a.rows) return;
	const float *xt = x + (size_t)blockIdx.y * a.cols;
	uint il = lane % 8, j = il / 2, off = 32 * j + (il % 2) * 16;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 8; b < a.nb; b += 4) {
		const float4 *x1 = (const float4 *)(xt + b * 256 + 64 * j + (il % 2) * 16);
		float4 a1[4], a2[4];
		float t1 = 0, t2 = 0;
#pragma unroll
		for (int i = 0; i < 4; i++) {
			a1[i] = x1[i];
			a2[i] = x1[i + 8];
			t1 += a1[i].x + a1[i].y + a1[i].z + a1[i].w;
			t2 += a2[i].x + a2[i].y + a2[i].z + a2[i].w;
		}
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			const uchar *blk = w + (size_t)min(r0 + r, a.rows - 1) * a.rowBytes + b * 144;
			float d = f16(blk), dmin = f16(blk + 2), sc1, m1, sc2, m2;
			scale_min_k4(2 * j, blk + 4, &sc1, &m1);
			scale_min_k4(2 * j + 1, blk + 4, &sc2, &m2);
			uint4 q = *(const uint4 *)(blk + 16 + off);
			uint qv[4] = {q.x, q.y, q.z, q.w};
			float s1 = 0, s2 = 0;
#pragma unroll
			for (int i = 0; i < 4; i++) {
				uint v = qv[i];
				s1 += (v & 0xF) * a1[i].x + ((v >> 8) & 0xF) * a1[i].y + ((v >> 16) & 0xF) * a1[i].z + ((v >> 24) & 0xF) * a1[i].w;
				s2 += ((v >> 4) & 0xF) * a2[i].x + ((v >> 12) & 0xF) * a2[i].y + ((v >> 20) & 0xF) * a2[i].z + (v >> 28) * a2[i].w;
			}
			sum[r] += d * (sc1 * s1 + sc2 * s2) - dmin * (m1 * t1 + m2 * t2);
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = warp_sum(sum[r]);
		if (lane == 0 && r0 + r < a.rows) y[(size_t)blockIdx.y * a.rows + r0 + r] = v;
	}
}

// Q6_K (210-byte blocks: ql[128], qh[64], scales[16] int8, d f16). Eight lanes share a block; lane i takes half
// h = i/4 and l = 8*(i%4)..+8, which with l+32, l+64, l+96 is 32 elements.
extern "C" __global__ void matvec_q6k_f32(const uchar *w, const float *x, float *y, MatvecArgs a) {
	uint r0 = (blockIdx.x * 4 + threadIdx.x / 32) * MV_ROWS, lane = threadIdx.x % 32;
	if (r0 >= a.rows) return;
	const float *xt = x + (size_t)blockIdx.y * a.cols;
	uint il = lane % 8, h = il / 4, l0 = 8 * (il % 4), is = l0 / 16;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 8; b < a.nb; b += 4) {
		const float *xh = xt + b * 256 + 128 * h + l0;
		float xv[4][8];
#pragma unroll
		for (int q = 0; q < 4; q++)
#pragma unroll
			for (int i = 0; i < 8; i++) xv[q][i] = xh[32 * q + i];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			const uchar *blk = w + (size_t)min(r0 + r, a.rows - 1) * a.rowBytes + b * 210;
			const uchar *ql = blk + 64 * h + l0, *qh = blk + 128 + 32 * h + l0;
			const signed char *sc = (const signed char *)(blk + 192 + 8 * h);
			float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
			for (int i = 0; i < 8; i++) {
				uint lo = ql[i], hi = ql[i + 32], hb = qh[i];
				s0 += (float)((int)((lo & 0xF) | ((hb & 3) << 4)) - 32) * xv[0][i];
				s1 += (float)((int)((hi & 0xF) | (((hb >> 2) & 3) << 4)) - 32) * xv[1][i];
				s2 += (float)((int)((lo >> 4) | (((hb >> 4) & 3) << 4)) - 32) * xv[2][i];
				s3 += (float)((int)((hi >> 4) | (((hb >> 6) & 3) << 4)) - 32) * xv[3][i];
			}
			sum[r] += f16(blk + 208) * (sc[is] * s0 + sc[is + 2] * s1 + sc[is + 4] * s2 + sc[is + 6] * s3);
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = warp_sum(sum[r]);
		if (lane == 0 && r0 + r < a.rows) y[(size_t)blockIdx.y * a.rows + r0 + r] = v;
	}
}

// Q8_0 (34-byte blocks); a.nb counts 32-element blocks. Four lanes share a block (lane groups take blocks
// g, g+8, ...), lane i of a group covering elements 8i..8i+7.
extern "C" __global__ void matvec_q8_0_f32(const uchar *w, const float *x, float *y, MatvecArgs a) {
	uint r0 = (blockIdx.x * 4 + threadIdx.x / 32) * MV_ROWS, lane = threadIdx.x % 32;
	if (r0 >= a.rows) return;
	const float *xt = x + (size_t)blockIdx.y * a.cols;
	uint e = (lane % 4) * 8;
	float sum[MV_ROWS] = {};
	for (uint b = lane / 4; b < a.nb; b += 8) {
		const float4 *xb = (const float4 *)(xt + b * 32 + e);
		float4 x0 = xb[0], x1 = xb[1];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			const uchar *blk = w + (size_t)min(r0 + r, a.rows - 1) * a.rowBytes + b * 34;
			const ushort *q2 = (const ushort *)(blk + 2 + e); // 2-byte aligned: blocks are 34 bytes
			ushort u0 = q2[0], u1 = q2[1], u2 = q2[2], u3 = q2[3];
			float s = (signed char)(u0 & 0xFF) * x0.x + (signed char)(u0 >> 8) * x0.y + (signed char)(u1 & 0xFF) * x0.z +
			          (signed char)(u1 >> 8) * x0.w + (signed char)(u2 & 0xFF) * x1.x + (signed char)(u2 >> 8) * x1.y +
			          (signed char)(u3 & 0xFF) * x1.z + (signed char)(u3 >> 8) * x1.w;
			sum[r] += f16(blk) * s;
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = warp_sum(sum[r]);
		if (lane == 0 && r0 + r < a.rows) y[(size_t)blockIdx.y * a.rows + r0 + r] = v;
	}
}

// F16 weights; a.nb counts 256-element steps. Lane l takes elements 8l..8l+7 of each 256 (one 16-byte load).
extern "C" __global__ void matvec_f16_f32(const uchar *w, const float *x, float *y, MatvecArgs a) {
	uint r0 = (blockIdx.x * 4 + threadIdx.x / 32) * MV_ROWS, lane = threadIdx.x % 32;
	if (r0 >= a.rows) return;
	const float *xt = x + (size_t)blockIdx.y * a.cols;
	float sum[MV_ROWS] = {};
	for (uint b = 0; b < a.nb; b++) {
		const float4 *xb = (const float4 *)(xt + b * 256 + 8 * lane);
		float4 x0 = xb[0], x1 = xb[1];
#pragma unroll
		for (int r = 0; r < MV_ROWS; r++) {
			uint4 v = *(const uint4 *)(w + (size_t)min(r0 + r, a.rows - 1) * a.rowBytes + (b * 256 + 8 * lane) * 2);
			float2 p0 = __half22float2(*(__half2 *)&v.x), p1 = __half22float2(*(__half2 *)&v.y);
			float2 p2 = __half22float2(*(__half2 *)&v.z), p3 = __half22float2(*(__half2 *)&v.w);
			sum[r] += p0.x * x0.x + p0.y * x0.y + p1.x * x0.z + p1.y * x0.w + p2.x * x1.x + p2.y * x1.y + p3.x * x1.z + p3.y * x1.w;
		}
	}
#pragma unroll
	for (int r = 0; r < MV_ROWS; r++) {
		float v = warp_sum(sum[r]);
		if (lane == 0 && r0 + r < a.rows) y[(size_t)blockIdx.y * a.rows + r0 + r] = v;
	}
}

struct MatmulArgs {
	uint rows, nb, rowBytes, cols, tokens;
};

// dequant1 is weight k of a row. typ: 0 Q8_0, 1 Q4_K, 2 Q6_K, 3 F16.
template <int typ> __device__ __forceinline__ float dequant1(const uchar *row, uint k) {
	if (typ == 0) {
		const uchar *blk = row + (k / 32) * 34;
		return f16(blk) * (float)((const signed char *)(blk + 2))[k % 32];
	} else if (typ == 1) {
		const uchar *blk = row + (k / 256) * 144;
		uint e = k % 256, j = e / 64, hi = (e % 64) / 32;
		float sc, mn;
		scale_min_k4(2 * j + hi, blk + 4, &sc, &mn);
		uchar q = blk[16 + 32 * j + e % 32];
		return f16(blk) * sc * (hi ? q >> 4 : q & 0xF) - f16(blk + 2) * mn;
	} else if (typ == 2) {
		const uchar *blk = row + (k / 256) * 210;
		uint e = k % 256, h = e / 128, quarter = (e % 128) / 32, l = e % 32;
		uint lo = blk[64 * h + l + (quarter & 1) * 32] >> (quarter >= 2 ? 4 : 0);
		uint hb = (blk[128 + 32 * h + l] >> (2 * quarter)) & 3;
		return f16(blk + 208) * ((const signed char *)(blk + 192 + 8 * h))[l / 16 + 2 * quarter] *
		       (float)((int)((lo & 0xF) | (hb << 4)) - 32);
	} else {
		return f16(row + 2 * k);
	}
}

// Tiled matmul for prompt batches (any weight type, through dequant1): a block computes 64 rows x 64 tokens with 256 threads (16x16); thread
// (tx, ty) owns rows ty+16i and tokens tx+16j (i, j < 4), 16 outputs from 8 shared-memory reads per step.
// Grid (rows/64, tokens/64). a.nb counts 32-element steps. Plain f32 FMA, no tensor cores.
template <int typ> __device__ void matmul_tile(const uchar *w, const float *x, float *y, MatmulArgs a) {
	__shared__ float sw[32][65], sx[32][65]; // [k][row], [k][token]; 65 keeps the transposing stores conflict-free
	uint tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
	uint r0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
	float acc[4][4] = {};
	for (uint b = 0; b < a.nb; b++) {
		for (uint it = tid; it < 64 * 32; it += 256) { // 64 rows (and tokens) x the block's 32 elements
			uint l = it / 32, e = it % 32;
			uint r = r0 + l, t = t0 + l;
			if (r < a.rows) {
				sw[e][l] = dequant1<typ>(w + (size_t)r * a.rowBytes, b * 32 + e);
			} else {
				sw[e][l] = 0;
			}
			sx[e][l] = t < a.tokens ? x[(size_t)t * a.cols + b * 32 + e] : 0;
		}
		__syncthreads();
#pragma unroll 8
		for (int k = 0; k < 32; k++) {
			float wv[4], xv[4];
			for (int i = 0; i < 4; i++) wv[i] = sw[k][ty + 16 * i];
			for (int j = 0; j < 4; j++) xv[j] = sx[k][tx + 16 * j];
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++) acc[i][j] += wv[i] * xv[j];
		}
		__syncthreads();
	}
	for (int j = 0; j < 4; j++) {
		uint t = t0 + tx + 16 * j;
		if (t >= a.tokens) continue;
		for (int i = 0; i < 4; i++) {
			uint r = r0 + ty + 16 * i;
			if (r < a.rows) y[(size_t)t * a.rows + r] = acc[i][j];
		}
	}
}

// dequant8h writes weights k..k+7 of a row (k a multiple of 8) to dst as halves. typ as in dequant1.
template <int typ> __device__ __forceinline__ void dequant8h(const uchar *row, uint k, half *dst) {
	if (typ == 0) {
		const uchar *blk = row + (k / 32) * 34;
		float d = f16(blk);
		const ushort *q = (const ushort *)(blk + 2 + k % 32); // 2-byte aligned
#pragma unroll
		for (int i = 0; i < 4; i++) {
			ushort u = q[i];
			dst[2 * i] = __float2half(d * (signed char)(u & 0xFF));
			dst[2 * i + 1] = __float2half(d * (signed char)(u >> 8));
		}
	} else if (typ == 1) {
		const uchar *blk = row + (k / 256) * 144;
		uint e = k % 256, j = e / 64, hi = (e % 64) / 32;
		float sc, mn;
		scale_min_k4(2 * j + hi, blk + 4, &sc, &mn);
		float d = f16(blk) * sc, m = f16(blk + 2) * mn;
		uint2 q = *(const uint2 *)(blk + 16 + 32 * j + e % 32); // 8-byte aligned
		uint qv[2] = {q.x, q.y};
#pragma unroll
		for (int i = 0; i < 8; i++) {
			uint v = (qv[i / 4] >> (8 * (i % 4) + 4 * hi)) & 0xF;
			dst[i] = __float2half(d * v - m);
		}
	} else if (typ == 2) {
		const uchar *blk = row + (k / 256) * 210;
		uint e = k % 256, h = e / 128, quarter = (e % 128) / 32, l = e % 32;
		const uchar *ql = blk + 64 * h + l + (quarter & 1) * 32, *qh = blk + 128 + 32 * h + l;
		float d = f16(blk + 208) * ((const signed char *)(blk + 192 + 8 * h))[l / 16 + 2 * quarter];
		uint shift = quarter >= 2 ? 4 : 0;
#pragma unroll
		for (int i = 0; i < 8; i++) {
			dst[i] = __float2half(d * (float)((int)(((ql[i] >> shift) & 0xF) | (((qh[i] >> (2 * quarter)) & 3) << 4)) - 32));
		}
	} else {
		*(uint4 *)dst = *(const uint4 *)(row + 2 * k);
	}
}

// Tensor-core version of matmul_tile: weights dequantized and activations rounded to half in shared memory,
// multiplied with WMMA 16x16x16 into f32 accumulators (compute 7.0+). A block computes 128 rows x 64 tokens with 8
// warps; warp w owns rows 32*(w%4).. and tokens 32*(w/4).. (2x2 tiles of 16x16). K steps by 64. Grid (rows/128,
// tokens/64); a.cols is the K length.
#define TC_K 64
#define TC_LD (TC_K + 8) // keeps fragment rows 16-byte aligned and spreads banks
template <int typ> __device__ void matmul_tile_tc(const uchar *w, const float *x, float *y, MatmulArgs a) {
	using namespace nvcuda;
	// sw [128 rows][TC_LD] and sx [64 tokens][TC_LD] halves; after the K loop the same memory holds the output
	// tile, [64 tokens][132] floats. Rounding the dequantized weights to half limits the accuracy (about 4e-5 of
	// a row's magnitude); splitting the activations into two halves didn't help that and cost 15%.
	__shared__ __align__(32) uchar smem[64 * 132 * 4];
	half *sw = (half *)smem, *sx = sw + 128 * TC_LD;
	float *so = (float *)smem;
	uint tid = threadIdx.x, warp = tid / 32;
	uint r0 = blockIdx.x * 128, t0 = blockIdx.y * 64;
	uint tr = (warp % 4) * 32, tc = (warp / 4) * 32;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
#pragma unroll
	for (int i = 0; i < 2; i++)
#pragma unroll
		for (int j = 0; j < 2; j++) wmma::fill_fragment(c[i][j], 0.0f);
	for (uint k0 = 0; k0 < a.cols; k0 += TC_K) {
#pragma unroll
		for (int g = 0; g < 4; g++) { // weights: 128 rows x 64 k, 8 per thread per group
			uint it = tid + 256 * g, l = it / 8, e = (it % 8) * 8;
			uint r = r0 + l;
			half *dst = sw + l * TC_LD + e;
			if (r < a.rows) {
				dequant8h<typ>(w + (size_t)r * a.rowBytes, k0 + e, dst);
			} else {
				*(uint4 *)dst = make_uint4(0, 0, 0, 0);
			}
		}
#pragma unroll
		for (int g = 0; g < 2; g++) { // activations: 64 tokens x 64 k
			uint it = tid + 256 * g, l = it / 8, e = (it % 8) * 8;
			uint t = t0 + l;
			half *dst = sx + l * TC_LD + e;
			if (t < a.tokens) {
				const float4 *src = (const float4 *)(x + (size_t)t * a.cols + k0 + e);
				float4 p = src[0], q = src[1];
				__half2 h[4] = {__floats2half2_rn(p.x, p.y), __floats2half2_rn(p.z, p.w), __floats2half2_rn(q.x, q.y),
				                __floats2half2_rn(q.z, q.w)};
				*(uint4 *)dst = *(uint4 *)h;
			} else {
				*(uint4 *)dst = make_uint4(0, 0, 0, 0);
			}
		}
		__syncthreads();
#pragma unroll
		for (int kk = 0; kk < TC_K; kk += 16) {
			wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
			wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[2];
#pragma unroll
			for (int i = 0; i < 2; i++) {
				wmma::load_matrix_sync(fa[i], sw + (tr + 16 * i) * TC_LD + kk, TC_LD);
				wmma::load_matrix_sync(fb[i], sx + (tc + 16 * i) * TC_LD + kk, TC_LD);
			}
#pragma unroll
			for (int i = 0; i < 2; i++)
#pragma unroll
				for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], fa[i], fb[j], c[i][j]);
		}
		__syncthreads();
	}
#pragma unroll
	for (int i = 0; i < 2; i++)
#pragma unroll
		for (int j = 0; j < 2; j++)
			wmma::store_matrix_sync(so + (tc + 16 * j) * 132 + tr + 16 * i, c[i][j], 132, wmma::mem_col_major);
	__syncthreads();
	for (uint it = tid; it < 64 * 128; it += 256) {
		uint tt = it / 128, rr = it % 128;
		if (t0 + tt < a.tokens && r0 + rr < a.rows) y[(size_t)(t0 + tt) * a.rows + r0 + rr] = so[tt * 132 + rr];
	}
}

extern "C" __global__ void matmul_q8_0_tc(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile_tc<0>(w, x, y, a); }
extern "C" __global__ void matmul_q4k_tc(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile_tc<1>(w, x, y, a); }
extern "C" __global__ void matmul_q6k_tc(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile_tc<2>(w, x, y, a); }
extern "C" __global__ void matmul_f16_tc(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile_tc<3>(w, x, y, a); }

extern "C" __global__ void matmul_q8_0_f32(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile<0>(w, x, y, a); }
extern "C" __global__ void matmul_q4k_f32(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile<1>(w, x, y, a); }
extern "C" __global__ void matmul_q6k_f32(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile<2>(w, x, y, a); }
extern "C" __global__ void matmul_f16_f32(const uchar *w, const float *x, float *y, MatmulArgs a) { matmul_tile<3>(w, x, y, a); }

struct NormArgs {
	uint n;
	float eps;
};

// One block per row.
extern "C" __global__ void rmsnorm_f32(const float *x, const float *w, float *out, NormArgs a) {
	__shared__ float red[32];
	const float *xr = x + (size_t)blockIdx.x * a.n;
	float *o = out + (size_t)blockIdx.x * a.n;
	float ss = 0;
	for (uint i = threadIdx.x; i < a.n; i += blockDim.x) ss += xr[i] * xr[i];
	ss = block_sum(ss, red);
	float scale = rsqrtf(ss / a.n + a.eps);
	for (uint i = threadIdx.x; i < a.n; i += blockDim.x) o[i] = xr[i] * scale * w[i];
}

struct RopeArgs {
	uint heads, headDim;
};

// Grid (half/64, heads, tokens), block 64.
extern "C" __global__ void rope_neox_f32(float *x, const float *cs, const float *sn, RopeArgs a) {
	uint half_ = a.headDim / 2;
	uint i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, t = blockIdx.z;
	if (i >= half_) return;
	float *v = x + ((size_t)t * a.heads + h) * a.headDim;
	float c = cs[t * half_ + i], s = sn[t * half_ + i];
	float x0 = v[i], x1 = v[i + half_];
	v[i] = x0 * c - x1 * s;
	v[i + half_] = x0 * s + x1 * c;
}

extern "C" __global__ void swiglu_f32(const float *gate, const float *up, float *out, uint n) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float g = gate[i];
		out[i] = g / (1 + expf(-g)) * up[i];
	}
}

extern "C" __global__ void add_f32(float *x, const float *y, uint n) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) x[i] += y[i];
}

struct KVStoreArgs {
	uint kvd, ctxP, pos;
};

// Grid (kvd/256, tokens), block 256.
extern "C" __global__ void kv_store_f32(const float *k, const float *v, float *kT, float *vc, KVStoreArgs a) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
	if (i >= a.kvd) return;
	kT[(size_t)i * a.ctxP + a.pos + t] = k[(size_t)t * a.kvd + i];
	vc[((size_t)a.pos + t) * a.kvd + i] = v[(size_t)t * a.kvd + i];
}

struct AttnArgs {
	uint heads, kvHeads, headDim, ctxP, pos;
	float scale;
};

// One block (256 threads) per (query head, token); see attention_f32 in gpu.metal. Dynamic shared memory:
// headDim + 256 floats.
extern "C" __global__ void attention_f32(const float *q, const float *kT, const float *v, float *out, float *sc,
                                         const uint2 *mask, AttnArgs a) {
	extern __shared__ float shm[];
	uint h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint ctx = a.pos + t + 1;
	uint2 m = mask[t];
	float *qs = shm, *red = shm + hd;
	float *s = sc + ((size_t)t * a.heads + h) * a.ctxP;
	const float *qh = q + ((size_t)t * a.heads + h) * hd;
	for (uint d = tid; d < hd; d += nt) qs[d] = qh[d];
	__syncthreads();

	const float *kh = kT + (size_t)kvh * hd * a.ctxP;
	float mx = -INFINITY;
	for (uint j = tid; j < ctx; j += nt) {
		float x = -INFINITY;
		if (j < m.x || j >= m.y) {
			x = 0;
			for (uint d = 0; d < hd; d++) x += qs[d] * kh[(size_t)d * a.ctxP + j];
			x *= a.scale;
		}
		s[j] = x;
		mx = fmaxf(mx, x);
	}
	mx = block_max(mx, red);
	float sum = 0;
	for (uint j = tid; j < ctx; j += nt) {
		float e = __expf(s[j] - mx);
		s[j] = e;
		sum += e;
	}
	sum = block_sum(sum, red);
	float inv = 1 / sum;
	__syncthreads();

	uint parts = nt / hd, d = tid % hd, part = tid / hd;
	float o = 0;
	if (part < parts) {
		const float *vh = v + kvh * hd + d;
		for (uint j = part; j < ctx; j += parts) o += s[j] * vh[(size_t)j * kvd];
	}
	red[tid] = o; // red has nt floats here (shared memory is headDim + nt)
	__syncthreads();
	if (tid < hd) {
		float acc = 0;
		for (uint p = 0; p < parts; p++) acc += red[p * hd + tid];
		out[((size_t)t * a.heads + h) * hd + tid] = acc * inv;
	}
}

extern "C" __global__ void copy_f32(float *dst, const float *src, uint n) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) dst[i] = src[i];
}

// Split attention for a few tokens over a long context; see attention_split_f32 in gpu.metal. Grid (heads, tokens,
// splits), 256 threads; dynamic shared memory headDim + chunk + 8*headDim floats.
struct AttnSplitArgs {
	uint heads, kvHeads, headDim, ctxP, pos;
	float scale;
	uint chunk, splits;
};

extern "C" __global__ void attention_split_f32(const float *q, const float *kT, const float *v, float *part,
                                               const uint2 *mask, AttnSplitArgs a) {
	extern __shared__ float shm[];
	__shared__ float red[32];
	uint h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, nt = blockDim.x;
	uint lane = tid % 32, sg = tid / 32, nsg = nt / 32;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint ctx = a.pos + t + 1, c0 = sp * a.chunk, c1 = min(ctx, c0 + a.chunk);
	uint2 m = mask[t];
	float *P = part + (((size_t)t * a.heads + h) * a.splits + sp) * (hd + 2);
	float *qs = shm, *ss = shm + hd, *vr = shm + hd + a.chunk;
	if (c0 >= ctx) {
		for (uint d = tid; d < hd + 2; d += nt) P[d] = d == 0 ? -INFINITY : 0;
		return;
	}
	const float *qh = q + ((size_t)t * a.heads + h) * hd;
	for (uint d = tid; d < hd; d += nt) qs[d] = qh[d];
	__syncthreads();

	const float *kh = kT + (size_t)kvh * hd * a.ctxP;
	float mx = -INFINITY;
	for (uint j = c0 + tid; j < c1; j += nt) {
		float x = -INFINITY;
		if (j < m.x || j >= m.y) {
			x = 0;
			for (uint d = 0; d < hd; d++) x += qs[d] * kh[(size_t)d * a.ctxP + j];
			x *= a.scale;
		}
		ss[j - c0] = x;
		mx = fmaxf(mx, x);
	}
	mx = block_max(mx, red);
	if (mx == -INFINITY) {
		for (uint d = tid; d < hd + 2; d += nt) P[d] = d == 0 ? -INFINITY : 0;
		return;
	}
	__syncthreads(); // ss is complete
	float sum = 0;
	for (uint j = c0 + tid; j < c1; j += nt) {
		float e = __expf(ss[j - c0] - mx);
		ss[j - c0] = e;
		sum += e;
	}
	sum = block_sum(sum, red);
	__syncthreads();

	float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // headDim <= 256
	const float *vh = v + kvh * hd;
	for (uint j = c0 + sg; j < c1; j += nsg) {
		float w = ss[j - c0];
		const float *row = vh + (size_t)j * kvd;
		for (uint i = 0, d = lane; d < hd; i++, d += 32) acc[i] += w * row[d];
	}
	for (uint i = 0, d = lane; d < hd; i++, d += 32) vr[sg * hd + d] = acc[i];
	__syncthreads();
	for (uint d = tid; d < hd; d += nt) {
		float o = 0;
		for (uint i = 0; i < nsg; i++) o += vr[i * hd + d];
		P[2 + d] = o;
	}
	if (tid == 0) {
		P[0] = mx;
		P[1] = sum;
	}
}

// Grid (heads, tokens), headDim threads (rounded up to 32).
extern "C" __global__ void attention_combine_f32(const float *part, float *out, AttnSplitArgs a) {
	uint h = blockIdx.x, t = blockIdx.y, hd = a.headDim, tid = threadIdx.x;
	if (tid >= hd) return;
	const float *P = part + ((size_t)t * a.heads + h) * a.splits * (hd + 2);
	float M = -INFINITY;
	for (uint s = 0; s < a.splits; s++) M = fmaxf(M, P[s * (hd + 2)]);
	float num = 0, den = 0;
	for (uint s = 0; s < a.splits; s++) {
		const float *p = P + s * (hd + 2);
		if (p[0] != -INFINITY) {
			float w = __expf(p[0] - M);
			den += w * p[1];
			num += w * p[2 + tid];
		}
	}
	out[((size_t)t * a.heads + h) * hd + tid] = num / den;
}

// Tiled causal attention for prompts ("flash attention", f32, no tensor cores). A block of 256 threads takes one
// query head and ATT_QT consecutive tokens; it walks the keys in chunks of ATT_KC positions, staging each chunk's
// keys and values in shared memory once for all its tokens, with an online softmax. Same arguments and layouts as
// attention_f32 (no scratch). headDim <= 128. Grid (heads, ceil(tokens/ATT_QT)).
#define ATT_QT 16
#define ATT_KC 32
struct AttnTileArgs {
	uint heads, kvHeads, headDim, ctxP, pos;
	float scale;
	uint tokens;
};

extern "C" __global__ void attention_tile_f32(const float *q, const float *kT, const float *v, float *out,
                                              const uint2 *mask, AttnTileArgs a) {
	__shared__ float qs[ATT_QT][128], ks[128][ATT_KC + 1], vs[ATT_KC][128], ps[ATT_QT][ATT_KC + 1];
	__shared__ float alpha[ATT_QT], lsum[ATT_QT], mrow[ATT_QT];
	uint h = blockIdx.x, t0 = blockIdx.y * ATT_QT, tid = threadIdx.x;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint nt = min((uint)ATT_QT, a.tokens - t0);
	for (uint i = tid; i < ATT_QT * hd; i += 256) {
		uint t = i / hd, d = i % hd;
		qs[t][d] = t < nt ? q[((size_t)(t0 + t) * a.heads + h) * hd + d] * a.scale : 0;
	}
	if (tid < ATT_QT) {
		mrow[tid] = -INFINITY;
		lsum[tid] = 0;
	}
	// this thread's output: row orow, channels ocol + 16*i
	uint orow = tid / 16, ocol = tid % 16;
	float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	uint ctxEnd = a.pos + t0 + nt; // the last token sees positions < ctxEnd
	const float *kh = kT + (size_t)kvh * hd * a.ctxP;
	for (uint c0 = 0; c0 < ctxEnd; c0 += ATT_KC) {
		__syncthreads(); // previous chunk done with ks/vs/ps
		for (uint i = tid; i < hd * ATT_KC; i += 256) {
			uint d = i / ATT_KC, j = i % ATT_KC;
			ks[d][j] = c0 + j < ctxEnd ? kh[(size_t)d * a.ctxP + c0 + j] : 0;
		}
		for (uint i = tid; i < ATT_KC * hd; i += 256) {
			uint j = i / hd, d = i % hd;
			vs[j][d] = c0 + j < ctxEnd ? v[(size_t)(c0 + j) * kvd + kvh * hd + d] : 0;
		}
		__syncthreads();
		for (uint i = tid; i < ATT_QT * ATT_KC; i += 256) { // scores
			uint t = i / ATT_KC, j = i % ATT_KC, p = c0 + j;
			uint2 m = mask[t0 + min(t, nt - 1)];
			float x = -INFINITY;
			if (t < nt && p <= a.pos + t0 + t && (p < m.x || p >= m.y)) {
				x = 0;
				for (uint d = 0; d < hd; d++) x += qs[t][d] * ks[d][j];
			}
			ps[t][j] = x;
		}
		__syncthreads();
		if (tid < ATT_QT) { // online softmax, one thread per token
			float mx = mrow[tid];
			for (int j = 0; j < ATT_KC; j++) mx = fmaxf(mx, ps[tid][j]);
			float al = mx == -INFINITY ? 1 : __expf(mrow[tid] - mx), l = lsum[tid] * al;
			for (int j = 0; j < ATT_KC; j++) {
				float e = mx == -INFINITY ? 0 : __expf(ps[tid][j] - mx);
				ps[tid][j] = e;
				l += e;
			}
			alpha[tid] = al;
			mrow[tid] = mx;
			lsum[tid] = l;
		}
		__syncthreads();
		float al = alpha[orow];
#pragma unroll
		for (int i = 0; i < 8; i++) o[i] *= al;
		for (int j = 0; j < ATT_KC; j++) {
			float pj = ps[orow][j];
#pragma unroll
			for (int i = 0; i < 8; i++) {
				uint d = ocol + 16 * i;
				if (d < hd) o[i] += pj * vs[j][d];
			}
		}
	}
	__syncthreads();
	if (orow < nt) {
		float inv = 1 / lsum[orow];
#pragma unroll
		for (int i = 0; i < 8; i++) {
			uint d = ocol + 16 * i;
			if (d < hd) out[((size_t)(t0 + orow) * a.heads + h) * hd + d] = o[i] * inv;
		}
	}
}
