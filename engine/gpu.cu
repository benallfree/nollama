// nollama's CUDA kernels: the same operations, arguments and layouts as gpu.metal.
// gpu_cuda.ptx is generated from this file (go generate ./engine, see gpu_cuda.go); the NVIDIA driver compiles the
// PTX for the GPU at load time, so running nollama needs no CUDA toolkit.
#include <cuda_fp16.h>
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

// y[t*rows + r] = W row r · x[t]; one warp per row, 4 warps per block; grid (rows/4, tokens).
extern "C" __global__ void matvec_q4k_f32(const uchar *w, const float *x, float *y, MatvecArgs a) {
	uint r = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
	if (r >= a.rows) return;
	const uchar *row = w + (size_t)r * a.rowBytes;
	const float *xt = x + (size_t)blockIdx.y * a.cols;
	float sum = 0;
	for (uint b = 0; b < a.nb; b++) {
		const uchar *blk = row + b * 144;
		float d = f16(blk), dmin = f16(blk + 2);
		const uchar *sc = blk + 4, *qs = blk + 16;
		const float *xb = xt + b * 256;
		for (int j = 0; j < 4; j++) {
			float s1, m1, s2, m2;
			scale_min_k4(2 * j, sc, &s1, &m1);
			scale_min_k4(2 * j + 1, sc, &s2, &m2);
			uchar q = qs[32 * j + lane];
			sum += (d * s1 * (q & 0xF) - dmin * m1) * xb[64 * j + lane];
			sum += (d * s2 * (q >> 4) - dmin * m2) * xb[64 * j + 32 + lane];
		}
	}
	sum = warp_sum(sum);
	if (lane == 0) y[(size_t)blockIdx.y * a.rows + r] = sum;
}

extern "C" __global__ void matvec_q8_0_f32(const uchar *w, const float *x, float *y, MatvecArgs a) {
	uint r = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
	if (r >= a.rows) return;
	const uchar *row = w + (size_t)r * a.rowBytes;
	const float *xt = x + (size_t)blockIdx.y * a.cols;
	uint e = (lane % 8) * 4;
	float sum = 0;
	for (uint b = lane / 8; b < a.nb; b += 4) {
		const uchar *blk = row + b * 34;
		const signed char *qs = (const signed char *)(blk + 2 + e);
		const float *xb = xt + b * 32 + e;
		sum += f16(blk) * (qs[0] * xb[0] + qs[1] * xb[1] + qs[2] * xb[2] + qs[3] * xb[3]);
	}
	sum = warp_sum(sum);
	if (lane == 0) y[(size_t)blockIdx.y * a.rows + r] = sum;
}

struct MatmulArgs {
	uint rows, nb, rowBytes, cols, tokens;
};

// Tiled Q8_0 matmul for prompt batches: a block computes 64 rows x 64 tokens with 256 threads (16x16); thread
// (tx, ty) owns rows ty+16i and tokens tx+16j (i, j < 4), 16 outputs from 8 shared-memory reads per step.
// Grid (rows/64, tokens/64). Plain f32 FMA, no tensor cores.
extern "C" __global__ void matmul_q8_0_f32(const uchar *w, const float *x, float *y, MatmulArgs a) {
	__shared__ float sw[32][65], sx[32][65]; // [k][row], [k][token]; 65 keeps the transposing stores conflict-free
	uint tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
	uint r0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
	float acc[4][4] = {};
	for (uint b = 0; b < a.nb; b++) {
		for (uint it = tid; it < 64 * 32; it += 256) { // 64 rows (and tokens) x the block's 32 elements
			uint l = it / 32, e = it % 32;
			uint r = r0 + l, t = t0 + l;
			if (r < a.rows) {
				const uchar *blk = w + (size_t)r * a.rowBytes + b * 34;
				sw[e][l] = f16(blk) * (float)((const signed char *)(blk + 2))[e];
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
