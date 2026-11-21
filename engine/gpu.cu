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

// Tiled Q8_0 matmul for prompt batches: a block computes 32 rows x 32 tokens with 256 threads (32x8); thread
// (tx, ty) owns rows ty, ty+8, ty+16, ty+24 of token tx. Grid (rows/32, tokens/32). Plain f32 FMA, no tensor cores.
extern "C" __global__ void matmul_q8_0_f32(const uchar *w, const float *x, float *y, MatmulArgs a) {
	__shared__ float sw[32][33], sx[32][33];
	uint tx = threadIdx.x % 32, ty = threadIdx.x / 32;
	uint r0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
	float acc[4] = {0, 0, 0, 0};
	for (uint b = 0; b < a.nb; b++) {
		for (int i = 0; i < 4; i++) { // 256 threads load 32x32 weights and 32x32 activations
			uint lr = ty + 8 * i;
			uint r = r0 + lr, t = t0 + lr;
			if (r < a.rows) {
				const uchar *blk = w + (size_t)r * a.rowBytes + b * 34;
				sw[lr][tx] = f16(blk) * (float)((const signed char *)(blk + 2))[tx];
			} else {
				sw[lr][tx] = 0;
			}
			sx[lr][tx] = t < a.tokens ? x[(size_t)t * a.cols + b * 32 + tx] : 0;
		}
		__syncthreads();
		for (int k = 0; k < 32; k++) {
			float xv = sx[tx][k];
			for (int i = 0; i < 4; i++) acc[i] += sw[ty + 8 * i][k] * xv;
		}
		__syncthreads();
	}
	uint t = t0 + tx;
	if (t < a.tokens) {
		for (int i = 0; i < 4; i++) {
			uint r = r0 + ty + 8 * i;
			if (r < a.rows) y[(size_t)t * a.rows + r] = acc[i];
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
