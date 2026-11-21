// nollama's deterministic CUDA kernels: each computes exactly what the CPU's reference code computes, bit for bit
// (deterministic mode, LoadOptions.Deterministic). The same kernels as gpu_det.metal, in the same orders.
//
// gpu_cuda_det.ptx is generated from this file with -fmad=false -ftz=false -prec-div=true -prec-sqrt=true (gen_ptx.sh):
// every product is rounded on its own unless written as fmaf (the CPU's fma32), division and square roots round
// correctly, and subnormals are kept, as on the CPU. Where the CPU flushes them (ftz32 in detmath.go), ftz() does too.
// It includes gpu.cu, so the shared kernels (rope, add, copy, kv_store) are compiled the same way.
//
// Each kernel names the Go function whose order it follows. Sums of integers are exact, so their order is free;
// every float sum runs in the reference's order.
#include "gpu.cu"

// ftz is ftz32 (detmath.go): a subnormal becomes a zero of the same sign.
__device__ __forceinline__ float ftz(float v) {
	return (__float_as_uint(v) & 0x7F800000u) == 0 ? __uint_as_float(__float_as_uint(v) & 0x80000000u) : v;
}

// tree32 is sum32 (quant.go) over a warp's 32 lanes: lane i adds lane i+s for s = 16, 8, 4, 2, 1. Every lane gets
// the total.
__device__ __forceinline__ float tree32(float v) {
	for (int s = 16; s > 0; s >>= 1) {
		v += __shfl_down_sync(0xffffffff, v, s);
	}
	return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ int warp_isum(int v) {
	for (int o = 16; o > 0; o >>= 1) {
		v += __shfl_xor_sync(0xffffffff, v, o);
	}
	return v;
}

// expf_ref is expf (kernels.go).
__device__ __forceinline__ float expf_ref(float x) {
	if (x < -87.0f) {
		return 0;
	}
	if (x > 88.0f) {
		x = 88.0f;
	}
	// n = floor(t + 0.5) with t + 0.5 exact, as Go computes it in float64: t - floor(t) is exact here (Sterbenz)
	float t = x * 1.44269504088896341f;
	float n = floorf(t);
	if (t - n >= 0.5f) {
		n += 1;
	}
	float r = x - n * 0.693359375f - n * -2.12194440e-4f;
	float p = 1.9875691500e-4f;
	p = p * r + 1.3981999507e-3f;
	p = p * r + 8.3334519073e-3f;
	p = p * r + 4.1665795894e-2f;
	p = p * r + 1.6666665459e-1f;
	p = p * r + 5.0000001201e-1f;
	p = p * r * r + r + 1;
	return p * __uint_as_float((uint)((int)n + 127) << 23);
}

// i32 reads a 4-byte word at a 2-byte aligned address (Q8_0 weight quants follow a 2-byte scale).
__device__ __forceinline__ int i32at2(const uchar *p) {
	const unsigned short *s = (const unsigned short *)p;
	return (int)((uint)s[0] | (uint)s[1] << 16);
}

// rmsnorm_det is rmsnorm (forward.go): one warp per row; lane l sums x[i]^2 for i = l mod 32.
extern "C" __global__ void rmsnorm_det(const float *x, const float *w, float *out, NormArgs a) {
	uint row = blockIdx.x, lane = threadIdx.x;
	const float *xr = x + (size_t)row * a.n;
	float *orow = out + (size_t)row * a.n;
	float acc = 0;
	for (uint i = lane; i < a.n; i += 32) {
		float v = ftz(xr[i]);
		acc = ftz(fmaf(v, v, acc));
	}
	float ss = tree32(acc);
	float scale = 1.0f / sqrtf(ss / (float)a.n + a.eps);
	for (uint i = lane; i < a.n; i += 32) {
		orow[i] = ftz(ftz(ftz(xr[i]) * scale) * ftz(w[i]));
	}
}

// swiglu_det is swiglu (kernels.go).
extern "C" __global__ void swiglu_det(const float *gate, const float *up, float *out, uint n) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float g = ftz(gate[i]);
		out[i] = ftz(ftz(g / (1.0f + expf_ref(-g))) * ftz(up[i]));
	}
}

// quant_q8_0_det is quantizeQ8_0 (kernels.go): one warp per 32-element block, one element per lane; out is 36-byte
// blocks (d f32, qs[32]).
extern "C" __global__ void quant_q8_0_det(const float *x, uchar *out, uint nblocks) {
	uint b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
	if (b >= nblocks) {
		return;
	}
	float v = x[(size_t)b * 32 + lane];
	float amax = warp_max(fabsf(v));
	float d = 0, id = 0;
	if (amax >= 0x1p-64f) { // qTiny (detmath.go): a block of tiny values quantizes to zero
		d = amax / 127.0f;
		id = 1.0f / d;
	}
	uchar *ob = out + (size_t)b * 36;
	ob[4 + lane] = (uchar)(signed char)(int)rintf(v * id);
	if (lane == 0) {
		*(float *)ob = d;
	}
}

// quant_q8_k_det is quantizeQ8K (quant.go): one warp per 256-element block, lane l takes elements 8l..8l+7; out is
// ggml's block_q8_K (d f32, qs[256] int8, bsums[16] int16), 292 bytes.
extern "C" __global__ void quant_q8_k_det(const float *x, uchar *out, uint nblocks) {
	uint b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
	if (b >= nblocks) {
		return;
	}
	const float *xb = x + (size_t)b * 256 + lane * 8;
	float v[8];
	// the first element with the largest magnitude, as quantizeQ8K's strict > picks it
	float amax = 0, mx = 0;
	uint at = 0xFFFFFFFF;
	for (uint i = 0; i < 8; i++) {
		v[i] = xb[i];
		float av = fabsf(v[i]);
		if (av > amax) {
			amax = av, mx = v[i], at = lane * 8 + i;
		}
	}
	for (int s = 16; s > 0; s >>= 1) {
		float oa = __shfl_xor_sync(0xffffffff, amax, s), om = __shfl_xor_sync(0xffffffff, mx, s);
		uint oat = __shfl_xor_sync(0xffffffff, at, s);
		if (oa > amax || (oa == amax && oat < at)) {
			amax = oa, mx = om, at = oat;
		}
	}
	uchar *ob = out + (size_t)b * 292;
	if (amax < 0x1p-64f) { // qTiny: zeros, or values too small for a finite -128/max
		for (uint i = 0; i < 8; i++) {
			ob[4 + lane * 8 + i] = 0;
		}
		if (lane < 16) {
			*(short *)(ob + 260 + 2 * lane) = 0;
		}
		if (lane == 0) {
			*(float *)ob = 0;
		}
		return;
	}
	float iscale = -128.0f / mx;
	int s = 0;
	for (uint i = 0; i < 8; i++) {
		int q = min(max((int)rintf(iscale * v[i]), -128), 127);
		ob[4 + lane * 8 + i] = (uchar)(signed char)q;
		s += q;
	}
	s += __shfl_down_sync(0xffffffff, s, 1); // bsums[j] covers elements 16j..16j+15: lanes 2j and 2j+1
	if (lane % 2 == 0) {
		*(short *)(ob + 260 + lane) = (short)s;
	}
	if (lane == 0) {
		*(float *)ob = 1.0f / iscale;
	}
}

struct DetMatvecArgs {
	uint rows;     // output rows
	uint nb;       // blocks per row
	uint rowBytes; // bytes per weight row
	uint actBytes; // bytes per quantized activation row
	uint tokens;
};

// Tokens per warp in the deterministic matvecs: each weight row is read once for this many tokens (detT in
// gpu_cuda.go).
#define DET_T 8

// matvec_q8_0_det is dotQ8_0Go (kernels.go): y[t*rows + r] = W row r . activation row t (36-byte Q8_0 blocks).
// One warp per row and DET_T tokens. Blocks go in chunks of 32, one block per lane: each lane computes its block's
// scale product and exact integer total (dp4a) for every token into shared memory; then lane t adds the chunk's 32
// blocks into token t's sum, one multiply-add per block, in block order.
// Grid: (row groups, token groups). Dynamic shared memory: 2*32*min(DET_T, tokens) words per warp.
extern "C" __global__ void matvec_q8_0_det(const uchar *w, const uchar *x, float *y, DetMatvecArgs a) {
	extern __shared__ float shm[];
	uint sg = threadIdx.x / 32, lane = threadIdx.x % 32, nsg = blockDim.x / 32;
	uint r = blockIdx.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	uint t0 = blockIdx.y * DET_T, nt = min((uint)DET_T, a.tokens - t0);
	uint st = min((uint)DET_T, a.tokens);
	float *td = shm + sg * (64 * st);
	int *ts = (int *)(td + 32 * st);
	const uchar *row = w + (size_t)r * a.rowBytes;
	const uchar *xa = x + (size_t)t0 * a.actBytes;
	float sum = 0; // lane t < nt: token t0+t
	for (uint c = 0; c < a.nb; c += 32) {
		uint b = c + lane;
		if (b < a.nb) {
			const uchar *blk = row + b * 34;
			float dw = f16(blk);
			int wv[8];
			for (int i = 0; i < 8; i++) {
				wv[i] = i32at2(blk + 2 + 4 * i);
			}
			for (uint t = 0; t < nt; t++) {
				const uchar *ab = xa + (size_t)t * a.actBytes + b * 36;
				const int *xq = (const int *)(ab + 4);
				int s = 0;
				for (int i = 0; i < 8; i++) {
					s = __dp4a(wv[i], xq[i], s);
				}
				td[lane * st + t] = dw * *(const float *)ab;
				ts[lane * st + t] = s;
			}
		}
		__syncwarp();
		if (lane < nt) {
			uint kn = min(32u, a.nb - c);
			for (uint k = 0; k < kn; k++) {
				sum = fmaf(td[k * st + lane], (float)ts[k * st + lane], sum);
			}
		}
		__syncwarp();
	}
	if (lane < nt) {
		y[(size_t)(t0 + lane) * a.rows + r] = sum;
	}
}

// matvec1_q8_0_det is matvec_q8_0_det for one token, with coalesced loads: four 8-lane groups take blocks b..b+3, 4
// elements per lane; each block's exact integer total (8 lanes added with shuffles) and scale product go to shared
// memory, and lane 0 then runs the multiply-add chain in block order. Dynamic shared memory: 2*nb words per warp.
extern "C" __global__ void matvec1_q8_0_det(const uchar *w, const uchar *x, float *y, DetMatvecArgs a) {
	extern __shared__ float shm[];
	uint sg = threadIdx.x / 32, lane = threadIdx.x % 32, nsg = blockDim.x / 32;
	uint r = blockIdx.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	float *td = shm + sg * 2 * a.nb;
	int *tsum = (int *)(td + a.nb);
	const uchar *row = w + (size_t)r * a.rowBytes;
	uint e = (lane % 8) * 4, grp = lane / 8;
	for (uint b = grp; b < a.nb; b += 4) {
		const uchar *ab = x + (size_t)b * 36;
		int s = __dp4a(i32at2(row + b * 34 + 2 + e), *(const int *)(ab + 4 + e), 0);
		s += __shfl_xor_sync(0xffffffff, s, 1);
		s += __shfl_xor_sync(0xffffffff, s, 2);
		s += __shfl_xor_sync(0xffffffff, s, 4);
		if (lane % 8 == 0) {
			tsum[b] = s;
			td[b] = f16(row + b * 34) * *(const float *)ab;
		}
	}
	__syncwarp();
	if (lane == 0) {
		float sum = 0;
		for (uint b = 0; b < a.nb; b++) {
			sum = fmaf(td[b], (float)tsum[b], sum);
		}
		y[r] = sum;
	}
}

// matmul_q8_0_det is matvec_q8_0_det for prompt batches: a 64-row x 64-token tile per block of 256 threads, each
// thread holding a 4x4 set of outputs (rows tr+16i, tokens tt+16j). Blocks are staged 4 at a time in shared memory;
// for each block, in order, a thread computes its 16 exact integer totals with dp4a and adds scale*total with one
// multiply-add each, as dotQ8_0Go does. Grid: (rows/64, tokens/64).
#define DTM 64 // tile rows and tokens
#define DKB 4  // blocks staged per step
#define DWS (DKB * 8 + 1) // shared words per row: DKB blocks of 8 words, plus one to spread the banks
extern "C" __global__ void __launch_bounds__(256) matmul_q8_0_det(const uchar *w, const uchar *x, float *y, DetMatvecArgs a) {
	__shared__ int sw[DTM * DWS], sx[DTM * DWS];
	__shared__ float dw[DKB][DTM], dx[DKB][DTM];
	uint tid = threadIdx.x, tr = tid / 16, tt = tid % 16;
	uint r0 = blockIdx.x * DTM, t0 = blockIdx.y * DTM;
	float acc[4][4];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			acc[i][j] = 0;
		}
	}
	for (uint b0 = 0; b0 < a.nb; b0 += DKB) {
		uint kb = min((uint)DKB, a.nb - b0);
		// load: DTM rows x kb blocks x 8 words, for weights and activations alike
		for (uint idx = tid; idx < DTM * DKB * 8; idx += 256) {
			uint rr = idx / (DKB * 8), k = idx % (DKB * 8), bb = k / 8, wd = k % 8;
			int wv = 0, xv = 0;
			if (bb < kb) {
				uint r = r0 + rr, t = t0 + rr;
				if (r < a.rows) {
					wv = i32at2(w + (size_t)r * a.rowBytes + (b0 + bb) * 34 + 2 + 4 * wd);
				}
				if (t < a.tokens) {
					xv = *(const int *)(x + (size_t)t * a.actBytes + (b0 + bb) * 36 + 4 + 4 * wd);
				}
			}
			sw[rr * DWS + k] = wv;
			sx[rr * DWS + k] = xv;
		}
		for (uint idx = tid; idx < DTM * DKB; idx += 256) {
			uint rr = idx / DKB, bb = idx % DKB;
			float vw = 0, vx = 0;
			if (bb < kb) {
				uint r = r0 + rr, t = t0 + rr;
				if (r < a.rows) {
					vw = f16(w + (size_t)r * a.rowBytes + (b0 + bb) * 34);
				}
				if (t < a.tokens) {
					vx = *(const float *)(x + (size_t)t * a.actBytes + (b0 + bb) * 36);
				}
			}
			dw[bb][rr] = vw;
			dx[bb][rr] = vx;
		}
		__syncthreads();
		for (uint bb = 0; bb < kb; bb++) {
			int s[4][4];
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					s[i][j] = 0;
				}
			}
			for (int wd = 0; wd < 8; wd++) {
				int wr[4], xr[4];
				for (int i = 0; i < 4; i++) {
					wr[i] = sw[(tr + 16 * i) * DWS + bb * 8 + wd];
					xr[i] = sx[(tt + 16 * i) * DWS + bb * 8 + wd];
				}
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						s[i][j] = __dp4a(wr[i], xr[j], s[i][j]);
					}
				}
			}
			for (int i = 0; i < 4; i++) {
				float d0 = dw[bb][tr + 16 * i];
				for (int j = 0; j < 4; j++) {
					acc[i][j] = fmaf(d0 * dx[bb][tt + 16 * j], (float)s[i][j], acc[i][j]);
				}
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			uint r = r0 + tr + 16 * i, t = t0 + tt + 16 * j;
			if (r < a.rows && t < a.tokens) {
				y[(size_t)t * a.rows + r] = acc[i][j];
			}
		}
	}
}

// Decodes the 6-bit scale and min of sub-block j from a Q4_K scales array (scaleMinK4 in quant.go).
__device__ __forceinline__ void det_scale_min_k4(int j, const uchar *q, int *sc, int *mn) {
	if (j < 4) {
		*sc = q[j] & 63;
		*mn = q[j + 4] & 63;
	} else {
		*sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
	}
}

// matvec_q4k_det is dotQ4KQ8KGo (quant.go) against block_q8_K activations, for DET_T tokens per warp. One warp per
// row; lane l holds elements 64j+l and 64j+32+l of each super-block (read once for all the tokens) and adds their
// exact integer products; lane t runs token t's two multiply-adds per super-block, in order.
extern "C" __global__ void matvec_q4k_det(const uchar *w, const uchar *x, float *y, DetMatvecArgs a) {
	uint sg = threadIdx.x / 32, lane = threadIdx.x % 32, nsg = blockDim.x / 32;
	uint r = blockIdx.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	uint t0 = blockIdx.y * DET_T, nt = min((uint)DET_T, a.tokens - t0);
	const uchar *row = w + (size_t)r * a.rowBytes;
	const uchar *xa = x + (size_t)t0 * a.actBytes;
	float sum = 0; // lane t < nt: token t0+t
	for (uint b = 0; b < a.nb; b++) {
		const uchar *blk = row + b * 144;
		const uchar *sc = blk + 4;
		const uchar *qs = blk + 16;
		float dw = f16(blk), dminw = f16(blk + 2);
		int lo[4], hi[4], s1[4], s2[4], m1[4], m2[4];
		for (int j = 0; j < 4; j++) {
			det_scale_min_k4(2 * j, sc, &s1[j], &m1[j]);
			det_scale_min_k4(2 * j + 1, sc, &s2[j], &m2[j]);
			uchar q = qs[32 * j + lane];
			lo[j] = q & 0xF, hi[j] = q >> 4;
		}
		for (uint t = 0; t < nt; t++) {
			const uchar *ab = xa + (size_t)t * a.actBytes + b * 292;
			const signed char *q8 = (const signed char *)(ab + 4);
			const short *bs = (const short *)(ab + 260);
			int part = 0, mins = 0;
			for (int j = 0; j < 4; j++) {
				part += lo[j] * (int)q8[64 * j + lane] * s1[j] + hi[j] * (int)q8[64 * j + 32 + lane] * s2[j];
				mins += m1[j] * ((int)bs[4 * j] + (int)bs[4 * j + 1]) + m2[j] * ((int)bs[4 * j + 2] + (int)bs[4 * j + 3]);
			}
			int sumi = warp_isum(part);
			if (lane == t) {
				float yd = *(const float *)ab;
				sum = fmaf(dw * yd, (float)sumi, sum);
				sum = fmaf(-(dminw * yd), (float)mins, sum);
			}
		}
	}
	if (lane < nt) {
		y[(size_t)(t0 + lane) * a.rows + r] = sum;
	}
}

// attention_det is forward's attention over the f32 KV cache (scores4Go, softmax, wsum4Go), one block of 128 threads
// per (query head, token). Scores: one multiply-add per d, in order; softmax: expf, the sum in position order;
// output: one multiply-add per position, in order. Layouts as attention_f32. Dynamic shared memory: headDim + 4 floats.
extern "C" __global__ void attention_det(const float *q, const float *kT, const float *v, float *out, float *sc,
                                         const uint2 *mask, AttnArgs a) {
	extern __shared__ float shm[];
	uint nt = blockDim.x, tid = threadIdx.x, lane = tid % 32, sg = tid / 32, nsg = nt / 32;
	uint h = blockIdx.x, t = blockIdx.y;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint ctx = a.pos + t + 1;
	uint2 m = mask[t];
	float *qs = shm;       // [headDim]
	float *red = shm + hd; // [nsg] partial maxima, then [0] = 1/sum
	float *s = sc + ((size_t)t * a.heads + h) * a.ctxP;
	const float *qh = q + ((size_t)t * a.heads + h) * hd;
	for (uint d = tid; d < hd; d += nt) {
		qs[d] = qh[d];
	}
	__syncthreads();

	const float *kh = kT + (size_t)kvh * hd * a.ctxP;
	float mx = -INFINITY;
	for (uint j = tid; j < ctx; j += nt) {
		float xv = -INFINITY;
		if (j < m.x || j >= m.y) {
			xv = 0;
			for (uint d = 0; d < hd; d++) {
				xv = fmaf(qs[d], kh[(size_t)d * a.ctxP + j], xv);
			}
			xv = xv * a.scale;
		}
		s[j] = xv;
		mx = fmaxf(mx, xv);
	}
	mx = warp_max(mx);
	if (lane == 0) {
		red[sg] = mx;
	}
	__syncthreads();
	mx = red[0];
	for (uint i = 1; i < nsg; i++) {
		mx = fmaxf(mx, red[i]);
	}
	for (uint j = tid; j < ctx; j += nt) {
		s[j] = expf_ref(s[j] - mx);
	}
	__syncthreads();
	if (tid == 0) {
		float sum = 0;
		uint j = 0;
		for (; j + 8 <= ctx; j += 8) { // loads first, then the adds in order
			float e[8];
			for (int i = 0; i < 8; i++) {
				e[i] = s[j + i];
			}
			for (int i = 0; i < 8; i++) {
				sum += e[i];
			}
		}
		for (; j < ctx; j++) {
			sum += s[j];
		}
		red[0] = 1.0f / sum;
	}
	__syncthreads();
	float inv = red[0];
	for (uint j = tid; j < ctx; j += nt) {
		s[j] = ftz(s[j] * inv); // as forward's softmax: probabilities can fall below 2^-126
	}
	__syncthreads();
	const float *vh = v + kvh * hd;
	for (uint d = tid; d < hd; d += nt) {
		float o = 0;
		uint j = 0;
		for (; j + 8 <= ctx; j += 8) { // loads first, then the multiply-adds in order
			float pj[8], vj[8];
			for (int i = 0; i < 8; i++) {
				pj[i] = s[j + i];
				vj[i] = vh[(size_t)(j + i) * kvd + d];
			}
			for (int i = 0; i < 8; i++) {
				o = fmaf(pj[i], vj[i], o);
			}
		}
		for (; j < ctx; j++) {
			o = fmaf(s[j], vh[(size_t)j * kvd + d], o);
		}
		out[((size_t)t * a.heads + h) * hd + d] = o;
	}
}

// ieee_probe_det checks the arithmetic the kernels above rely on (TestCUDAIEEE): IEEE float32 with subnormals.
extern "C" __global__ void ieee_probe_det(const float *a, const float *b, const float *c, float *o, uint n) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float x = a[i], y = b[i], z = c[i];
	float p = x * y;
	o[8 * i + 0] = p + z;
	o[8 * i + 1] = x * y + z; // not fused
	o[8 * i + 2] = fmaf(x, y, z);
	o[8 * i + 3] = x / y;
	o[8 * i + 4] = sqrtf(fabsf(x));
	o[8 * i + 5] = rintf(x * 1000.0f);
	o[8 * i + 6] = 1.0f / y;
	o[8 * i + 7] = x * y - z * x;
}
