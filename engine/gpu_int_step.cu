// A step (one token) on CUDA, as the reference's fast path runs it (llm/intgpu.py in compress13k): matvecs split
// along K into many small blocks whose int64 partial sums are added with integer atomics (exact in any order); the
// consumers (qkv, norm, MLP) apply the matvec's shift and zero the sums for the next step; attention adds into
// per-head totals with atomics and one merge divides. The same integers as gpu_int.metal's kernels, step by step.
// Included by gpu_int_mma.cu (helpers, attention structs).

typedef unsigned long long ull;

static __device__ __forceinline__ int bitlen64(long v) { return v == 0 ? 0 : 64 - __clzll(v); }
static __device__ __forceinline__ long labs64(long v) { return v < 0 ? -v : v; }

// exact a / b for a >= 0, b > 0: a double estimate (off by at most a few), corrected in integers
static __device__ __forceinline__ long udiv64(long a, long b) {
	if (b <= 0) {
		return 0; // never in a real step (Z includes the maximum's own weight); keeps profiling runs from hanging
	}
	long q = (long)((double)a / (double)b);
	long r = a - q * b;
	while (r < 0) {
		q--;
		r += b;
	}
	while (r >= b) {
		q++;
		r -= b;
	}
	return q;
}
static __device__ __forceinline__ long sdiv64(long a, long b) { return a < 0 ? -udiv64(-a, b) : udiv64(a, b); }

// a / b for a >= 0 given inv = 1.0 / b: a multiply for the estimate, then the same integer correction
static __device__ __forceinline__ long udivr(long a, long b, double inv) {
	long q = (long)((double)a * inv);
	long r = a - q * b;
	while (r < 0) {
		q--;
		r += b;
	}
	while (r >= b) {
		q++;
		r -= b;
	}
	return q;
}

static __device__ __forceinline__ long isqrt64(long v) {
	long r = (long)sqrt((double)v);
	while (r > 0 && r * r > v) {
		r--;
	}
	while ((r + 1) * (r + 1) <= v) {
		r++;
	}
	return r;
}

// block-wide max and sum (blockDim.x a multiple of 32); sm: 32 longs
static __device__ __forceinline__ long blk_max(long v, long *sm) {
	for (int o = 16; o > 0; o >>= 1) {
		v = max(v, shx(v, o));
	}
	if (threadIdx.x % 32 == 0) {
		sm[threadIdx.x / 32] = v;
	}
	__syncthreads();
	long r = (int)(threadIdx.x % 32) < (int)blockDim.x / 32 ? sm[threadIdx.x % 32] : LMIN; // each warp reduces the partials
	for (int o = 16; o > 0; o >>= 1) {
		r = max(r, shx(r, o));
	}
	__syncthreads();
	return r;
}
static __device__ __forceinline__ long blk_sum(long v, long *sm) {
	for (int o = 16; o > 0; o >>= 1) {
		v += shx(v, o);
	}
	if (threadIdx.x % 32 == 0) {
		sm[threadIdx.x / 32] = v;
	}
	__syncthreads();
	long r = (int)(threadIdx.x % 32) < (int)blockDim.x / 32 ? sm[threadIdx.x % 32] : 0;
	for (int o = 16; o > 0; o >>= 1) {
		r += shx(r, o);
	}
	__syncthreads();
	return r;
}

// 32 weights at w times 32 activations at xs (shared memory)
static __device__ __forceinline__ int dot32s(const signed char *w, const short *xs) {
	int4 wa = *(const int4 *)w, wb = *(const int4 *)(w + 16);
	const signed char *ws = (const signed char *)&wa, *wt = (const signed char *)&wb;
	int d = 0;
#pragma unroll
	for (int i = 0; i < 16; i++) {
		d += ws[i] * xs[i] + wt[i] * xs[16 + i];
	}
	return d;
}

// int_mv2: out[row] += sum over one 128-column tile of s[row, g] * (q[row, g] . x[g]), raw (unshifted). Grid
// (ceil(O / 64), K / 128), 256 threads: 64 rows x 4 quarters of 32 columns; a 64-group is two quarters. The tile's
// activations go through shared memory once: the mantissas x, or (p given) the MLP's raw p block-floated as int_blockf
// would (its shift from max |p|).
extern "C" __global__ void __launch_bounds__(256) int_mv2(const signed char *q, const short *s, const short *x, const long *p,
                                                          const long *pmax, ull *out, int O, int K) {
	__shared__ __align__(16) short xs[128];
	int tile = blockIdx.y, row = blockIdx.x * 64 + threadIdx.x / 4, qt = threadIdx.x % 4;
	int ng = K / 64, col = tile * 128 + qt * 32, r = min(row, O - 1);
	if (threadIdx.x < 128) {
		int c = tile * 128 + threadIdx.x;
		if (p) {
			int psh = max(bitlen64(*pmax) - (AQ - 1), 0);
			xs[threadIdx.x] = (short)clamp(rsh(p[c], psh), -LIMB(AQ), LIMB(AQ));
		} else {
			xs[threadIdx.x] = x[c];
		}
	}
	__syncthreads();
	int d = dot32s(q + (long)r * K + col, xs + qt * 32);
	d += __shfl_xor_sync(0xffffffffu, d, 1); // |.| < 64 * 128 * 32767 < 2^31
	long v = qt % 2 == 0 ? (long)d * s[(long)r * ng + tile * 2 + qt / 2] : 0;
	v += shx(v, 2);
	if (qt == 0 && row < O) {
		atomicAdd(out + row, (ull)v);
	}
}

// int_mvlog: the head: lg[row] = rshift(sum over K, -10 - (E + ex[0])). Grid ceil(V / 64), 256 threads; the whole
// activation vector in shared memory (K <= 4096).
extern "C" __global__ void __launch_bounds__(256) int_mvlog(const signed char *q, const short *s, const short *x, long *lg,
                                                            const int *ex, int O, int K, int E) {
	__shared__ __align__(16) short xs[4096];
	int row = blockIdx.x * 64 + threadIdx.x / 4, qt = threadIdx.x % 4;
	int ng = K / 64, r = min(row, O - 1);
	for (int i = threadIdx.x; i < K; i += 256) {
		xs[i] = x[i];
	}
	__syncthreads();
	long acc = 0;
	for (int tile = 0; tile < K / 128; tile++) {
		int col = tile * 128 + qt * 32;
		int d = dot32s(q + (long)r * K + col, xs + col);
		d += __shfl_xor_sync(0xffffffffu, d, 1);
		if (qt % 2 == 0) {
			acc += (long)d * s[(long)r * ng + tile * 2 + qt / 2];
		}
	}
	acc += shx(acc, 2);
	if (qt == 0 && row < O) {
		lg[row] = rsh(acc, -10 - (E + ex[0]));
	}
}

struct QKVArgs {
	int nq, nkv, hd, maxlen;
	long sc;
};

// int_qkv2: int_qkv from raw sums (x = rshift(acc, -24 - (E + ex[0])) + b), also writing the key and value planes,
// then zeroing the sums. One threadgroup of 1024 (nkv * hd <= 1024: a kv head's key pairs and values are whole warps,
// one round; a warp per q head).
extern "C" __global__ void __launch_bounds__(1024) int_qkv2(ull *acc, const long *b, const short *rc, const short *rs, const IntParams *pr, short *Km,
                                    int *Vc, unsigned char *Kp, unsigned char *Vp, int Mp, int *mq, int *shq, const int *ex,
                                    QKVArgs a, int E) {
	int tid = threadIdx.x, w = tid / 32, lane = tid % 32, NT = blockDim.x, NW = NT / 32;
	int pos = pr->pos, hh2 = a.hd / 2, n = -24 - (E + ex[0]), hd = a.hd;
	const short *c = rc + pos * hh2, *sn = rs + pos * hh2;
	const long LIM = (1L << 22) - 1;
#define X(i) (rsh((long)acc[i], n) + b[i])
	{
		__shared__ long kvm[2][32];
		bool isk = tid < a.nkv * hh2, isv = tid < a.nkv * hd;
		int kh = tid / hh2, j = tid % hh2, vh = tid / hd, d = tid % hd;
		long k1 = 0, k2 = 0, v = 0;
		if (isk) {
			int base = (a.nq + kh) * hd;
			long x1 = X(base + j), x2 = X(base + j + hh2), cj = c[j], sj = sn[j];
			k1 = clamp(rsh(rsh(x1 * cj - x2 * sj, 14), 12), -LIM, LIM);
			k2 = clamp(rsh(rsh(x2 * cj + x1 * sj, 14), 12), -LIM, LIM);
		}
		if (isv) {
			v = clamp(rsh(X((a.nq + a.nkv) * hd + tid), 12), -LIM, LIM);
		}
		long mk = max(labs64(k1), labs64(k2)), mv = labs64(v);
		for (int o = 16; o > 0; o >>= 1) {
			mk = max(mk, shx(mk, o));
			mv = max(mv, shx(mv, o));
		}
		if (lane == 0) {
			kvm[0][w] = mk;
			kvm[1][w] = mv;
		}
		__syncthreads();
		if (isk) {
			long m = 0;
			for (int g = kh * hh2 / 32; g < (kh + 1) * hh2 / 32; g++) {
				m = max(m, kvm[0][g]);
			}
			int sk = max(0, bitlen64(m) - (KQ - 1));
			long m1 = clamp(rsh(k1, sk), -LIMB(KQ), LIMB(KQ)), m2 = clamp(rsh(k2, sk), -LIMB(KQ), LIMB(KQ));
			k1 = m1 << sk;
			k2 = m2 << sk;
			long kb = ((long)kh * a.maxlen + pos) * hd;
			Km[kb + j] = (short)m1; // the key cache as gpu_int.metal lays it out (KSH)
			Km[kb + j + hh2] = (short)m2;
			if (j == 0) {
				((int *)(Km + (long)a.nkv * a.maxlen * hd))[kh * a.maxlen + pos] = sk;
			}
			for (int pl = 0; pl < 3; pl++) {
				long pb = (((long)kh * 3 + pl) * Mp + pos) * hd;
				Kp[pb + j] = (unsigned char)(pl < 2 ? (k1 >> (8 * pl)) & 255 : k1 >> 16);
				Kp[pb + j + hh2] = (unsigned char)(pl < 2 ? (k2 >> (8 * pl)) & 255 : k2 >> 16);
			}
		}
		if (isv) {
			long m = 0;
			for (int g = vh * hd / 32; g < (vh + 1) * hd / 32; g++) {
				m = max(m, kvm[1][g]);
			}
			int sv = max(0, bitlen64(m) - (VQ - 1));
			v = clamp(rsh(v, sv), -LIMB(VQ), LIMB(VQ)) << sv;
			Vc[((long)vh * a.maxlen + pos) * hd + d] = (int)v;
			for (int pl = 0; pl < 3; pl++) {
				Vp[(((long)vh * 3 + pl) * hd + d) * Mp + pos] = (unsigned char)(pl < 2 ? (v >> (8 * pl)) & 255 : v >> 16);
			}
		}
	}
	for (int h = w; h < a.nq; h += NW) { // warp per q head; lanes over its pairs (hh2 <= 64: 2 per lane)
		long qv[2][2], mx = 0;
		for (int t = 0; t < 2; t++) {
			int j = lane + 32 * t;
			if (j < hh2) {
				int base = h * hd;
				long x1 = X(base + j), x2 = X(base + j + hh2), cj = c[j], sj = sn[j];
				qv[t][0] = rsh(x1 * cj - x2 * sj, 14) * a.sc;
				qv[t][1] = rsh(x2 * cj + x1 * sj, 14) * a.sc;
				mx = max(mx, max(labs64(qv[t][0]), labs64(qv[t][1])));
			}
		}
		for (int o = 16; o > 0; o >>= 1) {
			mx = max(mx, shx(mx, o));
		}
		int sh = max(0, bitlen64(mx) - (QQ - 1));
		for (int t = 0; t < 2; t++) {
			int j = lane + 32 * t;
			if (j < hh2) {
				mq[h * hd + j] = (int)clamp(rsh(qv[t][0], sh), -LIMB(QQ), LIMB(QQ));
				mq[h * hd + j + hh2] = (int)clamp(rsh(qv[t][1], sh), -LIMB(QQ), LIMB(QQ));
			}
		}
		if (lane == 0) {
			shq[h] = sh;
		}
	}
#undef X
	__syncthreads();
	for (int i = tid; i < (a.nq + 2 * a.nkv) * hd; i += NT) {
		acc[i] = 0;
	}
}

// int_norm2: h += rshift(acc, -24 - (E + ein)) (add; acc zeroed), ein = ex[einSlot] or, when pmax is given, the
// exponent of the MLP's block-floated p (-16 + its shift; pmax reset); then rmsnorm as int_norm -> m, ex[0]. One
// threadgroup of 256 (D <= 4096).
// one-sync block reductions for norm2: each warp's partial to its own slot of a (fresh) array, then every thread
// reduces the NT / 32 partials itself
template <int NT> static __device__ __forceinline__ long red_max(long v, long *arr) {
	for (int o = 16; o > 0; o >>= 1) {
		v = max(v, shx(v, o));
	}
	if (threadIdx.x % 32 == 0) {
		arr[threadIdx.x / 32] = v;
	}
	__syncthreads();
	long r = arr[0];
#pragma unroll
	for (int k = 1; k < NT / 32; k++) {
		r = max(r, arr[k]);
	}
	return r;
}
template <int NT> static __device__ __forceinline__ long red_sum(long v, long *arr) {
	for (int o = 16; o > 0; o >>= 1) {
		v += shx(v, o);
	}
	if (threadIdx.x % 32 == 0) {
		arr[threadIdx.x / 32] = v;
	}
	__syncthreads();
	long r = 0;
#pragma unroll
	for (int k = 0; k < NT / 32; k++) {
		r += arr[k];
	}
	return r;
}

// each thread's (up to 4) elements stay in registers: h, acc and g are read once, up front
template <int NT>
static __device__ __forceinline__ void norm2(long *h, ull *acc, const long *g, short *m, int *ex, int D, int E, int einSlot,
                                             long *pmax, int add) {
	__shared__ long r1[NT / 32], r2[NT / 32], r3[NT / 32];
	int tid = threadIdx.x;
	long hv[4], gv[4], av[4];
	long pm = add && pmax ? *pmax : 0;
	int es = add && !pmax ? ex[einSlot] : 0;
#pragma unroll
	for (int k = 0; k < 4; k++) {
		int i = tid + NT * k;
		hv[k] = i < D ? h[i] : 0;
		gv[k] = i < D ? g[i] : 0;
		av[k] = add && i < D ? (long)acc[i] : 0;
	}
	if (add) {
		int ein = pmax ? max(bitlen64(pm) - (AQ - 1), 0) - 16 : es;
		int n = -24 - (E + ein);
#pragma unroll
		for (int k = 0; k < 4; k++) {
			int i = tid + NT * k;
			if (i < D) {
				hv[k] += rsh(av[k], n);
				h[i] = hv[k];
				acc[i] = 0;
			}
		}
	}
	long mx = 0;
#pragma unroll
	for (int k = 0; k < 4; k++) {
		mx = max(mx, labs64(hv[k]));
	}
	int sh = max(0, bitlen64(red_max<NT>(mx, r1)) - 20);
	if (add && pmax && tid == 0) {
		*pmax = 0; // every thread has read it (before the barrier in red_max)
	}
	long ss = 0;
#pragma unroll
	for (int k = 0; k < 4; k++) {
		long x = rsh(hv[k], sh);
		ss += x * x;
	}
	long ms = udiv64(red_sum<NT>(ss, r2), D) + udiv64(1L << max(0, 48 - 2 * sh), 1000000);
	long r = max(1L, isqrt64(ms));
	long y[4];
	long ym = 0;
	double inv = 1.0 / (double)r;
#pragma unroll
	for (int k = 0; k < 4; k++) {
		int i = tid + NT * k;
		long a = i < D ? rsh(hv[k], sh) * gv[k] * 16384 : 0, qa = udivr(labs64(a), r, inv);
		y[k] = a < 0 ? -qa : qa;
		ym = max(ym, labs64(y[k]));
	}
	int sh2 = max(0, bitlen64(red_max<NT>(ym, r3)) - (AQ - 1));
#pragma unroll
	for (int k = 0; k < 4; k++) {
		int i = tid + NT * k;
		if (i < D) {
			m[i] = (short)clamp(rsh(y[k], sh2), -LIMB(AQ), LIMB(AQ));
		}
	}
	if (tid == 0) {
		ex[0] = sh2 - 30;
	}
}

// D <= 4 * threads
extern "C" __global__ void __launch_bounds__(1024) int_norm2(long *h, ull *acc, const long *g, short *m, int *ex, int D, int E,
                                                             int einSlot, long *pmax, int add) {
	norm2<1024>(h, acc, g, m, ex, D, E, einSlot, pmax, add);
}
extern "C" __global__ void __launch_bounds__(256) int_norm2s(long *h, ull *acc, const long *g, short *m, int *ex, int D, int E,
                                                            int einSlot, long *pmax, int add) {
	norm2<256>(h, acc, g, m, ex, D, E, einSlot, pmax, add);
}

// int_mlp2: p = rshift(a * u, 16) from the gate/up sums (zeroed), max |p| into pmax (atomic). Grid ceil(I / 256).
extern "C" __global__ void int_mlp2(ull *acc, const int *sig, long *p, long *pmax, const int *ex, int I, int E) {
	__shared__ long sm[32];
	int i = blockIdx.x * 256 + threadIdx.x;
	const long NS = 16384;
	long v = 0;
	if (i < I) {
		int n = -16 - (E + ex[0]);
		long g = rsh((long)acc[i], n), u = rsh((long)acc[I + i], n);
		acc[i] = 0;
		acc[I + i] = 0;
		long av = rsh(g * (long)sig[clamp(rsh(g, 6), -NS, NS) + NS], 16);
		v = rsh(av * u, 16);
		p[i] = v;
	}
	long mx = blk_max(labs64(v), sm);
	if (threadIdx.x == 0) {
		atomicMax((ull *)pmax, (ull)mx);
	}
}

// int_mergea: o = O / Z (e = -24) per head and dimension from the atomic totals (zeroed), then blockf -> m, ex[1].
// One threadgroup of 1024; each head's Z and its reciprocal once, in shared memory.
extern "C" __global__ void __launch_bounds__(1024) int_mergea(ull *Oacc, ull *Zacc, long *o, short *m, int *ex, int nq, int hd) {
	__shared__ long sm[32];
	__shared__ long zs[64];
	__shared__ double zi[64];
	int tid = threadIdx.x, n = nq * hd;
	if (tid < nq) {
		zs[tid] = (long)Zacc[tid];
		zi[tid] = 1.0 / (double)zs[tid];
	}
	__syncthreads();
	long mx = 0;
	for (int i = tid; i < n; i += 1024) {
		long O = (long)Oacc[i], Z = zs[i / hd];
		double inv = zi[i / hd];
		long aO = labs64(O), hi = Z > 0 ? udivr(aO, Z, inv) : 0;
		long r = Z > 0 ? (hi << 12) + udivr((aO - hi * Z) << 12, Z, inv) : 0;
		long v = O < 0 ? -r : r;
		o[i] = v;
		mx = max(mx, labs64(v));
	}
	int sh = max(0, bitlen64(blk_max(mx, sm)) - (AQ - 1));
	for (int i = tid; i < n; i += 1024) {
		m[i] = (short)clamp(rsh(o[i], sh), -LIMB(AQ), LIMB(AQ));
		Oacc[i] = 0;
	}
	if (tid < nq) {
		Zacc[tid] = 0;
	}
	if (tid == 0) {
		ex[1] = sh - 24;
	}
}

// int_find2: an arithmetic decoder's search, on the device, for pipelined decoding. st = (num, den): the target count
// is (num * total - 1) / den (unsigned, as the decoder computes it), total = cum[V - 1]; the token is the first t
// with cum[t] > target (V: none). res = (t, cum[t], cum[t - 1], total), and params (the step's int32 token and
// position) become (t, position + 1), so the next step can start without the host. One block of 1024 threads, two
// rounds: segments of S = ceil(V / 1024) (cum is nondecreasing: the count of segment ends <= target is the segment
// holding t), then that segment's elements.
extern "C" __global__ void __launch_bounds__(1024) int_find2(const long *cum, const unsigned long long *st, long *res,
                                                             int *params, int V) {
	int tid = threadIdx.x;
	long total = cum[V - 1];
	long target = (long)(((unsigned long long)st[0] * (unsigned long long)total - 1ULL) / st[1]);
	int S = (V + 1023) / 1024;
	int e = min((tid + 1) * S, V) - 1;
	int c = __syncthreads_count(tid * S < V && cum[e] <= target); // whole segments below t
	int t = V;
	if (c * S < V) {
		int i = c * S + tid;
		int n = __syncthreads_count(tid < S && i < V && cum[i] <= target);
		t = c * S + n;
	}
	if (tid == 0) {
		res[0] = t;
		res[1] = t < V ? cum[t] : 0;
		res[2] = t > 0 && t <= V ? cum[t - 1] : 0;
		res[3] = total;
		params[0] = t < V ? t : 0;
		params[1] += 1;
	}
}
