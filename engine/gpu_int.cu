// Generated from gpu_int.metal by internal/gencu (go generate ./engine). Do not edit.

#include "gpu_int_prelude.cuh"

// Integer models (intmodel.go) on Metal: the same integer arithmetic as the CPU, so the same results, bit for bit.
// Every value is an integer; 64-bit where the CPU uses int64. Reductions are integer sums and maxima, which do not
// depend on order.

#define TG 1024
#define LMIN (-9223372036854775807L - 1)
// mantissa widths (SPEC §2-§3): activations into a matrix product, attention queries, cached keys and values
#define AQ 12
#define QQ 12
#define KQ 16
#define VQ 12
#define LIMB(b) ((1L << ((b) - 1)) - 1)

// the attention window (SPEC §3.2.5): position T-1 sees key t when win == 0, t < sink or t >= T - win; kvis for one
// key, cvis for any of the keys [k0, k1) and any of the positions T0-1 .. T1-1
static __device__ __forceinline__ bool kvis(int t, int T, int win, int sink) { return win == 0 || t < sink || t >= T - win; }
static __device__ __forceinline__ bool cvis(int k0, int k1, int T0, int win, int sink) { return win == 0 || k0 < sink || k1 > T0 - win; }

// A layer's key cache: KQ-bit mantissas Km[nkv][maxlen][hd] (short), then each key vector's shift [nkv][maxlen]
// (int): key = mantissa << shift. A query (QQ = 12 bits) times a mantissa is below 2^26 and 32 such products fit an
// int: one multiply per term, folded into 64 bits every 32 dimensions. The score shift absorbs the key's shift:
// rshift((q . m) << sk, n) = rshift(q . m, n - sk).
#define KSH(Km, nkv, maxlen, hd) ((int *)((Km) + (long)(nkv) * (maxlen) * (hd)))
#define KSHR(Km, nkv, maxlen, hd) ((const int *)((Km) + (long)(nkv) * (maxlen) * (hd)))

static __device__ __forceinline__ long rsh(long x, int n) { return n <= 0 ? (x << (-n)) : ((x + (1L << (n - 1))) >> n); }

static __device__ __forceinline__ int bitlen(long v) { return v == 0 ? 0 : 64 - (int)clz((ulong)v); }

// long <-> float through native 32-bit conversions (the 64-bit ones are emulated). Approximations, used only for
// estimates that integer arithmetic then corrects.
static __device__ __forceinline__ float l2f(long a) { return (float)(int)(a >> 32) * 4294967296.0f + (float)(uint)a; }
static __device__ __forceinline__ long f2l(float f) {
	if (fabs(f) < 2147483648.0f) {
		return (long)(int)f;
	}
	float hi = floor(f * (1.0f / 4294967296.0f));
	float lo = clamp(f - hi * 4294967296.0f, 0.0f, 4294967295.0f);
	return ((long)(int)hi << 32) + (long)(uint)lo;
}

// a / b for a >= 0, b > 0, exactly: a float estimate, refined once, then corrected in integers. Apple GPUs emulate
// 64-bit division, which costs hundreds of cycles; this is a few float ops and multiplies. b <= 0 never happens with
// correct inputs (an attention sum includes its maximum's weight); it gives 0 rather than looping forever.
static __device__ __forceinline__ long udivx(long a, long b) {
	if (b <= 0) {
		return 0;
	}
	float inv = 1.0f / l2f(b);
	long q = f2l(l2f(a) * inv);
	long r = a - q * b;
	q += f2l(l2f(r) * inv);
	r = a - q * b;
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

// a / b truncated toward zero (C and Go division), b > 0
static __device__ __forceinline__ long sdivx(long a, long b) { return a < 0 ? -udivx(-a, b) : udivx(a, b); }

static __device__ __forceinline__ long iabs(long v) { return v < 0 ? -v : v; }

// reductions within a simdgroup (32 lanes on Apple GPUs), by integer shuffles; MSL shuffles no 64-bit types, so a
// long travels as its two 32-bit halves.
static __device__ __forceinline__ long sg_sum(long v) {
	for (ushort o = 16; o > 0; o >>= 1) {
		v += shx(v, o);
	}
	return v;
}
static __device__ __forceinline__ long sg_max(long v) {
	for (ushort o = 16; o > 0; o >>= 1) {
		v = max(v, shx(v, o));
	}
	return v;
}

// max over a TG-thread threadgroup (tg: at least TG/32 longs of threadgroup memory): simdgroups first, then their
// TG/32 = 32 partials, which every simdgroup reduces for itself.
static __device__ __forceinline__ long tg_max(long v, threadgroup long *tg, uint tid) {
	v = sg_max(v);
	if ((tid & 31) == 0) {
		tg[tid >> 5] = v;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long r = sg_max(tg[tid & 31]);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}

static __device__ __forceinline__ long isqrt_exact(long v) {
	long r = f2l(sqrt(l2f(v)));
	while (r > 0 && r * r > v) {
		r--;
	}
	while ((r + 1) * (r + 1) <= v) {
		r++;
	}
	return r;
}

// reductions over an NT-thread threadgroup, NT a multiple of 32 up to 1024
static __device__ __forceinline__ long tgn_max(long v, threadgroup long *tg, uint tid, uint nt) {
	v = sg_max(v);
	if ((tid & 31) == 0) {
		tg[tid >> 5] = v;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long r = sg_max((tid & 31) < nt / 32 ? tg[tid & 31] : LMIN);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}
static __device__ __forceinline__ long tgn_sum(long v, threadgroup long *tg, uint tid, uint nt) {
	v = sg_sum(v);
	if ((tid & 31) == 0) {
		tg[tid >> 5] = v;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long r = sg_sum((tid & 31) < nt / 32 ? tg[tid & 31] : 0);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}

struct IntParams {
	int tok;
	int pos;
};

// h = rshift(q[tok] * s[tok], -24 - E), row y of the grid: token toks[y] into h[y] (a step binds the params,
// whose first int is the token)
extern "C" __global__ void int_embed(const signed char *q, const short *s, const int *toks, long *h, int D, int E) {
	uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int k = gid.x;
	if (k >= D) {
		return;
	}
	long tok = toks[gid.y];
	h[(long)gid.y * D + k] = rsh((long)q[tok * D + k] * (long)s[tok * (D / 64) + k / 64], -24 - E);
}

// optionally h += acc (already at e = -24); then rmsnorm(h, g) -> mantissas m, exponent eout. One threadgroup of
// any size; each pass rereads h.
extern "C" __global__ void int_norm(long *h, const long *acc, const long *g, short *m, int *ex, int D, int E, int einSlot, int eoutSlot, int add, int2 st) {
	uint row = blockIdx.x;
	uint tid = threadIdx.x;
	uint nt = blockDim.x;
	__shared__ __align__(16) long tg[32];
	h += (long)row * D;
	acc += (long)row * st.x;
	m += (long)row * st.y;
	ex += 4 * row;
	int mx = 0;
	for (int i = tid; i < D; i += nt) {
		long v = h[i];
		if (add) {
			v += acc[i]; // the matvec already shifted it to e = -24
			h[i] = v;
		}
		mx = max(mx, bitlen(iabs(v)));
	}
	threadgroup_barrier(mem_flags::mem_device);
	int sh = max(0, (int)tgn_max(mx, tg, tid, nt) - 20); // bitlen of the max = the max of the bitlens
	long ss = 0;
	for (int i = tid; i < D; i += nt) {
		long x = rsh(h[i], sh);
		ss += x * x;
	}
	long ms = udivx(tgn_sum(ss, tg, tid, nt), D) + udivx(1L << max(0, 48 - 2 * sh), 1000000);
	long r = max(1L, isqrt_exact(ms));
	// y = x * g * 16384 / r, kept for the second pass (the division is the costly part); NY per thread at most
	const int NY = 16; // D <= 16 * threads
	long y[NY];
	int ym = 0;
	for (int j = 0; j < NY; j++) {
		int i = tid + j * nt;
		y[j] = i < D ? sdivx(rsh(h[i], sh) * g[i] * 16384, r) : 0;
		ym = max(ym, bitlen(iabs(y[j])));
	}
	int sh2 = max(0, (int)tgn_max(ym, tg, tid, nt) - (AQ - 1));
	for (int j = 0; j < NY; j++) {
		int i = tid + j * nt;
		if (i < D) {
			m[i] = (short)clamp(rsh(y[j], sh2), -LIMB(AQ), LIMB(AQ));
		}
	}
	if (tid == 0) {
		ex[eoutSlot] = sh2 - 30;
	}
}

// out[row] = rshift(sum_g s[row,g] * sum_k q[row,k] x[k], to - (E + ein)): one row per simdgroup (many simdgroups in flight hide memory latency). Each step covers 512 columns:
// lane l reads bytes 16l..16l+15 and the matching activations; a 64-group is 4 lanes, summed by shuffles in int32
// (|sum| < 64 * 127 * 32767 < 2^31), scaled in 64 bits; the lanes' totals are summed at the end.
extern "C" __global__ void int_matvec(const signed char *q, const short *s, const short *x, long *out, const int *ex, int rows, int K, int E, int einSlot, int to) {
	uint tg_id = blockIdx.x;
	uint sg = threadIdx.x / 32;
	uint lane = threadIdx.x % 32;
	int row = tg_id * 8 + sg;
	if (row >= rows) {
		return;
	}
	int ng = K / 64;
	const signed char *qr = q + (long)row * K;
	const short *sr = s + (long)row * ng;
	long acc = 0;
#pragma unroll 4
	for (int col = 16 * lane; col < K + 16 * 31; col += 512) {
		bool in = col < K;
		int d = 0;
		if (in) {
			const short4 *xp = (const short4 *)(x + col);
			const char4 *wp = (const char4 *)(qr + col);
			int4 t = int4(wp[0]) * int4(xp[0]) + int4(wp[1]) * int4(xp[1]) + int4(wp[2]) * int4(xp[2]) +
			         int4(wp[3]) * int4(xp[3]);
			d = t.x + t.y + t.z + t.w;
		}
		d += simd_shuffle_xor(d, 1);
		d += simd_shuffle_xor(d, 2);
		if (in && (lane & 3) == 0) {
			acc += (long)d * (long)sr[col / 64];
		}
	}
	acc = sg_sum(acc);
	if (lane == 0) {
		out[row] = rsh(acc, to - (E + ex[einSlot]));
	}
}

struct QKVArgs {
	int nq, nkv, hd, maxlen;
	long sc; // attention scale, folded into q
};

// x = rshift(acc, -24 - (E + ein)) + b; rope q and k at pos; K, V = clip(rshift(., 12)), each vector rounded to KQ /
// VQ-bit mantissas with one exponent, into the cache (mantissa << shift); q heads: blockf(q * sc) -> mq, shq (QQ
// bits). One threadgroup of TG threads; a kv head's keys (hd/2 pairs) and values (hd) are whole simdgroups
// (nkv * hd <= TG); simdgroup h handles q head h.
extern "C" __global__ void int_qkv(const long *acc, const long *b, const short *rc, const short *rs, const IntParams *pr, short *Km, int *Vc, int *mq, int *shq, const int *ex, QKVArgs a, int E, int einSlot, int accS) {
	uint row = blockIdx.x;
	uint tid = threadIdx.x;
	uint sg = threadIdx.x / 32;
	uint lane = threadIdx.x % 32;
	__shared__ __align__(16) long part[32][32];
	__shared__ __align__(16) long kvm[2][32];
	// row r of a batch: position pos + r
	acc += (long)row * accS;
	mq += (long)row * a.nq * a.hd;
	shq += row * a.nq;
	int pos = pr->pos + row, hh2 = a.hd / 2; // acc is already at e = -24 (the matvec shifted it)
	const short *c = rc + pos * hh2;
	const short *sn = rs + pos * hh2;
	const long LIM = (1L << 22) - 1;
	// k pairs: threads 0 .. nkv*hh2-1; values: threads 0 .. nkv*hd-1
	{
		bool isk = tid < (uint)(a.nkv * hh2), isv = tid < (uint)(a.nkv * a.hd);
		int khh = tid / hh2, kj = tid % hh2, vhh = tid / a.hd;
		long k1 = 0, k2 = 0, v = 0;
		if (isk) {
			int base = (a.nq + khh) * a.hd;
			long x1 = acc[base + kj] + b[base + kj], x2 = acc[base + kj + hh2] + b[base + kj + hh2];
			long cj = c[kj], sj = sn[kj];
			k1 = clamp(rsh(rsh(x1 * cj - x2 * sj, 14), 12), -LIM, LIM);
			k2 = clamp(rsh(rsh(x2 * cj + x1 * sj, 14), 12), -LIM, LIM);
		}
		if (isv) {
			int idx = (a.nq + a.nkv) * a.hd + tid;
			v = clamp(rsh(acc[idx] + b[idx], 12), -LIM, LIM);
		}
		long mk = sg_max(max(iabs(k1), iabs(k2))), mv = sg_max(iabs(v));
		if (lane == 0) {
			kvm[0][sg] = mk;
			kvm[1][sg] = mv;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (isk) {
			long m = 0;
			for (int g = khh * hh2 / 32; g < (khh + 1) * hh2 / 32; g++) {
				m = max(m, kvm[0][g]);
			}
			int sk = max(0, bitlen(m) - (KQ - 1));
			long kb = ((long)khh * a.maxlen + pos) * a.hd;
			Km[kb + kj] = (short)clamp(rsh(k1, sk), -LIMB(KQ), LIMB(KQ));
			Km[kb + kj + hh2] = (short)clamp(rsh(k2, sk), -LIMB(KQ), LIMB(KQ));
			if (kj == 0) {
				KSH(Km, a.nkv, a.maxlen, a.hd)[khh * a.maxlen + pos] = sk;
			}
		}
		if (isv) {
			long m = 0;
			for (int g = vhh * a.hd / 32; g < (vhh + 1) * a.hd / 32; g++) {
				m = max(m, kvm[1][g]);
			}
			int sv = max(0, bitlen(m) - (VQ - 1));
			Vc[((long)vhh * a.maxlen + pos) * a.hd + tid % a.hd] = (int)(clamp(rsh(v, sv), -LIMB(VQ), LIMB(VQ)) << sv);
		}
	}
	// q heads: simdgroup sg handles head sg (nq <= 32); lanes over the head's pairs
	long qv[4][2]; // hh2 <= 128 -> up to 4 pairs per lane
	long mx = 0;
	int hq = sg;
	if (hq < a.nq) {
		for (int t = 0; t < 4; t++) {
			int j = lane + t * 32;
			if (j < hh2) {
				int base = hq * a.hd;
				long x1 = acc[base + j] + b[base + j], x2 = acc[base + j + hh2] + b[base + j + hh2];
				long cj = c[j], sj = sn[j];
				long q1 = rsh(x1 * cj - x2 * sj, 14) * a.sc, q2 = rsh(x2 * cj + x1 * sj, 14) * a.sc;
				qv[t][0] = q1;
				qv[t][1] = q2;
				mx = max(mx, max(iabs(q1), iabs(q2)));
			}
		}
	}
	part[sg][lane] = mx;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long hm = 0;
	for (int i = 0; i < 32; i++) {
		hm = max(hm, part[sg][i]);
	}
	if (hq < a.nq) {
		int sh = max(0, bitlen(hm) - (QQ - 1));
		for (int t = 0; t < 4; t++) {
			int j = lane + t * 32;
			if (j < hh2) {
				mq[hq * a.hd + j] = (int)clamp(rsh(qv[t][0], sh), -LIMB(QQ), LIMB(QQ));
				mq[hq * a.hd + j + hh2] = (int)clamp(rsh(qv[t][1], sh), -LIMB(QQ), LIMB(QQ));
			}
		}
		if (lane == 0) {
			shq[hq] = sh;
		}
	}
}

// blockf over n values -> mantissas m, exponent ex[eoutSlot] = ein + shift. One threadgroup.
extern "C" __global__ void int_blockf(const long *x, short *m, int *ex, int n, int ein, int eoutSlot, int2 st) {
	uint row = blockIdx.x;
	uint tid = threadIdx.x;
	__shared__ __align__(16) long tg[TG];
	x += (long)row * st.x;
	m += (long)row * st.y;
	ex += 4 * row;
	long mx = 0;
	for (int i = tid; i < n; i += TG) {
		mx = max(mx, iabs(x[i]));
	}
	int sh = max(0, bitlen(tg_max(mx, tg, tid)) - (AQ - 1));
	for (int i = tid; i < n; i += TG) {
		m[i] = (short)clamp(rsh(x[i], sh), -LIMB(AQ), LIMB(AQ));
	}
	if (tid == 0) {
		ex[eoutSlot] = ein + sh;
	}
}

// p = rshift(a * u, 16), a = rshift(g * SIG[clip(rshift(g, 6), -ns, ns) + ns], 16), with g, u = rshift(acc, -16 - (E+ein))
extern "C" __global__ void int_mlp(const long *acc, const int *sig, long *p, const int *ex, int I, int E, int einSlot, int2 st) {
	uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gid.x;
	if (i >= I) {
		return;
	}
	acc += (long)gid.y * st.x;
	p += (long)gid.y * st.y;
	const long NS = 16384;
	long g = acc[i], u = acc[I + i]; // the matvec already shifted them to e = -16
	long a = rsh(g * (long)sig[clamp(rsh(g, 6), -NS, NS) + NS], 16);
	p[i] = rsh(a * u, 16);
}

// The frequency table: f = EXPF[min(M - l, nf - 1)], Z = sum f, freq = f * (2^28 - V) / Z + 1, cum = prefix sums.
// TCH values per 256-thread group, 8 per
// thread; four passes: chunk maxima; M and chunk sums of f = EXPF[min(M - lg, nf - 1)]; Z, the counts and their scan
// within each chunk plus chunk totals; then each chunk adds the totals before it.
#define TCH 2048
extern "C" __global__ void int_table1(const long *lg, long *pm, int V) {
	uint tgi = blockIdx.x;
	uint tid = threadIdx.x;
	__shared__ __align__(16) long tg[32];
	long mx = LMIN;
	for (int i = tgi * TCH + tid; i < min((int)(tgi + 1) * TCH, V); i += 256) {
		mx = max(mx, lg[i]);
	}
	mx = tgn_max(mx, tg, tid, 256);
	if (tid == 0) {
		pm[tgi] = mx;
	}
}

static __device__ __forceinline__ long table_max(const long *pm, int nch, threadgroup long *tg, uint tid) {
	long mx = LMIN;
	for (int c = tid; c < nch; c += 256) {
		mx = max(mx, pm[c]);
	}
	return tgn_max(mx, tg, tid, 256);
}

extern "C" __global__ void int_table2(const long *lg, const int *expf_, long *pm, long *pz, long *cum, int V, int nf) {
	uint tgi = blockIdx.x;
	uint tid = threadIdx.x;
	uint nch = gridDim.x;
	__shared__ __align__(16) long tg[32];
	long M = table_max(pm, nch, tg, tid);
	long z = 0;
	for (int i = tgi * TCH + tid; i < min((int)(tgi + 1) * TCH, V); i += 256) {
		long f = expf_[min(M - lg[i], (long)(nf - 1))];
		cum[i] = f;
		z += f;
	}
	z = tgn_sum(z, tg, tid, 256);
	if (tid == 0) {
		pz[tgi] = z;
	}
}

extern "C" __global__ void int_table3(const long *pz, long *pt, long *cum, int V) {
	uint tgi = blockIdx.x;
	uint tid = threadIdx.x;
	uint nch = gridDim.x;
	__shared__ __align__(16) long tg[32];
	__shared__ __align__(16) long sc[256];
	long z = 0;
	for (int c = tid; c < (int)nch; c += 256) {
		z += pz[c];
	}
	long Z = tgn_sum(z, tg, tid, 256);
	int lo = tgi * TCH + tid * (TCH / 256), hi = min(lo + TCH / 256, V);
	long run = 0;
	for (int i = lo; i < hi; i++) {
		run += udivx(cum[i] * ((1L << 28) - V), Z) + 1;
		cum[i] = run;
	}
	// exclusive scan of the 256 thread totals (Hillis-Steele)
	sc[tid] = run;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint o = 1; o < 256; o <<= 1) {
		long v = tid >= o ? sc[tid - o] : 0;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		sc[tid] += v;
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	long off = sc[tid] - run;
	for (int i = lo; i < hi; i++) {
		cum[i] += off;
	}
	if (tid == 255) {
		pt[tgi] = sc[255];
	}
}

extern "C" __global__ void int_table4(const long *pt, long *cum, int V) {
	uint tgi = blockIdx.x;
	uint tid = threadIdx.x;
	__shared__ __align__(16) long tg[32];
	long o = 0;
	for (int c = tid; c < (int)tgi; c += 256) {
		o += pt[c];
	}
	o = tgn_sum(o, tg, tid, 256);
	for (int i = tgi * TCH + tid; i < min((int)(tgi + 1) * TCH, V); i += 256) {
		cum[i] += o;
	}
}

// Attention, split (as the CUDA reference does): chunks of ACH positions run in parallel; all integer, so exact.
// Pass 1: scores and chunk maxima; pass 2: weights, chunk sums Z and O; pass 3: o = sum O / sum Z.
#define ACH 256
struct AttnArgs {
	int nq, nkv, hd, maxlen, na, maxch, win, sink;
};

// Split attention by KV head: a threadgroup per (kv head, chunk) serves the grp = nq / nkv query heads that share it,
// so each K and V row is read once instead of grp times.
#define GMAX 8   // query heads per KV head, at most
#define HDMAX 128
extern "C" __global__ void int_attn1(const int *mq, const int *shq, const short *Km, long *S, long *cmax, const IntParams *pr, AttnArgs a) {
	uint2 tg2 = make_uint2(blockIdx.x, blockIdx.y);
	uint tid = threadIdx.x;
	__shared__ __align__(16) int qs[GMAX * HDMAX];
	int j = tg2.x, c = tg2.y, T = pr->pos + 1, hd = a.hd, grp = a.nq / a.nkv;
	if (c * ACH >= T || !cvis(c * ACH, c * ACH + ACH, T, a.win, a.sink)) {
		return; // a chunk past the position (a recorded launch list covers every chunk), or outside the window
	}
	for (int i = tid; i < grp * hd; i += ACH) {
		qs[i] = mq[j * grp * hd + i];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	int t = c * ACH + tid;
	long acc[GMAX];
	for (int g = 0; g < GMAX; g++) {
		acc[g] = 0;
	}
	int sk = 0;
	if (t < T) {
		const short *k = Km + ((long)j * a.maxlen + t) * hd;
		sk = KSHR(Km, a.nkv, a.maxlen, hd)[j * a.maxlen + t];
		for (int d0 = 0; d0 < hd; d0 += 32) {
			int as[GMAX];
			for (int g = 0; g < GMAX; g++) {
				as[g] = 0;
			}
			for (int d = d0; d < d0 + 32; d += 4) {
				int4 kk = int4(*(const short4 *)(k + d));
				for (int g = 0; g < GMAX; g++) {
					if (g < grp) {
						int4 pr4 = *(threadgroup const int4 *)(qs + g * hd + d) * kk;
						as[g] += pr4.x + pr4.y + pr4.z + pr4.w;
					}
				}
			}
			for (int g = 0; g < GMAX; g++) {
				acc[g] += as[g];
			}
		}
	}
	__shared__ __align__(16) long part[GMAX][ACH / 32];
	for (int g = 0; g < grp; g++) {
		int h = j * grp + g;
		long v = LMIN;
		if (t < T && kvis(t, T, a.win, a.sink)) {
			v = rsh(acc[g], 42 - shq[h] - sk);
			S[(long)h * a.maxlen + t] = v;
		}
		v = sg_max(v);
		if ((tid & 31) == 0) {
			part[g][tid >> 5] = v;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < (uint)grp) {
		long v = LMIN;
		for (int k = 0; k < ACH / 32; k++) {
			v = max(v, part[tid][k]);
		}
		cmax[(j * grp + tid) * a.maxch + c] = v;
	}
}

extern "C" __global__ void int_attn2(const long *S, const long *cmax, const int *Vc, const int *expa, long *zpart, long *opart, const IntParams *pr, AttnArgs a) {
	uint2 tg2 = make_uint2(blockIdx.x, blockIdx.y);
	uint tid = threadIdx.x;
	__shared__ __align__(16) int w[GMAX][ACH]; // weights, <= 2^15
	__shared__ __align__(16) long red[GMAX][ACH]; // O partials: (ACH / hd - 1) slices x hd lanes per head
	int j = tg2.x, c = tg2.y, T = pr->pos + 1, hd = a.hd, grp = a.nq / a.nkv;
	if (c * ACH >= T || !cvis(c * ACH, c * ACH + ACH, T, a.win, a.sink)) {
		return;
	}
	int nch = (T + ACH - 1) / ACH;
	int t = c * ACH + tid;
	// M per head: simdgroup g reduces head g's chunk maxima (ACH / 32 = 8 simdgroups >= grp)
	__shared__ __align__(16) long Ms[GMAX];
	__shared__ __align__(16) long part[GMAX][ACH / 32];
	uint sgi = tid >> 5, lane = tid & 31;
	if (sgi < (uint)grp) {
		long mx = LMIN;
		for (int i = lane; i < nch; i += 32) {
			if (cvis(i * ACH, i * ACH + ACH, T, a.win, a.sink)) {
				mx = max(mx, cmax[(j * grp + sgi) * a.maxch + i]);
			}
		}
		mx = sg_max(mx);
		if (lane == 0) {
			Ms[sgi] = mx;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (int g = 0; g < grp; g++) {
		long wt = t < T && kvis(t, T, a.win, a.sink)
		              ? (long)expa[min(Ms[g] - S[(long)(j * grp + g) * a.maxlen + t], (long)(a.na - 1))]
		              : 0;
		w[g][tid] = (int)wt;
		wt = sg_sum(wt);
		if (lane == 0) {
			part[g][sgi] = wt;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < (uint)grp) {
		long z = 0;
		for (int k = 0; k < ACH / 32; k++) {
			z += part[tid][k];
		}
		zpart[(j * grp + tid) * a.maxch + c] = z;
	}
	// O: ACH threads = (ACH / hd) slices x hd lanes over the chunk's positions, all grp heads at once.
	// w * v in 32-bit pieces: v = vh * 2^11 + vl; w <= 2^15, |v| < 2^22: products below 2^26, 16 to an int.
	int slices = ACH / hd, sl = tid / hd, d = tid % hd;
	int n = min(ACH, T - c * ACH);
	const int *V = Vc + ((long)j * a.maxlen + c * ACH) * hd + d;
	long odh[GMAX], odl[GMAX];
	for (int g = 0; g < GMAX; g++) {
		odh[g] = 0;
		odl[g] = 0;
	}
	for (int i0 = sl; i0 < n; i0 += 16 * slices) {
		int sh[GMAX], so[GMAX];
		for (int g = 0; g < GMAX; g++) {
			sh[g] = 0;
			so[g] = 0;
		}
		// 16 independent loads first (positions past the end have weight 0, and the cache is padded by a chunk)
		int vv[16];
		for (int k = 0; k < 16; k++) {
			vv[k] = V[(long)(i0 + k * slices) * hd];
		}
		for (int k = 0; k < 16; k++) {
			int i = i0 + k * slices;
			int vh = vv[k] >> 11, vl = vv[k] & 0x7ff;
			for (int g = 0; g < GMAX; g++) {
				if (g < grp) {
					int ww = w[g][i];
					sh[g] += ww * vh;
					so[g] += ww * vl;
				}
			}
		}
		for (int g = 0; g < GMAX; g++) {
			odh[g] += sh[g];
			odl[g] += so[g];
		}
	}
	if (slices > 1) {
		for (int g = 0; g < grp; g++) {
			if (sl > 0) {
				red[g][(sl - 1) * hd + d] = (odh[g] << 11) + odl[g];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if (sl == 0) {
		for (int g = 0; g < grp; g++) {
			long O = (odh[g] << 11) + odl[g];
			for (int k = 1; k < slices; k++) {
				O += red[g][(k - 1) * hd + d];
			}
			opart[((long)(j * grp + g) * a.maxch + c) * hd + d] = O;
		}
	}
}

// pass 3, one threadgroup per head (hd threads): sum the chunks' Z and O, o = O / Z (e = -24).
extern "C" __global__ void int_attn3(const long *zpart, const long *opart, long *o, const IntParams *pr, AttnArgs a) {
	uint h = blockIdx.x;
	uint tid = threadIdx.x;
	int T = pr->pos + 1, nch = (T + ACH - 1) / ACH;
	long Z = 0, O = 0;
	for (int c = 0; c < nch; c++) {
		if (cvis(c * ACH, c * ACH + ACH, T, a.win, a.sink)) {
			Z += zpart[h * a.maxch + c];
			O += opart[((long)h * a.maxch + c) * a.hd + tid];
		}
	}
	long aO = iabs(O), hi = udivx(aO, Z);
	long r = (hi << 12) + udivx((aO - hi * Z) << 12, Z);
	o[h * a.hd + tid] = O < 0 ? -r : r;
}

// ---- batches: many positions at once (compression knows every token in advance) ----

// int_gemm: out[t, o] = rshift(sum_g s[o,g] * sum_{k in g} q[o,k] x[t,k], to - (E + ex[4t + einSlot])) for the rows t
// of a batch; the same integers as int_matvec, row by row. The 64-group sums run in float: |q| <= 127 and |x| <= 2047
// (AQ = 12-bit activations), so every product and every partial sum of a group is an integer below 2^24, which
// float32 holds exactly whatever the order (and half holds the operands exactly). A threadgroup computes GBO outputs x
// GBT rows; each 64-group's weights and activations go through threadgroup memory as half; a thread holds 4 outputs x
// 4 rows. The group sum (an exact integer) is scaled in integers, split into 16-bit halves so each product fits 32
// bits. st = (unused, x row stride, out row stride, unused).
#define GBO 64
#define GBT 64
extern "C" __global__ void int_gemm(const signed char *q, const short *s, const short *x, long *out, const int *ex, int O, int K, int E, int einSlot, int to, int4 st) {
	uint2 tg2 = make_uint2(blockIdx.x, blockIdx.y);
	uint tid = threadIdx.x;
	__shared__ __align__(16) half W[GBO][64];
	__shared__ __align__(16) half X[GBT][64];
	__shared__ __align__(16) short S[GBO];
	int xS = st.y, outS = st.z;
	int o0 = tg2.x * GBO, t0 = tg2.y * GBT;
	int to4 = (tid % 16) * 4, tt4 = (tid / 16) * 4;
	long acc[4][4];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			acc[i][j] = 0;
		}
	}
	int ng = K / 64;
	int wr = tid >> 2, wc = (tid & 3) * 16; // each thread copies 16 weights and 16 activations per group
	const signed char *wsrc = q + (long)min(o0 + wr, O - 1) * K + wc;
	const short *xsrc = x + (long)(t0 + wr) * xS + wc;
	for (int g = 0; g < ng; g++) {
		{
			const char4 *src = (const char4 *)(wsrc + g * 64);
			threadgroup half4 *dst = (threadgroup half4 *)(&W[wr][wc]);
			const short4 *xs = (const short4 *)(xsrc + g * 64);
			threadgroup half4 *xd = (threadgroup half4 *)(&X[wr][wc]);
			for (int i = 0; i < 4; i++) {
				dst[i] = half4(src[i]);
				xd[i] = half4(xs[i]);
			}
		}
		if (tid < GBO) {
			S[tid] = s[(long)min(o0 + (int)tid, O - 1) * ng + g];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		float gs[4][4];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				gs[i][j] = 0;
			}
		}
		for (int k = 0; k < 64; k += 4) {
			float4 xv[4];
			for (int j = 0; j < 4; j++) {
				xv[j] = float4(*(threadgroup const half4 *)&X[tt4 + j][k]);
			}
			for (int i = 0; i < 4; i++) {
				float4 w = float4(*(threadgroup const half4 *)&W[to4 + i][k]);
				for (int j = 0; j < 4; j++) {
					gs[i][j] += dot(w, xv[j]);
				}
			}
		}
		for (int i = 0; i < 4; i++) {
			int sc = S[to4 + i];
			for (int j = 0; j < 4; j++) {
				int d = (int)gs[i][j];
				acc[i][j] += ((long)((d >> 16) * sc) << 16) + (long)((d & 0xffff) * sc);
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	for (int i = 0; i < 4; i++) {
		int o = o0 + to4 + i;
		if (o < O) {
			for (int j = 0; j < 4; j++) {
				int t = t0 + tt4 + j;
				out[(long)t * outS + o] = rsh(acc[i][j], to - (E + ex[4 * t + einSlot]));
			}
		}
	}
}

// Batched attention: a threadgroup per (query head, block of BR rows); rows are consecutive positions pos + t.
// Two passes over the keys in tiles of KT (through threadgroup memory): the row maxima, then the weights, Z and O.
// Score products as in int_attn1 (one multiply, folded every 32 terms); values in 32-bit pieces as in int_attn2.
// Rows past the batch's end compute garbage nobody reads; keys past a row's position are masked; the cache is padded
// so tiles may read past the last position.
template <int HD, int BR, int KT>
static __device__ __forceinline__ void attnb(const int *mq, const int *shq, const short *Km, const int *Vc,
                         const int *expa, long *o, int base, AttnArgs &a, uint h, uint blk,
                         uint tid, threadgroup int *qs, threadgroup int *ks, threadgroup int *vs, threadgroup int *wt,
                         threadgroup int *kss) {
	constexpr int TPR = 256 / BR, KPT = KT / TPR, DPT = HD / TPR, KS = HD + 4; // ks rows padded (banks)
	int grp = a.nq / a.nkv, j = h / grp;
	int t0 = blk * BR, r = tid / TPR, sub = tid % TPR;
	int row = t0 + r, p = base + row, pmax = base + t0 + BR - 1;
	for (int i = tid; i < BR * HD; i += 256) {
		qs[i] = mq[((long)(t0 + i / HD) * a.nq + h) * HD + i % HD];
	}
	int sh = shq[row * a.nq + h];
	const short *K = Km + (long)j * a.maxlen * HD;
	const int *Ksh = KSHR(Km, a.nkv, a.maxlen, HD) + (long)j * a.maxlen;
	const int *V = Vc + (long)j * a.maxlen * HD;
	threadgroup const int *qr = qs + r * HD;
	long M = LMIN;
	for (int k0 = 0; k0 <= pmax; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int i = tid; i < KT * HD; i += 256) {
			ks[(i / HD) * KS + i % HD] = K[(long)k0 * HD + i];
		}
		if (tid < (uint)KT) {
			kss[tid] = Ksh[k0 + tid];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int c = 0; c < KPT; c++) {
			int kk = c * TPR + sub;
			threadgroup const int *kr = ks + kk * KS;
			long acc = 0;
			for (int d0 = 0; d0 < HD; d0 += 32) {
				int as = 0;
				for (int d = d0; d < d0 + 32; d += 4) {
					int4 p4 = *(threadgroup const int4 *)(qr + d) * *(threadgroup const int4 *)(kr + d);
					as += p4.x + p4.y + p4.z + p4.w;
				}
				acc += as;
			}
			if (k0 + kk <= p && kvis(k0 + kk, p + 1, a.win, a.sink)) {
				M = max(M, rsh(acc, 42 - sh - kss[kk]));
			}
		}
	}
	for (ushort x = 1; x < TPR; x <<= 1) {
		M = max(M, shx(M, x));
	}
	long Z = 0;
	long od[DPT];
	for (int e = 0; e < DPT; e++) {
		od[e] = 0;
	}
	for (int k0 = 0; k0 <= pmax; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int i = tid; i < KT * HD; i += 256) {
			ks[(i / HD) * KS + i % HD] = K[(long)k0 * HD + i];
			vs[i] = V[(long)k0 * HD + i];
		}
		if (tid < (uint)KT) {
			kss[tid] = Ksh[k0 + tid];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int c = 0; c < KPT; c++) {
			int kk = c * TPR + sub;
			threadgroup const int *kr = ks + kk * KS;
			long acc = 0;
			for (int d0 = 0; d0 < HD; d0 += 32) {
				int as = 0;
				for (int d = d0; d < d0 + 32; d += 4) {
					int4 p4 = *(threadgroup const int4 *)(qr + d) * *(threadgroup const int4 *)(kr + d);
					as += p4.x + p4.y + p4.z + p4.w;
				}
				acc += as;
			}
			int w = 0;
			if (k0 + kk <= p && kvis(k0 + kk, p + 1, a.win, a.sink)) {
				w = expa[min(M - rsh(acc, 42 - sh - kss[kk]), (long)(a.na - 1))];
			}
			wt[r * KT + kk] = w;
			Z += w;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		// O: this thread's DPT dimensions of row r; w <= 2^15, |v| < 2^22 -> 16 products per int
		threadgroup const int *wr = wt + r * KT;
		for (int k1 = 0; k1 < KT; k1 += 16) {
			int shh[DPT], sl[DPT];
			for (int e = 0; e < DPT; e++) {
				shh[e] = 0;
				sl[e] = 0;
			}
			for (int kk = k1; kk < k1 + 16; kk++) {
				int w = wr[kk];
				threadgroup const int *vr = vs + kk * HD + sub * DPT;
				for (int e = 0; e < DPT; e++) {
					int v = vr[e];
					shh[e] += w * (v >> 11);
					sl[e] += w * (v & 0x7ff);
				}
			}
			for (int e = 0; e < DPT; e++) {
				od[e] += ((long)shh[e] << 11) + sl[e];
			}
		}
	}
	for (ushort x = 1; x < TPR; x <<= 1) {
		Z += shx(Z, x);
	}
	for (int e = 0; e < DPT; e++) {
		long O = od[e], aO = iabs(O), hi = udivx(aO, Z);
		long rr = (hi << 12) + udivx((aO - hi * Z) << 12, Z);
		o[((long)row * a.nq + h) * HD + sub * DPT + e] = O < 0 ? -rr : rr;
	}
}

#define ATTNB(NAME, HD, BR, KT)                                                                                            \
	extern "C" __global__ void NAME(const int *mq, const int *shq, const short *Km, const int *Vc, const int *expa, long *o, const IntParams *pr, AttnArgs a) { \
	uint2 tg2 = make_uint2(blockIdx.x, blockIdx.y); \
	uint tid = threadIdx.x;              \
		__shared__ __align__(16) int qs[BR * HD];                                                                                       \
		__shared__ __align__(16) int ks[KT * (HD + 4)];                                                                                 \
		__shared__ __align__(16) int vs[KT * HD];                                                                                       \
		__shared__ __align__(16) int wt[BR * KT];                                                                                       \
		__shared__ __align__(16) int kss[KT];                                                                                           \
		attnb<HD, BR, KT>(mq, shq, Km, Vc, expa, o, pr->pos, a, tg2.x, tg2.y, tid, qs, ks, vs, wt, kss);                  \
	}
ATTNB(int_attnb64, 64, 32, 32)
ATTNB(int_attnb128, 128, 16, 16)

// int_counts: for row t of a batch, the frequency table's total and the target token's cumulative counts:
// cnt[t] = (cum[V-1], cum[tgt[t]], cum[tgt[t]] - freq[tgt[t]]), cum as int_table1..4 build it. One threadgroup a row.
extern "C" __global__ void int_counts(const long *lg, const int *expf_, const int *tgt, long *cnt, int V, int nf) {
	uint row = blockIdx.x;
	uint tid = threadIdx.x;
	uint nt = blockDim.x;
	__shared__ __align__(16) long tg[32];
	lg += (long)row * V;
	long mx = LMIN;
	for (int i = tid; i < V; i += nt) {
		mx = max(mx, lg[i]);
	}
	long M = tgn_max(mx, tg, tid, nt);
	long z = 0;
	for (int i = tid; i < V; i += nt) {
		z += expf_[min(M - lg[i], (long)(nf - 1))];
	}
	long Z = tgn_sum(z, tg, tid, nt);
	int t = tgt[row];
	long C = (1L << 28) - V, tot = 0, hi = 0, ft = 0;
	for (int i = tid; i < V; i += nt) {
		long f = udivx(expf_[min(M - lg[i], (long)(nf - 1))] * C, Z) + 1;
		tot += f;
		if (i <= t) {
			hi += f;
		}
		if (i == t) {
			ft = f;
		}
	}
	tot = tgn_sum(tot, tg, tid, nt);
	hi = tgn_sum(hi, tg, tid, nt);
	ft = tgn_sum(ft, tg, tid, nt);
	if (tid == 0) {
		cnt[3 * row] = tot;
		cnt[3 * row + 1] = hi;
		cnt[3 * row + 2] = hi - ft;
	}
}

// Batched attention in three kernels (the batch's params: tok = its row count, pos = its first position), scores stored once (int32; a flag reports any that don't fit, and the caller
// then uses int_attnb*). KC keys per chunk; chunks run in parallel.
#define KC 1024
struct BAttnArgs {
	int nq, nkv, hd, maxlen, na, smax, nchmax, ez, win, sink, rows; // ez: EXPA[i] = 0 for i >= ez; rows: per batch
};

// the stored scores, key-major: tiles of 16 keys, each [head][row][16 keys], so the 16 lanes reading a row's tile in
// int_bpv* read one contiguous 64-byte run
#define SIDX(row, h, t, a) (((((long)((t) >> 4)) * (a).nq + (h)) * (a).rows + (row)) * 16 + ((t) & 15))

// pass 1: a threadgroup per (kv head, block of QR / grp rows, key chunk): the scores of every query head sharing the
// kv head (QR query rows: grp heads x rows) against the chunk's keys, in tiles of KT through threadgroup memory; a
// thread holds QR/32 query rows x KT/8 keys. S[SIDX(row, h, t)] = score; cmax = the chunk's maxima.
template <int HD, int QR, int KT>
static __device__ __forceinline__ void bscores(const int *mq, const int *shq, const short *Km, int *S,
                           long *cmax, int *ovf, int base, int R, BAttnArgs &a, uint3 tg3, uint tid,
                           threadgroup short *qs, threadgroup short *ks, threadgroup int *shs, threadgroup int *kss) {
	constexpr int TR = QR / 32, TK = KT / 8, QS = HD + 4, KS = HD + 4;
	int j = tg3.x, grp = a.nq / a.nkv, br = QR / grp; // rows per block
	int t0 = tg3.y * br, c = tg3.z;
	int pmax = base + min(t0 + br, R) - 1; // rows t0 + r < R only
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int qg = tid / 8, sub = tid % 8;
	if (!cvis(k0c, k0c + KC, base + t0 + 1, a.win, a.sink)) {
		return; // no row of the block sees the chunk (its readers skip it too)
	}
	for (int i = tid; i < QR * HD; i += 256) {
		int qr = i / HD, d = i % HD, g = qr / br, r = qr % br;
		qs[qr * QS + d] = g < grp && t0 + r < R ? (short)mq[((long)(t0 + r) * a.nq + j * grp + g) * HD + d] : (short)0;
	}
	if (tid < QR) {
		int g = tid / br, r = tid % br;
		shs[tid] = g < grp && t0 + r < R ? shq[(t0 + r) * a.nq + j * grp + g] : 0;
	}
	const short *K = Km + (long)j * a.maxlen * HD;
	const int *Ksh = KSHR(Km, a.nkv, a.maxlen, HD) + (long)j * a.maxlen;
	long mx[TR];
	for (int i = 0; i < TR; i++) {
		mx[i] = LMIN;
	}
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int i = tid; i < KT * HD / 4; i += 256) {
			int kk = i / (HD / 4), d = (i % (HD / 4)) * 4;
			*(threadgroup short4 *)(ks + kk * KS + d) = *(const short4 *)(K + (long)(k0 + kk) * HD + d);
		}
		if (tid < (uint)KT) {
			kss[tid] = Ksh[k0 + tid];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		long acc[TR][TK];
		for (int i = 0; i < TR; i++) {
			for (int k = 0; k < TK; k++) {
				acc[i][k] = 0;
			}
		}
		for (int d0 = 0; d0 < HD; d0 += 32) {
			int as[TR][TK];
			for (int i = 0; i < TR; i++) {
				for (int k = 0; k < TK; k++) {
					as[i][k] = 0;
				}
			}
			for (int d = d0; d < d0 + 32; d += 4) {
				int4 kv[TK];
				for (int k = 0; k < TK; k++) {
					kv[k] = int4(*(threadgroup const short4 *)(ks + (sub + 8 * k) * KS + d));
				}
				for (int i = 0; i < TR; i++) {
					int4 q = int4(*(threadgroup const short4 *)(qs + (qg + 32 * i) * QS + d));
					for (int k = 0; k < TK; k++) {
						int4 p4 = q * kv[k];
						as[i][k] += p4.x + p4.y + p4.z + p4.w;
					}
				}
			}
			for (int i = 0; i < TR; i++) {
				for (int k = 0; k < TK; k++) {
					acc[i][k] += as[i][k];
				}
			}
		}
		for (int i = 0; i < TR; i++) {
			int qr = qg + 32 * i, g = qr / br, r = qr % br;
			if (g >= grp || t0 + r >= R) {
				continue;
			}
			int p = base + t0 + r;
			for (int k = 0; k < TK; k++) {
				int t = k0 + sub + 8 * k;
				if (t <= p && kvis(t, p + 1, a.win, a.sink)) {
					long s = rsh(acc[i][k], 42 - shs[qr] - kss[sub + 8 * k]);
					if (s != (long)(int)s) {
						atomicExch(ovf, 1);
					}
					S[SIDX(t0 + r, j * grp + g, t, a)] = (int)s;
					mx[i] = max(mx[i], s);
				}
			}
		}
	}
	for (int i = 0; i < TR; i++) {
		long m = mx[i];
		for (ushort x = 1; x < 8; x <<= 1) {
			m = max(m, shx(m, x));
		}
		int qr = qg + 32 * i, g = qr / br, r = qr % br;
		if (sub == 0 && g < grp && t0 + r < R) {
			cmax[((long)(t0 + r) * a.nq + j * grp + g) * a.nchmax + c] = m;
		}
	}
}

#define BSCORES(NAME, HD, QR, KT)                                                                                          \
	extern "C" __global__ void NAME(const int *mq, const int *shq, const short *Km, int *S, long *cmax, int *ovf, const IntParams *pr, BAttnArgs a) { \
	uint3 tg3 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z); \
	uint tid = threadIdx.x;                                                          \
		__shared__ __align__(16) short qs[QR * (HD + 4)];                                                                               \
		__shared__ __align__(16) short ks[KT * (HD + 4)];                                                                               \
		__shared__ __align__(16) int shs[QR];                                                                                           \
		__shared__ __align__(16) int kss[KT];                                                                                           \
		bscores<HD, QR, KT>(mq, shq, Km, S, cmax, ovf, pr->pos, pr->tok, a, tg3, tid, qs, ks, shs, kss);                  \
	}
BSCORES(int_bscores64, 64, 128, 16)
BSCORES(int_bscores128, 128, 64, 16)

// pass 2: a threadgroup per (query head, block of BRV rows, key chunk): weights w = EXPA[min(M - s, na - 1)] (M from
// the chunk maxima), the chunk's Z and O = sum w V; a thread holds 4 rows x 4 dimensions. Partials to zp, op.
// EXPA's nonzero head (a.ez entries, <= ETN; the rest are 0) is copied into threadgroup memory as ushort: the weight
// lookups are random, and from memory they cost more than the arithmetic. Tiles whose weights are all 0 add
// nothing and are skipped.
#define ETN 11392
template <int HD, int KT, int ETS>
static __device__ __forceinline__ void bpv(const int *S, const long *cmax, const int *Vc, const int *expa,
                       long *zp, long *op, int base, int R, BAttnArgs &a, uint3 tg3, uint tid,
                       threadgroup int *vs, threadgroup ushort *wt, threadgroup long *Ms, threadgroup long *zs,
                       threadgroup ushort *et, threadgroup int *nz, threadgroup uchar *z8) {
	constexpr int DG = HD / 4, RG = 256 / DG, BRV = RG * 4, FK = KT < 16 ? KT : 16, RPS = 32 / KT;
	int h = tg3.x, t0 = tg3.y * BRV, c = tg3.z, grp = a.nq / a.nkv, j = h / grp;
	int pmax = base + t0 + BRV - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int ez = a.ez;
	if (!cvis(k0c, k0c + KC, base + t0 + 1, a.win, a.sink)) {
		return;
	}
	int ezs = min(ez, ETS); // the first ETS entries from threadgroup memory, any later ones from the device
	for (int i = tid; i < ezs; i += 256) {
		et[i] = (ushort)expa[i];
	}
	if (tid < (uint)BRV) {
		int p = base + t0 + tid, nch = p / KC + 1;
		const long *cm = cmax + ((long)(t0 + tid) * a.nq + h) * a.nchmax;
		long m = LMIN;
		for (int i = 0; i < nch; i++) {
			if (cvis(i * KC, i * KC + KC, p + 1, a.win, a.sink)) {
				m = max(m, cm[i]);
			}
		}
		Ms[tid] = m;
		zs[tid] = 0;
	}
	const int *V = Vc + (long)j * a.maxlen * HD;
	int rg = tid / DG, dg = tid % DG, g8 = rg / 2; // a simdgroup's rows lie in one 8-row group
	long od[4][4];
	for (int i = 0; i < 4; i++) {
		for (int e = 0; e < 4; e++) {
			od[i][e] = 0;
		}
	}
	uint sg = tid / 32, lane = tid % 32, key = lane % KT;
	constexpr int NR = BRV / (8 * RPS); // rows per thread in the weights phase (always the same rows)
	int zl[NR];                          // this lane's share of each row's Z, <= (KC / KT) * 2^15 < 2^31
	for (int i = 0; i < NR; i++) {
		zl[i] = 0;
	}
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid == 0) {
			*nz = 0;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		// weights: lanes cover RPS rows x KT keys; simdgroup sg does rows sg * RPS + lane / KT + 8 * RPS * i
		bool any = false;
		for (int i = 0; i < NR; i++) {
			int r = sg * RPS + lane / KT + 8 * RPS * i;
			int t = k0 + key, p = base + t0 + r;
			int w = 0;
			// rows past the batch's end hold stale scores: their indices could be anything
			if (t <= p && t0 + r < R && kvis(t, p + 1, a.win, a.sink)) {
				long d = Ms[r] - S[SIDX(t0 + r, h, t, a)];
				if (d >= 0 && d < ezs) {
					w = et[d];
				} else if (d >= ezs && d < ez) {
					w = expa[d];
				}
			}
			wt[r * KT + key] = (ushort)w;
			any = any || w != 0;
			zl[i] += w;
		}
		if (simd_any(any) && lane == 0) {
			*nz = 1;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (*nz == 0) {
			continue;
		}
		for (int i = tid; i < KT * HD; i += 256) {
			vs[i] = V[(long)k0 * HD + i];
		}
		// z8[g * KT + k]: rows 8g .. 8g+7 all have weight 0 for key k (most weights are 0: EXPA is 0 far below the max)
		for (int i = tid; i < (BRV / 8) * KT; i += 256) {
			int g = i / KT, k = i % KT;
			bool all0 = true;
			for (int r = 8 * g; r < 8 * g + 8; r++) {
				all0 = all0 && wt[r * KT + k] == 0;
			}
			z8[i] = all0;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int k1 = 0; k1 < KT; k1 += FK) {
			int4 sh[4], sl[4];
			for (int i = 0; i < 4; i++) {
				sh[i] = 0;
				sl[i] = 0;
			}
			for (int k = k1; k < k1 + FK; k++) {
				if (z8[g8 * KT + k]) {
					continue; // the same for the whole simdgroup: its rows lie in one 8-row group
				}
				int4 v = *(threadgroup const int4 *)(vs + k * HD + dg * 4);
				int4 vh = v >> 11, vl = v & 0x7ff;
				for (int i = 0; i < 4; i++) {
					int w = wt[(rg * 4 + i) * KT + k];
					sh[i] += w * vh;
					sl[i] += w * vl;
				}
			}
			for (int i = 0; i < 4; i++) {
				for (int e = 0; e < 4; e++) {
					od[i][e] += ((long)sh[i][e] << 11) + sl[i][e];
				}
			}
		}
	}
	// Z per row: sum the KT lanes that held it
	for (int i = 0; i < NR; i++) {
		int z = zl[i];
		for (ushort x = KT / 2; x > 0; x >>= 1) {
			z += simd_shuffle_xor(z, x);
		}
		if (key == 0) {
			zs[sg * RPS + lane / KT + 8 * RPS * i] = z;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (int i = 0; i < 4; i++) {
		int r = rg * 4 + i;
		long pi = ((long)(t0 + r) * a.nq + h) * a.nchmax + c;
		for (int e = 0; e < 4; e++) {
			op[pi * HD + dg * 4 + e] = od[i][e];
		}
	}
	if (tid < (uint)BRV) {
		zp[((long)(t0 + tid) * a.nq + h) * a.nchmax + c] = zs[tid];
	}
}

#define BPV(NAME, HD, KT, ETS)                                                                                             \
	extern "C" __global__ void NAME(const int *S, const long *cmax, const int *Vc, const int *expa, long *zp, long *op, const IntParams *pr, BAttnArgs a) { \
	uint3 tg3 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z); \
	uint tid = threadIdx.x;                                                          \
		__shared__ __align__(16) int vs[KT * HD];                                                                                       \
		__shared__ __align__(16) ushort wt[(256 / (HD / 4)) * 4 * KT];                                                                  \
		__shared__ __align__(16) long Ms[(256 / (HD / 4)) * 4];                                                                         \
		__shared__ __align__(16) long zs[(256 / (HD / 4)) * 4];                                                                         \
		__shared__ __align__(16) ushort et[ETS];                                                                                        \
		__shared__ __align__(16) int nz;                                                                                                \
		__shared__ __align__(16) uchar z8[(256 / (HD / 4)) / 2 * KT];                                                                   \
		bpv<HD, KT, ETS>(S, cmax, Vc, expa, zp, op, pr->pos, pr->tok, a, tg3, tid, vs, wt, Ms, zs, et, &nz, z8);          \
	}
// hd 64: 32-key tiles (half the barriers per key of 16), which leave room for 9,216 of EXPA's entries
BPV(int_bpv64, 64, 16, ETN)
BPV(int_bpv128, 128, 8, ETN)

// pass 3: a threadgroup per (query head, row), hd threads: sum the row's chunk partials, o = O / Z (e = -24).
extern "C" __global__ void int_bcomb(const long *zp, const long *op, long *o, const IntParams *pr, BAttnArgs a) {
	uint2 tg2 = make_uint2(blockIdx.x, blockIdx.y);
	uint d = threadIdx.x;
	int h = tg2.x, row = tg2.y, p = pr->pos + row, nch = p / KC + 1;
	long pi = ((long)row * a.nq + h) * a.nchmax;
	long Z = 0, O = 0;
	for (int c = 0; c < nch; c++) {
		if (cvis(c * KC, c * KC + KC, p + 1, a.win, a.sink)) {
			Z += zp[pi + c];
			O += op[(pi + c) * a.hd + d];
		}
	}
	long aO = iabs(O), hi = udivx(aO, Z);
	long r = (hi << 12) + udivx((aO - hi * Z) << 12, Z);
	o[((long)row * a.nq + h) * a.hd + d] = O < 0 ? -r : r;
}

// int_find: decoding's search, on the GPU so the table needn't be copied back: the first t with cum[t] > *tgt, and
// res = (t, cum[t], cum[t-1] (0 for t = 0), cum[V-1]); t = V if there is none.
extern "C" __global__ void int_find(const long *cum, const long *tgt, long *res, int V) {
	uint tid = threadIdx.x;
	if (tid != 0) {
		return;
	}
	long target = *tgt;
	int lo = 0, hi = V;
	while (lo < hi) {
		int mid = (lo + hi) / 2;
		if (cum[mid] > target) {
			hi = mid;
		} else {
			lo = mid + 1;
		}
	}
	res[0] = lo;
	res[1] = lo < V ? cum[lo] : 0;
	res[2] = lo > 0 ? cum[lo - 1] : 0;
	res[3] = cum[V - 1];
}

// int_mlpb: int_mlp and the blockf after it in one threadgroup: p, then mantissas m and exponent ex[2] = -16 + shift.
extern "C" __global__ void int_mlpb(const long *acc, const int *sig, long *p, short *m, int *ex, int I) {
	uint tid = threadIdx.x;
	__shared__ __align__(16) long tg[TG];
	const long NS = 16384;
	long mx = 0;
	for (int i = tid; i < I; i += TG) {
		long g = acc[i], u = acc[I + i]; // the matvec already shifted them to e = -16
		long av = rsh(g * (long)sig[clamp(rsh(g, 6), -NS, NS) + NS], 16);
		long v = rsh(av * u, 16);
		p[i] = v;
		mx = max(mx, iabs(v));
	}
	int sh = max(0, bitlen(tg_max(mx, tg, tid)) - (AQ - 1));
	for (int i = tid; i < I; i += TG) {
		m[i] = (short)clamp(rsh(p[i], sh), -LIMB(AQ), LIMB(AQ));
	}
	if (tid == 0) {
		ex[2] = -16 + sh;
	}
}
