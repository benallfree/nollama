// Integer models (intmodel.go) on Metal: the same integer arithmetic as the CPU, so the same results, bit for bit.
// Every value is an integer; 64-bit where the CPU uses int64. Reductions are integer sums and maxima, which do not
// depend on order.
#include <metal_stdlib>
using namespace metal;

#define TG 1024
#define LMIN (-9223372036854775807L - 1)

static inline long rsh(long x, int n) { return n <= 0 ? (x << (-n)) : ((x + (1L << (n - 1))) >> n); }

static inline int bitlen(long v) { return v == 0 ? 0 : 64 - (int)clz((ulong)v); }

// long <-> float through native 32-bit conversions (the 64-bit ones are emulated). Approximations, used only for
// estimates that integer arithmetic then corrects.
static inline float l2f(long a) { return (float)(int)(a >> 32) * 4294967296.0f + (float)(uint)a; }
static inline long f2l(float f) {
	if (fabs(f) < 2147483648.0f) {
		return (long)(int)f;
	}
	float hi = floor(f * (1.0f / 4294967296.0f));
	float lo = clamp(f - hi * 4294967296.0f, 0.0f, 4294967295.0f);
	return ((long)(int)hi << 32) + (long)(uint)lo;
}

// a / b for a >= 0, b > 0, exactly: a float estimate, refined once, then corrected in integers. Apple GPUs emulate
// 64-bit division, which costs hundreds of cycles; this is a few float ops and multiplies.
static inline long udivx(long a, long b) {
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
static inline long sdivx(long a, long b) { return a < 0 ? -udivx(-a, b) : udivx(a, b); }

static inline long iabs(long v) { return v < 0 ? -v : v; }

// reductions within a simdgroup (32 lanes on Apple GPUs), by integer shuffles; MSL shuffles no 64-bit types, so a
// long travels as its two 32-bit halves.
static inline long shx(long v, ushort o) { return as_type<long>(simd_shuffle_xor(as_type<uint2>(v), o)); }
static inline long sg_sum(long v) {
	for (ushort o = 16; o > 0; o >>= 1) {
		v += shx(v, o);
	}
	return v;
}
static inline long sg_max(long v) {
	for (ushort o = 16; o > 0; o >>= 1) {
		v = max(v, shx(v, o));
	}
	return v;
}

// max over a TG-thread threadgroup (tg: at least TG/32 longs of threadgroup memory): simdgroups first, then their
// TG/32 = 32 partials, which every simdgroup reduces for itself.
static inline long tg_max(long v, threadgroup long *tg, uint tid) {
	v = sg_max(v);
	if ((tid & 31) == 0) {
		tg[tid >> 5] = v;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long r = sg_max(tg[tid & 31]);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}

static inline long isqrt_exact(long v) {
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
static inline long tgn_max(long v, threadgroup long *tg, uint tid, uint nt) {
	v = sg_max(v);
	if ((tid & 31) == 0) {
		tg[tid >> 5] = v;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long r = sg_max((tid & 31) < nt / 32 ? tg[tid & 31] : LMIN);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}
static inline long tgn_sum(long v, threadgroup long *tg, uint tid, uint nt) {
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

// h = rshift(q[tok] * s[tok], -24 - E)
kernel void int_embed(device const char *q [[buffer(0)]], device const short *s [[buffer(1)]],
                      device const IntParams *pr [[buffer(2)]], device long *h [[buffer(3)]],
                      constant int &D [[buffer(4)]], constant int &E [[buffer(5)]], uint k [[thread_position_in_grid]]) {
	if ((int)k >= D) {
		return;
	}
	long tok = pr->tok;
	h[k] = rsh((long)q[tok * D + k] * (long)s[tok * (D / 64) + k / 64], -24 - E);
}

// optionally h += acc (already at e = -24); then rmsnorm(h, g) -> mantissas m, exponent eout. One threadgroup of
// any size; each pass rereads h.
kernel void int_norm(device long *h [[buffer(0)]], device const long *acc [[buffer(1)]], device const long *g [[buffer(2)]],
                     device short *m [[buffer(3)]], device int *ex [[buffer(4)]], constant int &D [[buffer(5)]],
                     constant int &E [[buffer(6)]], constant int &einSlot [[buffer(7)]], constant int &eoutSlot [[buffer(8)]],
                     constant int &add [[buffer(9)]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
	threadgroup long tg[32];
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
	int ym = 0;
	for (int i = tid; i < D; i += nt) {
		ym = max(ym, bitlen(iabs(sdivx(rsh(h[i], sh) * g[i] * 16384, r))));
	}
	int sh2 = max(0, (int)tgn_max(ym, tg, tid, nt) - 15);
	for (int i = tid; i < D; i += nt) {
		m[i] = (short)clamp(rsh(sdivx(rsh(h[i], sh) * g[i] * 16384, r), sh2), -32767L, 32767L);
	}
	if (tid == 0) {
		ex[eoutSlot] = sh2 - 30;
	}
}

// out[row] = rshift(sum_g s[row,g] * sum_k q[row,k] x[k], to - (E + ein)): one row per simdgroup (many simdgroups in flight hide memory latency). Each step covers 512 columns:
// lane l reads bytes 16l..16l+15 and the matching activations; a 64-group is 4 lanes, summed by shuffles in int32
// (|sum| < 64 * 127 * 32767 < 2^31), scaled in 64 bits; the lanes' totals are summed at the end.
kernel void int_matvec(device const char *q [[buffer(0)]], device const short *s [[buffer(1)]],
                       device const short *x [[buffer(2)]], device long *out [[buffer(3)]], device const int *ex [[buffer(4)]],
                       constant int &rows [[buffer(5)]], constant int &K [[buffer(6)]], constant int &E [[buffer(7)]],
                       constant int &einSlot [[buffer(8)]], constant int &to [[buffer(9)]],
                       uint tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
	int row = tg_id * 8 + sg;
	if (row >= rows) {
		return;
	}
	int ng = K / 64;
	device const char *qr = q + (long)row * K;
	device const short *sr = s + (long)row * ng;
	long acc = 0;
	for (int col = 16 * lane; col < K + 16 * 31; col += 512) {
		bool in = col < K;
		int d = 0;
		if (in) {
			device const short4 *xp = (device const short4 *)(x + col);
			device const char4 *wp = (device const char4 *)(qr + col);
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

// x = rshift(acc, -24 - (E + ein)) + b; rope q and k at pos; K, V = clip(rshift(., 12)) into the cache; q heads:
// blockf(q * sc) -> mq, shq. One threadgroup; simdgroup h handles q head h.
kernel void int_qkv(device const long *acc [[buffer(0)]], device const long *b [[buffer(1)]],
                    device const short *rc [[buffer(2)]], device const short *rs [[buffer(3)]],
                    device const IntParams *pr [[buffer(4)]], device int *Kc [[buffer(5)]], device int *Vc [[buffer(6)]],
                    device int *mq [[buffer(7)]], device int *shq [[buffer(8)]], device const int *ex [[buffer(9)]],
                    constant QKVArgs &a [[buffer(10)]], constant int &E [[buffer(11)]], constant int &einSlot [[buffer(12)]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
	threadgroup long part[32][32];
	int pos = pr->pos, hh2 = a.hd / 2; // acc is already at e = -24 (the matvec shifted it)
	device const short *c = rc + pos * hh2;
	device const short *sn = rs + pos * hh2;
	const long LIM = (1L << 22) - 1;
	// k and v heads (threads 0 .. nkv*hh2-1 do k pairs; nkv*hd threads do v)
	for (int i = tid; i < a.nkv * hh2; i += TG) {
		int hh = i / hh2, j = i % hh2;
		int base = (a.nq + hh) * a.hd;
		long x1 = acc[base + j] + b[base + j], x2 = acc[base + j + hh2] + b[base + j + hh2];
		long cj = c[j], sj = sn[j];
		long k1 = rsh(x1 * cj - x2 * sj, 14), k2 = rsh(x2 * cj + x1 * sj, 14);
		long kb = ((long)hh * a.maxlen + pos) * a.hd;
		Kc[kb + j] = (int)clamp(rsh(k1, 12), -LIM, LIM);
		Kc[kb + j + hh2] = (int)clamp(rsh(k2, 12), -LIM, LIM);
	}
	for (int i = tid; i < a.nkv * a.hd; i += TG) {
		int hh = i / a.hd, d = i % a.hd;
		int idx = (a.nq + a.nkv) * a.hd + i;
		long v = acc[idx] + b[idx];
		Vc[((long)hh * a.maxlen + pos) * a.hd + d] = (int)clamp(rsh(v, 12), -LIM, LIM);
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
		int sh = max(0, bitlen(hm) - 15);
		for (int t = 0; t < 4; t++) {
			int j = lane + t * 32;
			if (j < hh2) {
				mq[hq * a.hd + j] = (int)clamp(rsh(qv[t][0], sh), -32767L, 32767L);
				mq[hq * a.hd + j + hh2] = (int)clamp(rsh(qv[t][1], sh), -32767L, 32767L);
			}
		}
		if (lane == 0) {
			shq[hq] = sh;
		}
	}
}

// blockf over n values -> mantissas m, exponent ex[eoutSlot] = ein + shift. One threadgroup.
kernel void int_blockf(device const long *x [[buffer(0)]], device short *m [[buffer(1)]], device int *ex [[buffer(2)]],
                       constant int &n [[buffer(3)]], constant int &ein [[buffer(4)]], constant int &eoutSlot [[buffer(5)]],
                       uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[TG];
	long mx = 0;
	for (int i = tid; i < n; i += TG) {
		mx = max(mx, iabs(x[i]));
	}
	int sh = max(0, bitlen(tg_max(mx, tg, tid)) - 15);
	for (int i = tid; i < n; i += TG) {
		m[i] = (short)clamp(rsh(x[i], sh), -32767L, 32767L);
	}
	if (tid == 0) {
		ex[eoutSlot] = ein + sh;
	}
}

// p = rshift(a * u, 16), a = rshift(g * SIG[clip(rshift(g, 6), -ns, ns) + ns], 16), with g, u = rshift(acc, -16 - (E+ein))
kernel void int_mlp(device const long *acc [[buffer(0)]], device const int *sig [[buffer(1)]], device long *p [[buffer(2)]],
                    device const int *ex [[buffer(3)]], constant int &I [[buffer(4)]], constant int &E [[buffer(5)]],
                    constant int &einSlot [[buffer(6)]], uint i [[thread_position_in_grid]]) {
	if ((int)i >= I) {
		return;
	}
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
kernel void int_table1(device const long *lg [[buffer(0)]], device long *pm [[buffer(1)]], constant int &V [[buffer(2)]],
                       uint tgi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[32];
	long mx = LMIN;
	for (int i = tgi * TCH + tid; i < min((int)(tgi + 1) * TCH, V); i += 256) {
		mx = max(mx, lg[i]);
	}
	mx = tgn_max(mx, tg, tid, 256);
	if (tid == 0) {
		pm[tgi] = mx;
	}
}

static inline long table_max(device const long *pm, int nch, threadgroup long *tg, uint tid) {
	long mx = LMIN;
	for (int c = tid; c < nch; c += 256) {
		mx = max(mx, pm[c]);
	}
	return tgn_max(mx, tg, tid, 256);
}

kernel void int_table2(device const long *lg [[buffer(0)]], device const int *expf_ [[buffer(1)]], device long *pm [[buffer(2)]],
                       device long *pz [[buffer(3)]], device long *cum [[buffer(4)]], constant int &V [[buffer(5)]],
                       constant int &nf [[buffer(6)]], uint tgi [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint nch [[threadgroups_per_grid]]) {
	threadgroup long tg[32];
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

kernel void int_table3(device const long *pz [[buffer(0)]], device long *pt [[buffer(1)]], device long *cum [[buffer(2)]],
                       constant int &V [[buffer(3)]], uint tgi [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint nch [[threadgroups_per_grid]]) {
	threadgroup long tg[32];
	threadgroup long sc[256];
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

kernel void int_table4(device const long *pt [[buffer(0)]], device long *cum [[buffer(1)]], constant int &V [[buffer(2)]],
                       uint tgi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[32];
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
	int nq, nkv, hd, maxlen, na, maxch;
};

// Split attention by KV head: a threadgroup per (kv head, chunk) serves the grp = nq / nkv query heads that share it,
// so each K and V row is read once instead of grp times.
#define GMAX 8   // query heads per KV head, at most
#define HDMAX 128
kernel void int_attn1(device const int *mq [[buffer(0)]], device const int *shq [[buffer(1)]],
                      device const int *Kc [[buffer(2)]], device long *S [[buffer(3)]], device long *cmax [[buffer(4)]],
                      device const IntParams *pr [[buffer(5)]], constant AttnArgs &a [[buffer(6)]],
                      uint2 tg2 [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup int qs[GMAX * HDMAX];
	threadgroup long tg[32];
	int j = tg2.x, c = tg2.y, T = pr->pos + 1, hd = a.hd, grp = a.nq / a.nkv;
	for (int i = tid; i < grp * hd; i += ACH) {
		qs[i] = mq[j * grp * hd + i];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	int t = c * ACH + tid;
	long acc[GMAX];
	for (int g = 0; g < GMAX; g++) {
		acc[g] = 0;
	}
	// q . k in 32-bit pieces: k = kh * 2^11 + kl, kl in [0, 2^11); |q| < 2^15, |k| < 2^22, so each product is
	// below 2^26 and 16 of them fit an int; fold into 64 bits every 16 dimensions.
	if (t < T) {
		device const int *k = Kc + ((long)j * a.maxlen + t) * hd;
		for (int d0 = 0; d0 < hd; d0 += 16) {
			int ah[GMAX], al[GMAX];
			for (int g = 0; g < GMAX; g++) {
				ah[g] = 0;
				al[g] = 0;
			}
			for (int d = d0; d < d0 + 16; d += 4) {
				int4 kk = *(device const int4 *)(k + d);
				int4 kh = kk >> 11, kl = kk & 0x7ff;
				for (int g = 0; g < GMAX; g++) {
					if (g < grp) {
						int4 q = *(threadgroup const int4 *)(qs + g * hd + d);
						int4 ph = q * kh, pl = q * kl;
						ah[g] += ph.x + ph.y + ph.z + ph.w;
						al[g] += pl.x + pl.y + pl.z + pl.w;
					}
				}
			}
			for (int g = 0; g < GMAX; g++) {
				acc[g] += ((long)ah[g] << 11) + al[g];
			}
		}
	}
	threadgroup long part[GMAX][ACH / 32];
	for (int g = 0; g < grp; g++) {
		int h = j * grp + g;
		long v = LMIN;
		if (t < T) {
			v = rsh(acc[g], 42 - shq[h]);
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

kernel void int_attn2(device const long *S [[buffer(0)]], device const long *cmax [[buffer(1)]],
                      device const int *Vc [[buffer(2)]], device const int *expa [[buffer(3)]],
                      device long *zpart [[buffer(4)]], device long *opart [[buffer(5)]],
                      device const IntParams *pr [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],
                      uint2 tg2 [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup int w[GMAX][ACH]; // weights, <= 2^15
	threadgroup long red[GMAX][ACH]; // O partials: (ACH / hd - 1) slices x hd lanes per head
	int j = tg2.x, c = tg2.y, T = pr->pos + 1, hd = a.hd, grp = a.nq / a.nkv;
	int nch = (T + ACH - 1) / ACH;
	int t = c * ACH + tid;
	// M per head: simdgroup g reduces head g's chunk maxima (ACH / 32 = 8 simdgroups >= grp)
	threadgroup long Ms[GMAX];
	threadgroup long part[GMAX][ACH / 32];
	uint sgi = tid >> 5, lane = tid & 31;
	if (sgi < (uint)grp) {
		long mx = LMIN;
		for (int i = lane; i < nch; i += 32) {
			mx = max(mx, cmax[(j * grp + sgi) * a.maxch + i]);
		}
		mx = sg_max(mx);
		if (lane == 0) {
			Ms[sgi] = mx;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (int g = 0; g < grp; g++) {
		long wt = t < T ? (long)expa[min(Ms[g] - S[(long)(j * grp + g) * a.maxlen + t], (long)(a.na - 1))] : 0;
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
	device const int *V = Vc + ((long)j * a.maxlen + c * ACH) * hd + d;
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
kernel void int_attn3(device const long *zpart [[buffer(0)]], device const long *opart [[buffer(1)]],
                       device long *o [[buffer(2)]], device const IntParams *pr [[buffer(3)]],
                       constant AttnArgs &a [[buffer(4)]], uint h [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
	int T = pr->pos + 1, nch = (T + ACH - 1) / ACH;
	long Z = 0, O = 0;
	for (int c = 0; c < nch; c++) {
		Z += zpart[h * a.maxch + c];
		O += opart[((long)h * a.maxch + c) * a.hd + tid];
	}
	long aO = iabs(O), hi = udivx(aO, Z);
	long r = (hi << 12) + udivx((aO - hi * Z) << 12, Z);
	o[h * a.hd + tid] = O < 0 ? -r : r;
}
