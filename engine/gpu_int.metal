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

// h = rshift(q[tok] * s[tok], -24 - E), row y of the grid: token toks[y] into h[y] (a step binds the params,
// whose first int is the token)
kernel void int_embed(device const char *q [[buffer(0)]], device const short *s [[buffer(1)]],
                      device const int *toks [[buffer(2)]], device long *h [[buffer(3)]],
                      constant int &D [[buffer(4)]], constant int &E [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
	int k = gid.x;
	if (k >= D) {
		return;
	}
	long tok = toks[gid.y];
	h[(long)gid.y * D + k] = rsh((long)q[tok * D + k] * (long)s[tok * (D / 64) + k / 64], -24 - E);
}

// optionally h += acc (already at e = -24); then rmsnorm(h, g) -> mantissas m, exponent eout. One threadgroup of
// any size; each pass rereads h.
kernel void int_norm(device long *h [[buffer(0)]], device const long *acc [[buffer(1)]], device const long *g [[buffer(2)]],
                     device short *m [[buffer(3)]], device int *ex [[buffer(4)]], constant int &D [[buffer(5)]],
                     constant int &E [[buffer(6)]], constant int &einSlot [[buffer(7)]], constant int &eoutSlot [[buffer(8)]],
                     constant int &add [[buffer(9)]], constant int2 &st [[buffer(10)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
	threadgroup long tg[32];
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
                    constant int &accS [[buffer(13)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
	threadgroup long part[32][32];
	// row r of a batch: position pos + r
	acc += (long)row * accS;
	mq += (long)row * a.nq * a.hd;
	shq += row * a.nq;
	int pos = pr->pos + row, hh2 = a.hd / 2; // acc is already at e = -24 (the matvec shifted it)
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
                       constant int2 &st [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[TG];
	x += (long)row * st.x;
	m += (long)row * st.y;
	ex += 4 * row;
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
                    constant int &einSlot [[buffer(6)]], constant int2 &st [[buffer(7)]],
                    uint2 gid [[thread_position_in_grid]]) {
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

// ---- batches: many positions at once (compression knows every token in advance) ----

// int_gemm: out[t, o] = rshift(sum_g s[o,g] * sum_{k in g} q[o,k] x[t,k], to - (E + ex[4t + einSlot])) for the rows t
// of a batch; the same integers as int_matvec, row by row. A threadgroup computes GBO outputs x GBT rows; each
// 64-group's weights and activations go through threadgroup memory; a thread holds 4 outputs x 2 rows. Group sums are
// int32 (|.| < 64 * 128 * 32767 < 2^31); the scaling is split into 16-bit halves so each product fits 32 bits.
// st = (unused, x row stride, out row stride, unused).
#define GBO 64
#define GBT 32
kernel void int_gemm(device const char *q [[buffer(0)]], device const short *s [[buffer(1)]],
                     device const short *x [[buffer(2)]], device long *out [[buffer(3)]], device const int *ex [[buffer(4)]],
                     constant int &O [[buffer(5)]], constant int &K [[buffer(6)]], constant int &E [[buffer(7)]],
                     constant int &einSlot [[buffer(8)]], constant int &to [[buffer(9)]], constant int4 &st [[buffer(10)]],
                     uint2 tg2 [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup char W[GBO][68];  // padded rows
	threadgroup short X[GBT][72];
	threadgroup short S[GBO];
	int xS = st.y, outS = st.z;
	int o0 = tg2.x * GBO, t0 = tg2.y * GBT;
	int to4 = (tid % 16) * 4, tt2 = (tid / 16) * 2;
	long acc[4][2];
	for (int i = 0; i < 4; i++) {
		acc[i][0] = 0;
		acc[i][1] = 0;
	}
	int ng = K / 64;
	int wr = tid >> 2, wc = (tid & 3) * 16, xr = tid >> 3, xc = (tid & 7) * 8;
	device const char *wsrc = q + (long)min(o0 + wr, O - 1) * K + wc;
	device const short *xsrc = x + (long)(t0 + xr) * xS + xc;
	for (int g = 0; g < ng; g++) {
		{
			device const char4 *src = (device const char4 *)(wsrc + g * 64);
			threadgroup char4 *dst = (threadgroup char4 *)(&W[wr][wc]);
			dst[0] = src[0];
			dst[1] = src[1];
			dst[2] = src[2];
			dst[3] = src[3];
			device const short4 *xs = (device const short4 *)(xsrc + g * 64);
			threadgroup short4 *xd = (threadgroup short4 *)(&X[xr][xc]);
			xd[0] = xs[0];
			xd[1] = xs[1];
		}
		if (tid < GBO) {
			S[tid] = s[(long)min(o0 + (int)tid, O - 1) * ng + g];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		int gs[4][2];
		for (int i = 0; i < 4; i++) {
			gs[i][0] = 0;
			gs[i][1] = 0;
		}
		for (int k = 0; k < 64; k += 4) {
			int4 x0 = int4(*(threadgroup const short4 *)&X[tt2][k]), x1 = int4(*(threadgroup const short4 *)&X[tt2 + 1][k]);
			for (int i = 0; i < 4; i++) {
				int4 w = int4(*(threadgroup const char4 *)&W[to4 + i][k]);
				int4 a = w * x0, b = w * x1;
				gs[i][0] += a.x + a.y + a.z + a.w;
				gs[i][1] += b.x + b.y + b.z + b.w;
			}
		}
		for (int i = 0; i < 4; i++) {
			int sc = S[to4 + i];
			for (int j = 0; j < 2; j++) {
				int d = gs[i][j];
				acc[i][j] += ((long)((d >> 16) * sc) << 16) + (long)((d & 0xffff) * sc);
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	for (int i = 0; i < 4; i++) {
		int o = o0 + to4 + i;
		if (o < O) {
			for (int j = 0; j < 2; j++) {
				int t = t0 + tt2 + j;
				out[(long)t * outS + o] = rsh(acc[i][j], to - (E + ex[4 * t + einSlot]));
			}
		}
	}
}

// Batched attention: a threadgroup per (query head, block of BR rows); rows are consecutive positions pos + t.
// Two passes over the keys in tiles of KT (through threadgroup memory): the row maxima, then the weights, Z and O.
// Products in 32-bit pieces as in int_attn1/int_attn2 (k = kh * 2^11 + kl), folded into 64 bits every 16 terms.
// Rows past the batch's end compute garbage nobody reads; keys past a row's position are masked; the cache is padded
// so tiles may read past the last position.
template <int HD, int BR, int KT>
static inline void attnb(device const int *mq, device const int *shq, device const int *Kc, device const int *Vc,
                         device const int *expa, device long *o, int base, constant AttnArgs &a, uint h, uint blk,
                         uint tid, threadgroup int *qs, threadgroup int *ks, threadgroup int *vs, threadgroup int *wt) {
	constexpr int TPR = 256 / BR, KPT = KT / TPR, DPT = HD / TPR, KS = HD + 4; // ks rows padded (banks)
	int grp = a.nq / a.nkv, j = h / grp;
	int t0 = blk * BR, r = tid / TPR, sub = tid % TPR;
	int row = t0 + r, p = base + row, pmax = base + t0 + BR - 1;
	for (int i = tid; i < BR * HD; i += 256) {
		qs[i] = mq[((long)(t0 + i / HD) * a.nq + h) * HD + i % HD];
	}
	int sh = shq[row * a.nq + h];
	device const int *K = Kc + (long)j * a.maxlen * HD;
	device const int *V = Vc + (long)j * a.maxlen * HD;
	threadgroup const int *qr = qs + r * HD;
	long M = LMIN;
	for (int k0 = 0; k0 <= pmax; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int i = tid; i < KT * HD; i += 256) {
			ks[(i / HD) * KS + i % HD] = K[(long)k0 * HD + i];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int c = 0; c < KPT; c++) {
			int kk = c * TPR + sub;
			threadgroup const int *kr = ks + kk * KS;
			long acc = 0;
			for (int d0 = 0; d0 < HD; d0 += 16) {
				int ah = 0, al = 0;
				for (int d = d0; d < d0 + 16; d += 4) {
					int4 kv = *(threadgroup const int4 *)(kr + d), qv = *(threadgroup const int4 *)(qr + d);
					int4 ph = qv * (kv >> 11), pl = qv * (kv & 0x7ff);
					ah += ph.x + ph.y + ph.z + ph.w;
					al += pl.x + pl.y + pl.z + pl.w;
				}
				acc += ((long)ah << 11) + al;
			}
			if (k0 + kk <= p) {
				M = max(M, rsh(acc, 42 - sh));
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
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int c = 0; c < KPT; c++) {
			int kk = c * TPR + sub;
			threadgroup const int *kr = ks + kk * KS;
			long acc = 0;
			for (int d0 = 0; d0 < HD; d0 += 16) {
				int ah = 0, al = 0;
				for (int d = d0; d < d0 + 16; d += 4) {
					int4 kv = *(threadgroup const int4 *)(kr + d), qv = *(threadgroup const int4 *)(qr + d);
					int4 ph = qv * (kv >> 11), pl = qv * (kv & 0x7ff);
					ah += ph.x + ph.y + ph.z + ph.w;
					al += pl.x + pl.y + pl.z + pl.w;
				}
				acc += ((long)ah << 11) + al;
			}
			int w = 0;
			if (k0 + kk <= p) {
				w = expa[min(M - rsh(acc, 42 - sh), (long)(a.na - 1))];
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
	kernel void NAME(device const int *mq [[buffer(0)]], device const int *shq [[buffer(1)]],                              \
	                 device const int *Kc [[buffer(2)]], device const int *Vc [[buffer(3)]],                               \
	                 device const int *expa [[buffer(4)]], device long *o [[buffer(5)]],                                   \
	                 device const IntParams *pr [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],                        \
	                 uint2 tg2 [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {              \
		threadgroup int qs[BR * HD];                                                                                       \
		threadgroup int ks[KT * (HD + 4)];                                                                                 \
		threadgroup int vs[KT * HD];                                                                                       \
		threadgroup int wt[BR * KT];                                                                                       \
		attnb<HD, BR, KT>(mq, shq, Kc, Vc, expa, o, pr->pos, a, tg2.x, tg2.y, tid, qs, ks, vs, wt);                       \
	}
ATTNB(int_attnb64, 64, 32, 32)
ATTNB(int_attnb128, 128, 16, 16)

// int_counts: for row t of a batch, the frequency table's total and the target token's cumulative counts:
// cnt[t] = (cum[V-1], cum[tgt[t]], cum[tgt[t]] - freq[tgt[t]]), cum as int_table1..4 build it. One threadgroup a row.
kernel void int_counts(device const long *lg [[buffer(0)]], device const int *expf_ [[buffer(1)]],
                       device const int *tgt [[buffer(2)]], device long *cnt [[buffer(3)]], constant int &V [[buffer(4)]],
                       constant int &nf [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
	threadgroup long tg[32];
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
	int nq, nkv, hd, maxlen, na, smax, nchmax, ez; // ez: EXPA[i] = 0 for i >= ez
};

// pass 1: a threadgroup per (kv head, block of QR / grp rows, key chunk): the scores of every query head sharing the
// kv head (QR query rows: grp heads x rows) against the chunk's keys, in tiles of KT through threadgroup memory; a
// thread holds QR/32 query rows x KT/8 keys. S[(row * nq + h) * smax + t] = score; cmax = the chunk's maxima.
template <int HD, int QR, int KT>
static inline void bscores(device const int *mq, device const int *shq, device const int *Kc, device int *S,
                           device long *cmax, device atomic_int *ovf, int base, int R, constant BAttnArgs &a, uint3 tg3, uint tid,
                           threadgroup short *qs, threadgroup int *ks, threadgroup int *shs) {
	constexpr int TR = QR / 32, TK = KT / 8, QS = HD + 4, KS = HD + 4;
	int j = tg3.x, grp = a.nq / a.nkv, br = QR / grp; // rows per block
	int t0 = tg3.y * br, c = tg3.z;
	int pmax = base + min(t0 + br, R) - 1; // rows t0 + r < R only
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int qg = tid / 8, sub = tid % 8;
	for (int i = tid; i < QR * HD; i += 256) {
		int qr = i / HD, d = i % HD, g = qr / br, r = qr % br;
		qs[qr * QS + d] = g < grp && t0 + r < R ? (short)mq[((long)(t0 + r) * a.nq + j * grp + g) * HD + d] : (short)0;
	}
	if (tid < QR) {
		int g = tid / br, r = tid % br;
		shs[tid] = g < grp && t0 + r < R ? shq[(t0 + r) * a.nq + j * grp + g] : 0;
	}
	device const int *K = Kc + (long)j * a.maxlen * HD;
	long mx[TR];
	for (int i = 0; i < TR; i++) {
		mx[i] = LMIN;
	}
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (int i = tid; i < KT * HD; i += 256) {
			ks[(i / HD) * KS + i % HD] = K[(long)k0 * HD + i];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		long acc[TR][TK];
		for (int i = 0; i < TR; i++) {
			for (int k = 0; k < TK; k++) {
				acc[i][k] = 0;
			}
		}
		for (int d0 = 0; d0 < HD; d0 += 16) {
			int ah[TR][TK], al[TR][TK];
			for (int i = 0; i < TR; i++) {
				for (int k = 0; k < TK; k++) {
					ah[i][k] = 0;
					al[i][k] = 0;
				}
			}
			for (int d = d0; d < d0 + 16; d += 4) {
				int4 kh[TK], kl[TK];
				for (int k = 0; k < TK; k++) {
					int4 kv = *(threadgroup const int4 *)(ks + (sub + 8 * k) * KS + d);
					kh[k] = kv >> 11;
					kl[k] = kv & 0x7ff;
				}
				for (int i = 0; i < TR; i++) {
					int4 q = int4(*(threadgroup const short4 *)(qs + (qg + 32 * i) * QS + d));
					for (int k = 0; k < TK; k++) {
						int4 ph = q * kh[k], pl = q * kl[k];
						ah[i][k] += ph.x + ph.y + ph.z + ph.w;
						al[i][k] += pl.x + pl.y + pl.z + pl.w;
					}
				}
			}
			for (int i = 0; i < TR; i++) {
				for (int k = 0; k < TK; k++) {
					acc[i][k] += ((long)ah[i][k] << 11) + al[i][k];
				}
			}
		}
		for (int i = 0; i < TR; i++) {
			int qr = qg + 32 * i, g = qr / br, r = qr % br;
			if (g >= grp || t0 + r >= R) {
				continue;
			}
			int p = base + t0 + r;
			device int *Sr = S + ((long)(t0 + r) * a.nq + j * grp + g) * a.smax;
			for (int k = 0; k < TK; k++) {
				int t = k0 + sub + 8 * k;
				if (t <= p) {
					long s = rsh(acc[i][k], 42 - shs[qr]);
					if (s != (long)(int)s) {
						atomic_store_explicit(ovf, 1, memory_order_relaxed);
					}
					Sr[t] = (int)s;
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
	kernel void NAME(device const int *mq [[buffer(0)]], device const int *shq [[buffer(1)]],                              \
	                 device const int *Kc [[buffer(2)]], device int *S [[buffer(3)]], device long *cmax [[buffer(4)]],    \
	                 device atomic_int *ovf [[buffer(5)]], device const IntParams *pr [[buffer(6)]],                     \
	                 constant BAttnArgs &a [[buffer(7)]], uint3 tg3 [[threadgroup_position_in_grid]],                    \
	                 uint tid [[thread_index_in_threadgroup]]) {                                                          \
		threadgroup short qs[QR * (HD + 4)];                                                                               \
		threadgroup int ks[KT * (HD + 4)];                                                                                 \
		threadgroup int shs[QR];                                                                                           \
		bscores<HD, QR, KT>(mq, shq, Kc, S, cmax, ovf, pr->pos, pr->tok, a, tg3, tid, qs, ks, shs);                                 \
	}
BSCORES(int_bscores64, 64, 128, 16)
BSCORES(int_bscores128, 128, 64, 16)

// pass 2: a threadgroup per (query head, block of BRV rows, key chunk): weights w = EXPA[min(M - s, na - 1)] (M from
// the chunk maxima), the chunk's Z and O = sum w V; a thread holds 4 rows x 4 dimensions. Partials to zp, op.
// EXPA's nonzero head (a.ez entries, <= ETN; the rest are 0) is copied into threadgroup memory as ushort: the weight
// lookups are random, and from device memory they cost more than the arithmetic. Tiles whose weights are all 0 add
// nothing and are skipped.
#define ETN 11392
template <int HD, int KT>
static inline void bpv(device const int *S, device const long *cmax, device const int *Vc, device const int *expa,
                       device long *zp, device long *op, int base, constant BAttnArgs &a, uint3 tg3, uint tid,
                       threadgroup int *vs, threadgroup ushort *wt, threadgroup long *Ms, threadgroup long *zs,
                       threadgroup ushort *et, threadgroup int *nz, threadgroup uchar *z8) {
	constexpr int DG = HD / 4, RG = 256 / DG, BRV = RG * 4, FK = KT < 16 ? KT : 16, RPS = 32 / KT;
	int h = tg3.x, t0 = tg3.y * BRV, c = tg3.z, grp = a.nq / a.nkv, j = h / grp;
	int pmax = base + t0 + BRV - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int ez = a.ez;
	for (int i = tid; i < ez; i += 256) {
		et[i] = (ushort)expa[i];
	}
	if (tid < (uint)BRV) {
		int p = base + t0 + tid, nch = p / KC + 1;
		device const long *cm = cmax + ((long)(t0 + tid) * a.nq + h) * a.nchmax;
		long m = LMIN;
		for (int i = 0; i < nch; i++) {
			m = max(m, cm[i]);
		}
		Ms[tid] = m;
		zs[tid] = 0;
	}
	device const int *V = Vc + (long)j * a.maxlen * HD;
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
			if (t <= p) {
				long d = Ms[r] - S[((long)(t0 + r) * a.nq + h) * a.smax + t];
				if (d < ez) {
					w = et[d];
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

#define BPV(NAME, HD, KT)                                                                                                  \
	kernel void NAME(device const int *S [[buffer(0)]], device const long *cmax [[buffer(1)]],                            \
	                 device const int *Vc [[buffer(2)]], device const int *expa [[buffer(3)]], device long *zp [[buffer(4)]], \
	                 device long *op [[buffer(5)]], device const IntParams *pr [[buffer(6)]],                            \
	                 constant BAttnArgs &a [[buffer(7)]], uint3 tg3 [[threadgroup_position_in_grid]],                    \
	                 uint tid [[thread_index_in_threadgroup]]) {                                                          \
		threadgroup int vs[KT * HD];                                                                                       \
		threadgroup ushort wt[(256 / (HD / 4)) * 4 * KT];                                                                  \
		threadgroup long Ms[(256 / (HD / 4)) * 4];                                                                         \
		threadgroup long zs[(256 / (HD / 4)) * 4];                                                                         \
		threadgroup ushort et[ETN];                                                                                        \
		threadgroup int nz;                                                                                                \
		threadgroup uchar z8[(256 / (HD / 4)) / 2 * KT];                                                                   \
		bpv<HD, KT>(S, cmax, Vc, expa, zp, op, pr->pos, a, tg3, tid, vs, wt, Ms, zs, et, &nz, z8);                         \
	}
BPV(int_bpv64, 64, 16)
BPV(int_bpv128, 128, 8)

// pass 3: a threadgroup per (query head, row), hd threads: sum the row's chunk partials, o = O / Z (e = -24).
kernel void int_bcomb(device const long *zp [[buffer(0)]], device const long *op [[buffer(1)]], device long *o [[buffer(2)]],
                      device const IntParams *pr [[buffer(3)]], constant BAttnArgs &a [[buffer(4)]],
                      uint2 tg2 [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
	int h = tg2.x, row = tg2.y, p = pr->pos + row, nch = p / KC + 1;
	long pi = ((long)row * a.nq + h) * a.nchmax;
	long Z = 0, O = 0;
	for (int c = 0; c < nch; c++) {
		Z += zp[pi + c];
		O += op[(pi + c) * a.hd + d];
	}
	long aO = iabs(O), hi = udivx(aO, Z);
	long r = (hi << 12) + udivx((aO - hi * Z) << 12, Z);
	o[((long)row * a.nq + h) * a.hd + d] = O < 0 ? -r : r;
}
