// Integer models (intmodel.go) on Metal: the same integer arithmetic as the CPU, so the same results, bit for bit.
// Every value is an integer; 64-bit where the CPU uses int64. Reductions are integer sums and maxima, which do not
// depend on order.
#include <metal_stdlib>
using namespace metal;

#define TG 1024
#define LMIN (-9223372036854775807L - 1)

static inline long rsh(long x, int n) { return n <= 0 ? (x << (-n)) : ((x + (1L << (n - 1))) >> n); }

static inline int bitlen(long v) {
	ulong u = (ulong)v;
	int n = 0;
	while (u != 0) {
		u >>= 1;
		n++;
	}
	return n;
}

static inline long iabs(long v) { return v < 0 ? -v : v; }

// reduce over a TG-thread threadgroup (tg: TG longs of threadgroup memory)
static inline long tg_max(long v, threadgroup long *tg, uint tid) {
	tg[tid] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint s = TG / 2; s > 0; s >>= 1) {
		if (tid < s) {
			tg[tid] = max(tg[tid], tg[tid + s]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	long r = tg[0];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}

static inline long tg_sum(long v, threadgroup long *tg, uint tid) {
	tg[tid] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint s = TG / 2; s > 0; s >>= 1) {
		if (tid < s) {
			tg[tid] += tg[tid + s];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	long r = tg[0];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return r;
}

static inline long isqrt_exact(long v) {
	long r = (long)sqrt((float)v);
	while (r > 0 && r * r > v) {
		r--;
	}
	while ((r + 1) * (r + 1) <= v) {
		r++;
	}
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

// optionally h += rshift(acc, -24 - (E + ein)); then rmsnorm(h, g) -> mantissas m, exponent eout. One threadgroup.
kernel void int_norm(device long *h [[buffer(0)]], device const long *acc [[buffer(1)]], device const long *g [[buffer(2)]],
                     device int *m [[buffer(3)]], device int *ex [[buffer(4)]], constant int &D [[buffer(5)]],
                     constant int &E [[buffer(6)]], constant int &einSlot [[buffer(7)]], constant int &eoutSlot [[buffer(8)]],
                     constant int &add [[buffer(9)]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[TG];
	const int PER = 4; // D <= 4096
	long hv[PER];
	long mx = 0;
	for (int j = 0; j < PER; j++) {
		int i = tid + j * TG;
		long v = 0;
		if (i < D) {
			v = h[i];
			if (add) {
				v += acc[i]; // the matvec already shifted it to e = -24
				h[i] = v;
			}
		}
		hv[j] = v;
		mx = max(mx, iabs(v));
	}
	int sh = max(0, bitlen(tg_max(mx, tg, tid)) - 20);
	long ss = 0;
	for (int j = 0; j < PER; j++) {
		long x = rsh(hv[j], sh);
		hv[j] = x;
		ss += x * x;
	}
	long ms = tg_sum(ss, tg, tid) / D + (1L << max(0, 48 - 2 * sh)) / 1000000;
	long r = max(1L, isqrt_exact(ms));
	long ym = 0;
	for (int j = 0; j < PER; j++) {
		int i = tid + j * TG;
		long y = 0;
		if (i < D) {
			y = hv[j] * g[i] * 16384 / r;
		}
		hv[j] = y;
		ym = max(ym, iabs(y));
	}
	int sh2 = max(0, bitlen(tg_max(ym, tg, tid)) - 15);
	for (int j = 0; j < PER; j++) {
		int i = tid + j * TG;
		if (i < D) {
			m[i] = (int)clamp(rsh(hv[j], sh2), -32767L, 32767L);
		}
	}
	if (tid == 0) {
		ex[eoutSlot] = sh2 - 30;
	}
}

// out[row] = rshift(sum_g s[row,g] * sum_k q[row,k] x[k], to - (E + ein)). A simdgroup per row, 8 rows a threadgroup.
kernel void int_matvec(device const char *q [[buffer(0)]], device const short *s [[buffer(1)]],
                       device const int *x [[buffer(2)]], device long *out [[buffer(3)]], device const int *ex [[buffer(4)]],
                       constant int &rows [[buffer(5)]], constant int &K [[buffer(6)]], constant int &E [[buffer(7)]],
                       constant int &einSlot [[buffer(8)]], constant int &to [[buffer(9)]],
                       uint tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
	threadgroup long part[8][32];
	int row = tg_id * 8 + sg;
	long acc = 0;
	if (row < rows) {
		int ng = K / 64;
		device const char *qr = q + (long)row * K;
		device const short *sr = s + (long)row * ng;
		for (int gi = lane; gi < ng; gi += 32) {
			device const char4 *q4 = (device const char4 *)(qr + gi * 64);
			device const int4 *x4 = (device const int4 *)(x + gi * 64);
			int d = 0;
			for (int k = 0; k < 16; k++) {
				int4 a = int4(q4[k]);
				int4 b = x4[k];
				d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
			}
			acc += (long)d * (long)sr[gi];
		}
	}
	part[sg][lane] = acc;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (row < rows && lane == 0) {
		long t = 0;
		for (int i = 0; i < 32; i++) {
			t += part[sg][i];
		}
		out[row] = rsh(t, to - (E + ex[einSlot]));
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

struct AttnArgs {
	int nq, nkv, hd, maxlen, na;
};

// one threadgroup per q head: scores, max, weights EXPA[min(M - s, na - 1)], Z, O = sum w V, o = O / Z (e = -24).
kernel void int_attn(device const int *mq [[buffer(0)]], device const int *shq [[buffer(1)]],
                     device const int *Kc [[buffer(2)]], device const int *Vc [[buffer(3)]],
                     device const int *expa [[buffer(4)]], device long *S [[buffer(5)]], device long *o [[buffer(6)]],
                     device const IntParams *pr [[buffer(7)]], constant AttnArgs &a [[buffer(8)]],
                     uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[TG];
	int T = pr->pos + 1, hd = a.hd;
	int j = h / (a.nq / a.nkv);
	int n = 42 - shq[h];
	device const int *q = mq + h * hd;
	device const int *K = Kc + (long)j * a.maxlen * hd;
	device const int *V = Vc + (long)j * a.maxlen * hd;
	device long *s = S + (long)h * a.maxlen;
	long mx = LMIN;
	for (int t = tid; t < T; t += TG) {
		long acc = 0;
		for (int d = 0; d < hd; d++) {
			acc += (long)q[d] * (long)K[(long)t * hd + d];
		}
		long v = rsh(acc, n);
		s[t] = v;
		mx = max(mx, v);
	}
	long M = tg_max(mx, tg, tid);
	long z = 0;
	for (int t = tid; t < T; t += TG) {
		long w = expa[min(M - s[t], (long)(a.na - 1))];
		s[t] = w;
		z += w;
	}
	long Z = tg_sum(z, tg, tid);
	threadgroup_barrier(mem_flags::mem_device);
	// O[d] = sum_t w_t V[t][d]: TG threads = (TG / hd) slices of t x hd lanes
	int slices = TG / hd, sl = tid / hd, d = tid % hd;
	long od = 0;
	if (sl < slices) {
		for (int t = sl; t < T; t += slices) {
			long w = s[t];
			if (w != 0) {
				od += w * (long)V[(long)t * hd + d];
			}
		}
	}
	tg[tid] = od;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < (uint)hd) {
		long O = 0;
		for (int k = 0; k < slices; k++) {
			O += tg[k * hd + tid];
		}
		long aO = iabs(O), hi = aO / Z;
		long r = (hi << 12) + ((aO - hi * Z) << 12) / Z;
		o[h * hd + tid] = O < 0 ? -r : r;
	}
}

// blockf over n values -> mantissas m, exponent ex[eoutSlot] = ein + shift. One threadgroup.
kernel void int_blockf(device const long *x [[buffer(0)]], device int *m [[buffer(1)]], device int *ex [[buffer(2)]],
                       constant int &n [[buffer(3)]], constant int &ein [[buffer(4)]], constant int &eoutSlot [[buffer(5)]],
                       uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[TG];
	long mx = 0;
	for (int i = tid; i < n; i += TG) {
		mx = max(mx, iabs(x[i]));
	}
	int sh = max(0, bitlen(tg_max(mx, tg, tid)) - 15);
	for (int i = tid; i < n; i += TG) {
		m[i] = (int)clamp(rsh(x[i], sh), -32767L, 32767L);
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

// the frequency table: f = EXPF[min(M - l, nf - 1)], Z = sum f, freq = f * (2^28 - V) / Z + 1, cum = prefix sums.
kernel void int_table(device const long *lg [[buffer(0)]], device const int *expf_ [[buffer(1)]], device long *cum [[buffer(2)]],
                      constant int &V [[buffer(3)]], constant int &nf [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup long tg[TG];
	long mx = LMIN;
	for (int i = tid; i < V; i += TG) {
		mx = max(mx, lg[i]);
	}
	long M = tg_max(mx, tg, tid);
	long z = 0;
	for (int i = tid; i < V; i += TG) {
		z += expf_[min(M - lg[i], (long)(nf - 1))];
	}
	long Z = tg_sum(z, tg, tid);
	// contiguous chunk per thread, then a scan of the chunk totals
	int per = (V + TG - 1) / TG, lo = tid * per, hi = min(lo + per, V);
	long run = 0;
	for (int i = lo; i < hi; i++) {
		long f = expf_[min(M - lg[i], (long)(nf - 1))];
		run += f * ((1L << 28) - V) / Z + 1;
		cum[i] = run;
	}
	tg[tid] = run;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid == 0) {
		long acc = 0;
		for (int k = 0; k < TG; k++) {
			long v = tg[k];
			tg[k] = acc;
			acc += v;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	long off = tg[tid];
	for (int i = lo; i < hi; i++) {
		cum[i] += off;
	}
}
