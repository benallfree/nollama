// Integer-model kernels on tensor cores (compute 8.0+), loaded where they work and used in place of the
// same-named kernels of gpu_int.cu (int_bscores64 -> int_bscores_mma64, ...): same arguments, grid and results.
//
// Exactness: a 16-bit q is two byte planes (q = qh * 2^8 + ql, ql in [0, 255], qh in [-128, 127]); a 23-bit key
// value is three (k = k2 * 2^16 + k1 * 2^8 + k0, k0 and k1 in [0, 255], k2 in [-64, 63]). q . k is the sum over the
// 6 plane pairs of (qa . kb) * 2^(8a + 8b); each pair's sum over at most 128 dimensions is below 2^23 in magnitude,
// exact in the int8 mma's int32 accumulators, and the pairs combine in 64 bits: the integer int_bscores computes.

#include "gpu_int_prelude.cuh"

#define LMIN (-9223372036854775807L - 1)
#define KC 1024

struct IntParams {
	int tok;
	int pos;
};
struct BAttnArgs {
	int nq, nkv, hd, maxlen, na, smax, nchmax, ez;
};

static __device__ __forceinline__ long rsh(long x, int n) { return n <= 0 ? (x << (-n)) : ((x + (1L << (n - 1))) >> n); }

// D += A (16x32, row) * B (32x8, col), int8 x int8 -> int32; signedness per operand
template <bool AS, bool BS>
static __device__ __forceinline__ void mma8(int c[4], const uint a[4], const uint b[2]);
#define MMA8(AS, BS, AT, BT)                                                                                               \
	template <> __device__ __forceinline__ void mma8<AS, BS>(int c[4], const uint a[4], const uint b[2]) {                \
		asm volatile("mma.sync.aligned.m16n8k32.row.col.s32." AT "." BT ".s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "     \
		             "{%0,%1,%2,%3};"                                                                                      \
		             : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])                                                   \
		             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));                                 \
	}
MMA8(false, false, "u8", "u8")
MMA8(false, true, "u8", "s8")
MMA8(true, false, "s8", "u8")
MMA8(true, true, "s8", "s8")

// Scores on tensor cores. A threadgroup per (kv head, block of QR / grp rows, key chunk), as int_bscores: QR query
// rows (grp heads x rows) against the chunk's keys in tiles of KT. Warp w computes m-tile w % MT (16 query rows) for
// its share of the KT / 8 n-tiles (8 keys each).
template <int HD, int QR, int KT, int X = 0>
static __device__ __forceinline__ void bscores_mma(const int *mq, const int *shq, const int *Kc, int *S, long *cmax, int *ovf,
                                                   int base, int R, const BAttnArgs &a, uint3 tg3, uint tid,
                                                   unsigned char *qp, unsigned char *kp, int *shs, long *pmx) {
	constexpr int LD = HD + 16;          // plane rows padded (banks)
	constexpr int MT = QR / 16, NW = 8 / MT, NT = KT / 8, NPW = NT / NW;
	int j = tg3.x, grp = a.nq / a.nkv, br = QR / grp;
	int t0 = tg3.y * br, c = tg3.z;
	int pmax = base + min(t0 + br, R) - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	// query planes: qp[plane][qrow][d]
	for (int i = tid; i < QR * HD; i += 256) {
		int qr = i / HD, d = i % HD, g = qr / br, r = qr % br;
		int q = g < grp && t0 + r < R ? mq[((long)(t0 + r) * a.nq + j * grp + g) * HD + d] : 0;
		qp[qr * LD + d] = (unsigned char)(q & 255);
		qp[QR * LD + qr * LD + d] = (unsigned char)(q >> 8);
	}
	if (tid < QR) {
		int g = tid / br, r = tid % br;
		shs[tid] = g < grp && t0 + r < R ? shq[(t0 + r) * a.nq + j * grp + g] : 0;
	}
	const int *K = Kc + (long)j * a.maxlen * HD;
	int w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	int mt = w % MT, nw = w / MT;
	// this thread's two query rows (C layout): mt * 16 + gid and + 8
	int qrA = mt * 16 + gid, qrB = qrA + 8;
	long mx[2] = {LMIN, LMIN};
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		__syncthreads();
		for (int i = tid; i < KT * HD; i += 256) {
			int kk = i / HD, d = i % HD;
			int v = X == 2 ? i : K[(long)k0 * HD + i];
			kp[kk * LD + d] = (unsigned char)(v & 255);
			kp[KT * LD + kk * LD + d] = (unsigned char)((v >> 8) & 255);
			kp[2 * KT * LD + kk * LD + d] = (unsigned char)(v >> 16);
		}
		__syncthreads();
		long acc[NPW][4];
		for (int n = 0; n < NPW; n++) {
			for (int e = 0; e < 4; e++) {
				acc[n][e] = 0;
			}
		}
#pragma unroll
		for (int pa = 0; pa < 2; pa++) {
#pragma unroll
			for (int pb = 0; pb < 3; pb++) {
				int cc[NPW][4];
				for (int n = 0; n < NPW; n++) {
					for (int e = 0; e < 4; e++) {
						cc[n][e] = 0;
					}
				}
				const unsigned char *A = qp + pa * QR * LD, *B = kp + pb * KT * LD;
#pragma unroll
				for (int kd = 0; kd < HD; kd += 32) {
					uint af[4];
					af[0] = *(const uint *)(A + qrA * LD + kd + tig * 4);
					af[1] = *(const uint *)(A + qrB * LD + kd + tig * 4);
					af[2] = *(const uint *)(A + qrA * LD + kd + 16 + tig * 4);
					af[3] = *(const uint *)(A + qrB * LD + kd + 16 + tig * 4);
					for (int n = 0; n < NPW; n++) {
						int key = (nw * NPW + n) * 8 + gid;
						uint bf[2];
						bf[0] = *(const uint *)(B + key * LD + kd + tig * 4);
						bf[1] = *(const uint *)(B + key * LD + kd + 16 + tig * 4);
						if (X == 3) {
							cc[n][0] += af[0] ^ bf[0];
						} else if (pa == 0 && pb < 2) {
							mma8<false, false>(cc[n], af, bf);
						} else if (pa == 0) {
							mma8<false, true>(cc[n], af, bf);
						} else if (pb < 2) {
							mma8<true, false>(cc[n], af, bf);
						} else {
							mma8<true, true>(cc[n], af, bf);
						}
					}
				}
				int sh = 8 * pa + 8 * pb;
				for (int n = 0; n < NPW; n++) {
					for (int e = 0; e < 4; e++) {
						acc[n][e] += (long)cc[n][e] << sh;
					}
				}
			}
		}
		// scores: element e of n-tile n is query row (e < 2 ? qrA : qrB), key (nw * NPW + n) * 8 + tig * 2 + e % 2
		for (int n = 0; n < NPW; n++) {
			for (int e = 0; e < 4; e++) {
				int qr = e < 2 ? qrA : qrB, g = qr / br, r = qr % br;
				int t = k0 + (nw * NPW + n) * 8 + tig * 2 + (e & 1);
				int p = base + t0 + r;
				if (g < grp && t0 + r < R && t <= p) {
					long s = rsh(acc[n][e], 42 - shs[qr]);
					if (s != (long)(int)s) {
						atomicExch(ovf, 1);
					}
					if (X != 1) {
						S[((long)(t0 + r) * a.nq + j * grp + g) * a.smax + t] = (int)s;
					}
					mx[e >> 1] = max(mx[e >> 1], s);
				}
			}
		}
	}
	// row maxima: the 4 lanes of a group share rows; then the NW warps sharing an m-tile
	for (int h = 0; h < 2; h++) {
		long m = mx[h];
		m = max(m, shx(m, 1));
		m = max(m, shx(m, 2));
		mx[h] = m;
	}
	if (tig == 0) {
		pmx[nw * QR + qrA] = mx[0];
		pmx[nw * QR + qrB] = mx[1];
	}
	__syncthreads();
	if (tid < QR) {
		int qr = tid, g = qr / br, r = qr % br;
		long m = LMIN;
		for (int k = 0; k < NW; k++) {
			m = max(m, pmx[k * QR + qr]);
		}
		if (g < grp && t0 + r < R) {
			cmax[((long)(t0 + r) * a.nq + j * grp + g) * a.nchmax + c] = m;
		}
	}
}

#define BSCORES_MMA(NAME, HD, QR, KT, X)                                                                                      \
	extern "C" __global__ void NAME(const int *mq, const int *shq, const int *Kc, int *S, long *cmax, int *ovf,            \
	                                const IntParams *pr, BAttnArgs a) {                                                  \
		__shared__ __align__(16) unsigned char qp[2 * QR * (HD + 16)];                                                     \
		__shared__ __align__(16) unsigned char kp[3 * KT * (HD + 16)];                                                     \
		__shared__ int shs[QR];                                                                                            \
		__shared__ long pmx[(8 / (QR / 16)) * QR];                                                                         \
		bscores_mma<HD, QR, KT, X>(mq, shq, Kc, S, cmax, ovf, pr->pos, pr->tok, a,                                            \
		                        make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), threadIdx.x, qp, kp, shs, pmx);            \
	}
BSCORES_MMA(int_bscores_mma64, 64, 128, 64, 0)
BSCORES_MMA(int_bscores_mma128, 128, 64, 32, 0)

// Weights x V on tensor cores, in place of int_bpv: per (query head, block of BRV rows, key chunk) the weights
// w = EXPA[min(M - s, na - 1)] (w <= 2^15: two unsigned byte planes) times V (three planes, as the keys), the chunk's
// Z and O. Weights as int_bpv computes them (EXPA's nonzero head in shared memory, stale rows past the batch skipped,
// all-zero tiles skipped); O[row, d] = sum over keys of w V by mma: m = rows, n = dimensions, k = keys (KT = 32).
#define ETN 11392
template <int HD>
static __device__ __forceinline__ void bpv_mma(const int *S, const long *cmax, const int *Vc, const int *expa, long *zp,
                                               long *op, int base, int R, const BAttnArgs &a, uint3 tg3, uint tid,
                                               unsigned char *wp, unsigned char *vp, long *Ms, int *zs, ushort *et, int *nz) {
	constexpr int KT = 32, BRV = HD == 64 ? 64 : 32, MT = BRV / 16, NT = HD / 8, NW = 8 / MT, NPW = NT / NW;
	constexpr int LW = KT + 16, LV = KT + 16; // padded rows: wp[plane][row][key], vp[plane][d][key]
	int h = tg3.x, t0 = tg3.y * BRV, c = tg3.z, grp = a.nq / a.nkv, j = h / grp;
	int pmax = base + t0 + BRV - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int ez = a.ez;
	for (int i = tid; i < ez; i += 256) {
		et[i] = (ushort)expa[i];
	}
	if (tid < (uint)BRV) {
		int p = base + t0 + tid, nch = p / KC + 1;
		const long *cm = cmax + ((long)(t0 + tid) * a.nq + h) * a.nchmax;
		long m = LMIN;
		for (int i = 0; i < nch; i++) {
			m = max(m, cm[i]);
		}
		Ms[tid] = m;
		zs[tid] = 0;
	}
	const int *V = Vc + (long)j * a.maxlen * HD;
	int w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	int mt = w % MT, nw = w / MT;
	long acc[NPW][4];
	for (int n = 0; n < NPW; n++) {
		for (int e = 0; e < 4; e++) {
			acc[n][e] = 0;
		}
	}
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		__syncthreads();
		if (tid == 0) {
			*nz = 0;
		}
		__syncthreads();
		// weights: BRV rows x 32 keys, a warp per row at a time (lane = key)
		int any = 0;
		for (int r = w; r < BRV; r += 8) {
			int t = k0 + lane, p = base + t0 + r;
			int wv = 0;
			if (t <= p && t0 + r < R) {
				long d = Ms[r] - S[((long)(t0 + r) * a.nq + h) * a.smax + t];
				if (d >= 0 && d < ez) {
					wv = et[d];
				}
			}
			wp[r * LW + lane] = (unsigned char)(wv & 255);
			wp[BRV * LW + r * LW + lane] = (unsigned char)(wv >> 8);
			any |= wv;
			int z = wv;
			for (int o = 16; o > 0; o >>= 1) {
				z += __shfl_xor_sync(0xffffffffu, z, o);
			}
			if (lane == 0) {
				zs[r] += z;
			}
		}
		if (__any_sync(0xffffffffu, any != 0) && lane == 0) {
			*nz = 1;
		}
		__syncthreads();
		if (*nz == 0) {
			continue; // every weight 0: nothing to add
		}
		// V tile, transposed into planes: vp[plane][d][key]
		for (int i = tid; i < KT * HD; i += 256) {
			int kk = i / HD, d = i % HD;
			int v = V[(long)k0 * HD + i];
			vp[d * LV + kk] = (unsigned char)(v & 255);
			vp[HD * LV + d * LV + kk] = (unsigned char)((v >> 8) & 255);
			vp[2 * HD * LV + d * LV + kk] = (unsigned char)(v >> 16);
		}
		__syncthreads();
		int rA = mt * 16 + gid, rB = rA + 8;
#pragma unroll
		for (int pa = 0; pa < 2; pa++) {
			const unsigned char *A = wp + pa * BRV * LW;
			uint af[4];
			af[0] = *(const uint *)(A + rA * LW + tig * 4);
			af[1] = *(const uint *)(A + rB * LW + tig * 4);
			af[2] = *(const uint *)(A + rA * LW + 16 + tig * 4);
			af[3] = *(const uint *)(A + rB * LW + 16 + tig * 4);
#pragma unroll
			for (int pb = 0; pb < 3; pb++) {
				const unsigned char *B = vp + pb * HD * LV;
				for (int n = 0; n < NPW; n++) {
					int d = (nw * NPW + n) * 8 + gid;
					uint bf[2];
					bf[0] = *(const uint *)(B + d * LV + tig * 4);
					bf[1] = *(const uint *)(B + d * LV + 16 + tig * 4);
					int cc[4] = {0, 0, 0, 0};
					if (pb < 2) {
						mma8<false, false>(cc, af, bf);
					} else {
						mma8<false, true>(cc, af, bf);
					}
					for (int e = 0; e < 4; e++) {
						acc[n][e] += (long)cc[e] << (8 * pa + 8 * pb);
					}
				}
			}
		}
	}
	__syncthreads();
	// element e of n-tile n: row (e < 2 ? rA : rB), dimension (nw * NPW + n) * 8 + tig * 2 + e % 2
	int rA = mt * 16 + gid;
	for (int n = 0; n < NPW; n++) {
		for (int e = 0; e < 4; e++) {
			int r = e < 2 ? rA : rA + 8, d = (nw * NPW + n) * 8 + tig * 2 + (e & 1);
			op[(((long)(t0 + r) * a.nq + h) * a.nchmax + c) * HD + d] = acc[n][e];
		}
	}
	if (tid < (uint)BRV) {
		zp[((long)(t0 + tid) * a.nq + h) * a.nchmax + c] = zs[tid];
	}
}

#define BPV_MMA(NAME, HD)                                                                                                  \
	extern "C" __global__ void NAME(const int *S, const long *cmax, const int *Vc, const int *expa, long *zp, long *op,     \
	                                const IntParams *pr, BAttnArgs a) {                                                  \
		constexpr int BRV = HD == 64 ? 64 : 32;                                                                            \
		__shared__ __align__(16) unsigned char wp[2 * BRV * 48];                                                           \
		__shared__ __align__(16) unsigned char vp[3 * HD * 48];                                                            \
		__shared__ long Ms[BRV];                                                                                           \
		__shared__ int zs[BRV];                                                                                            \
		__shared__ ushort et[ETN];                                                                                         \
		__shared__ int nz;                                                                                                 \
		bpv_mma<HD>(S, cmax, Vc, expa, zp, op, pr->pos, pr->tok, a, make_uint3(blockIdx.x, blockIdx.y, blockIdx.z),       \
		            threadIdx.x, wp, vp, Ms, zs, et, &nz);                                                                 \
	}
BPV_MMA(int_bpv_mma64, 64)
BPV_MMA(int_bpv_mma128, 128)

// ---- byte-plane KV cache: K and V kept as planes as they are written, so attention needn't split them ----
// Per layer: Kp[j][p][t][d] (key rows) and Vp[j][p][d][t] (transposed, for weights x V), p = 0, 1 (unsigned bytes),
// 2 (signed high byte); Mp positions per plane (the cache length plus a tile's padding, a multiple of 16).

// int_kvplanes: after int_qkv, the planes of rows' keys and values (positions pos + row).
extern "C" __global__ void int_kvplanes(const int *Kc, const int *Vc, unsigned char *Kp, unsigned char *Vp,
                                        const IntParams *pr, int nkv, int hd, int maxlen, int Mp) {
	int row = blockIdx.x, t = pr->pos + row;
	for (int i = threadIdx.x; i < nkv * hd; i += blockDim.x) {
		int j = i / hd, d = i % hd;
		int k = Kc[((long)j * maxlen + t) * hd + d], v = Vc[((long)j * maxlen + t) * hd + d];
		for (int p = 0; p < 3; p++) {
			int kb = p < 2 ? (k >> (8 * p)) & 255 : k >> 16, vb = p < 2 ? (v >> (8 * p)) & 255 : v >> 16;
			Kp[(((long)j * 3 + p) * Mp + t) * hd + d] = (unsigned char)kb;
			Vp[(((long)j * 3 + p) * hd + d) * Mp + t] = (unsigned char)vb;
		}
	}
}

// pass 1: the chunk maxima of the scores (int_bscores without storing them), keys from the planes.
template <int HD, int QR, int KT>
static __device__ __forceinline__ void bmax_mma(const int *mq, const int *shq, long *cmax, int *ovf, const unsigned char *Kp,
                                                int Mp, int base, int R, const BAttnArgs &a, uint3 tg3, uint tid,
                                                unsigned char *qp, unsigned char *kp, int *shs, long *pmx) {
	constexpr int LD = HD + 16;
	constexpr int MT = QR / 16, NW = 8 / MT, NT = KT / 8, NPW = NT / NW;
	int j = tg3.x, grp = a.nq / a.nkv, br = QR / grp;
	int t0 = tg3.y * br, c = tg3.z;
	int pmax = base + min(t0 + br, R) - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	for (int i = tid; i < QR * HD; i += 256) {
		int qr = i / HD, d = i % HD, g = qr / br, r = qr % br;
		int q = g < grp && t0 + r < R ? mq[((long)(t0 + r) * a.nq + j * grp + g) * HD + d] : 0;
		qp[qr * LD + d] = (unsigned char)(q & 255);
		qp[QR * LD + qr * LD + d] = (unsigned char)(q >> 8);
	}
	if (tid < QR) {
		int g = tid / br, r = tid % br;
		shs[tid] = g < grp && t0 + r < R ? shq[(t0 + r) * a.nq + j * grp + g] : 0;
	}
	int w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	int mt = w % MT, nw = w / MT;
	int qrA = mt * 16 + gid, qrB = qrA + 8;
	long mx[2] = {LMIN, LMIN};
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		__syncthreads();
		for (int p = 0; p < 3; p++) {
			for (int i = tid; i < KT * HD / 16; i += 256) { // rows of HD bytes, 16 at a time
				int kk = i / (HD / 16), q = i % (HD / 16);
				*(uint4 *)(kp + p * KT * LD + kk * LD + q * 16) =
				    *(const uint4 *)(Kp + (((long)j * 3 + p) * Mp + k0 + kk) * HD + q * 16);
			}
		}
		__syncthreads();
		long acc[NPW][4];
		for (int n = 0; n < NPW; n++) {
			for (int e = 0; e < 4; e++) {
				acc[n][e] = 0;
			}
		}
#pragma unroll
		for (int pa = 0; pa < 2; pa++) {
#pragma unroll
			for (int pb = 0; pb < 3; pb++) {
				int cc[NPW][4];
				for (int n = 0; n < NPW; n++) {
					for (int e = 0; e < 4; e++) {
						cc[n][e] = 0;
					}
				}
				const unsigned char *A = qp + pa * QR * LD, *B = kp + pb * KT * LD;
#pragma unroll
				for (int kd = 0; kd < HD; kd += 32) {
					uint af[4];
					af[0] = *(const uint *)(A + qrA * LD + kd + tig * 4);
					af[1] = *(const uint *)(A + qrB * LD + kd + tig * 4);
					af[2] = *(const uint *)(A + qrA * LD + kd + 16 + tig * 4);
					af[3] = *(const uint *)(A + qrB * LD + kd + 16 + tig * 4);
					for (int n = 0; n < NPW; n++) {
						int key = (nw * NPW + n) * 8 + gid;
						uint bf[2];
						bf[0] = *(const uint *)(B + key * LD + kd + tig * 4);
						bf[1] = *(const uint *)(B + key * LD + kd + 16 + tig * 4);
						if (pa == 0 && pb < 2) {
							mma8<false, false>(cc[n], af, bf);
						} else if (pa == 0) {
							mma8<false, true>(cc[n], af, bf);
						} else if (pb < 2) {
							mma8<true, false>(cc[n], af, bf);
						} else {
							mma8<true, true>(cc[n], af, bf);
						}
					}
				}
				for (int n = 0; n < NPW; n++) {
					for (int e = 0; e < 4; e++) {
						acc[n][e] += (long)cc[n][e] << (8 * pa + 8 * pb);
					}
				}
			}
		}
		for (int n = 0; n < NPW; n++) {
			for (int e = 0; e < 4; e++) {
				int qr = e < 2 ? qrA : qrB, g = qr / br, r = qr % br;
				int t = k0 + (nw * NPW + n) * 8 + tig * 2 + (e & 1);
				if (g < grp && t0 + r < R && t <= base + t0 + r) {
					long s = rsh(acc[n][e], 42 - shs[qr]);
					if (s != (long)(int)s) {
						atomicExch(ovf, 1); // the caller then reruns the batch with int_attnb*
					}
					mx[e >> 1] = max(mx[e >> 1], s);
				}
			}
		}
	}
	for (int h = 0; h < 2; h++) {
		long m = mx[h];
		m = max(m, shx(m, 1));
		m = max(m, shx(m, 2));
		mx[h] = m;
	}
	if (tig == 0) {
		pmx[nw * QR + qrA] = mx[0];
		pmx[nw * QR + qrB] = mx[1];
	}
	__syncthreads();
	if (tid < QR) {
		int qr = tid, g = qr / br, r = qr % br;
		long m = LMIN;
		for (int k = 0; k < NW; k++) {
			m = max(m, pmx[k * QR + qr]);
		}
		if (g < grp && t0 + r < R) {
			cmax[((long)(t0 + r) * a.nq + j * grp + g) * a.nchmax + c] = m;
		}
	}
}

// arguments: int_bscores's, then the planes (Kp, Vp at the layer), Mp
#define BMAX_MMA(NAME, HD, QR, KT)                                                                                         \
	extern "C" __global__ void NAME(const int *mq, const int *shq, const int *Kc, int *S, long *cmax, int *ovf,            \
	                                const IntParams *pr, BAttnArgs a, const unsigned char *Kp, const unsigned char *Vp,  \
	                                int Mp) {                                                                            \
		__shared__ __align__(16) unsigned char qp[2 * QR * (HD + 16)];                                                     \
		__shared__ __align__(16) unsigned char kp[3 * KT * (HD + 16)];                                                     \
		__shared__ int shs[QR];                                                                                            \
		__shared__ long pmx[(8 / (QR / 16)) * QR];                                                                         \
		bmax_mma<HD, QR, KT>(mq, shq, cmax, ovf, Kp, Mp, pr->pos, pr->tok, a,                                              \
		                     make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), threadIdx.x, qp, kp, shs, pmx);               \
	}
BMAX_MMA(int_bmax_mma64, 64, 128, 64)
BMAX_MMA(int_bmax_mma128, 128, 64, 32)

// pass 2: per (query head, block of BRV rows, key chunk): the scores again (tensor cores make that cheaper than
// storing them), weights w = EXPA[min(M - s, na - 1)], the chunk's Z and O = sum w V; partials to zp, op as int_bpv.
template <int HD>
static __device__ __forceinline__ void bpv2_mma(const long *cmax, const int *expa, long *zp, long *op, const unsigned char *Kp,
                                                const unsigned char *Vp, int Mp, const int *mq, const int *shq, int base,
                                                int R, const BAttnArgs &a, uint3 tg3, uint tid, unsigned char *qp,
                                                unsigned char *kp, unsigned char *vp, unsigned char *wp, long *Ms, int *zs,
                                                int *shs) {
	constexpr int KT = 32, BRV = HD == 64 ? 64 : 32, MT = BRV / 16, NW = 8 / MT;
	constexpr int LD = HD + 16, LW = KT + 16;
	constexpr int SNT = KT / 8, SNPW = SNT / NW; // score n-tiles (keys) per warp
	constexpr int ONT = HD / 8, ONPW = ONT / NW; // output n-tiles (dimensions) per warp
	int h = tg3.x, t0 = tg3.y * BRV, c = tg3.z, grp = a.nq / a.nkv, j = h / grp;
	int pmax = base + t0 + BRV - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int ez = a.ez;
	for (int i = tid; i < BRV * HD; i += 256) {
		int r = i / HD, d = i % HD;
		int q = t0 + r < R ? mq[((long)(t0 + r) * a.nq + h) * HD + d] : 0;
		qp[r * LD + d] = (unsigned char)(q & 255);
		qp[BRV * LD + r * LD + d] = (unsigned char)(q >> 8);
	}
	if (tid < (uint)BRV) {
		int r = tid, p = base + t0 + r, nch = p / KC + 1;
		long m = LMIN;
		if (t0 + r < R) {
			const long *cm = cmax + ((long)(t0 + r) * a.nq + h) * a.nchmax;
			for (int i = 0; i < nch; i++) {
				m = max(m, cm[i]);
			}
		}
		Ms[r] = m;
		zs[r] = 0;
		shs[r] = t0 + r < R ? shq[(t0 + r) * a.nq + h] : 0;
	}
	int w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	int mt = w % MT, nw = w / MT;
	int rA = mt * 16 + gid, rB = rA + 8;
	int zl[2] = {0, 0};
	long acc[ONPW][4];
	for (int n = 0; n < ONPW; n++) {
		for (int e = 0; e < 4; e++) {
			acc[n][e] = 0;
		}
	}
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		__syncthreads();
		for (int p = 0; p < 3; p++) {
			for (int i = tid; i < KT * HD / 16; i += 256) {
				int kk = i / (HD / 16), q = i % (HD / 16);
				*(uint4 *)(kp + p * KT * LD + kk * LD + q * 16) =
				    *(const uint4 *)(Kp + (((long)j * 3 + p) * Mp + k0 + kk) * HD + q * 16);
			}
		}
		__syncthreads();
		// scores for rows rA, rB x this warp's keys
		long sc[SNPW][4];
		for (int n = 0; n < SNPW; n++) {
			for (int e = 0; e < 4; e++) {
				sc[n][e] = 0;
			}
		}
#pragma unroll
		for (int pa = 0; pa < 2; pa++) {
#pragma unroll
			for (int pb = 0; pb < 3; pb++) {
				int cc[SNPW][4];
				for (int n = 0; n < SNPW; n++) {
					for (int e = 0; e < 4; e++) {
						cc[n][e] = 0;
					}
				}
				const unsigned char *A = qp + pa * BRV * LD, *B = kp + pb * KT * LD;
#pragma unroll
				for (int kd = 0; kd < HD; kd += 32) {
					uint af[4];
					af[0] = *(const uint *)(A + rA * LD + kd + tig * 4);
					af[1] = *(const uint *)(A + rB * LD + kd + tig * 4);
					af[2] = *(const uint *)(A + rA * LD + kd + 16 + tig * 4);
					af[3] = *(const uint *)(A + rB * LD + kd + 16 + tig * 4);
					for (int n = 0; n < SNPW; n++) {
						int key = (nw * SNPW + n) * 8 + gid;
						uint bf[2];
						bf[0] = *(const uint *)(B + key * LD + kd + tig * 4);
						bf[1] = *(const uint *)(B + key * LD + kd + 16 + tig * 4);
						if (pa == 0 && pb < 2) {
							mma8<false, false>(cc[n], af, bf);
						} else if (pa == 0) {
							mma8<false, true>(cc[n], af, bf);
						} else if (pb < 2) {
							mma8<true, false>(cc[n], af, bf);
						} else {
							mma8<true, true>(cc[n], af, bf);
						}
					}
				}
				for (int n = 0; n < SNPW; n++) {
					for (int e = 0; e < 4; e++) {
						sc[n][e] += (long)cc[n][e] << (8 * pa + 8 * pb);
					}
				}
			}
		}
		// weights into planes wp[plane][row][key]
		int any = 0;
		for (int n = 0; n < SNPW; n++) {
			for (int e = 0; e < 4; e++) {
				int r = e < 2 ? rA : rB, kk = (nw * SNPW + n) * 8 + tig * 2 + (e & 1), t = k0 + kk;
				int wv = 0;
				if (t0 + r < R && t <= base + t0 + r) {
					long d = Ms[r] - rsh(sc[n][e], 42 - shs[r]);
					if (d >= 0 && d < ez) {
						wv = expa[d];
					}
				}
				wp[r * LW + kk] = (unsigned char)(wv & 255);
				wp[BRV * LW + r * LW + kk] = (unsigned char)(wv >> 8);
				zl[e >> 1] += wv;
				any |= wv;
			}
		}
		if (!__syncthreads_or(any)) {
			continue; // every weight 0: nothing to add
		}
		for (int p = 0; p < 3; p++) {
			for (int i = tid; i < HD * 2; i += 256) { // KT = 32 keys: two 16-byte pieces per dimension
				int d = i / 2, q = i % 2;
				*(uint4 *)(vp + p * HD * LW + d * LW + q * 16) = *(const uint4 *)(Vp + (((long)j * 3 + p) * HD + d) * Mp + k0 + q * 16);
			}
		}
		__syncthreads();
#pragma unroll
		for (int pa = 0; pa < 2; pa++) {
			const unsigned char *A = wp + pa * BRV * LW;
			uint af[4];
			af[0] = *(const uint *)(A + rA * LW + tig * 4);
			af[1] = *(const uint *)(A + rB * LW + tig * 4);
			af[2] = *(const uint *)(A + rA * LW + 16 + tig * 4);
			af[3] = *(const uint *)(A + rB * LW + 16 + tig * 4);
#pragma unroll
			for (int pb = 0; pb < 3; pb++) {
				const unsigned char *B = vp + pb * HD * LW;
				for (int n = 0; n < ONPW; n++) {
					int d = (nw * ONPW + n) * 8 + gid;
					uint bf[2];
					bf[0] = *(const uint *)(B + d * LW + tig * 4);
					bf[1] = *(const uint *)(B + d * LW + 16 + tig * 4);
					int cc[4] = {0, 0, 0, 0};
					if (pb < 2) {
						mma8<false, false>(cc, af, bf);
					} else {
						mma8<false, true>(cc, af, bf);
					}
					for (int e = 0; e < 4; e++) {
						acc[n][e] += (long)cc[e] << (8 * pa + 8 * pb);
					}
				}
			}
		}
	}
	// Z: the 4 lanes of a group, then the warps sharing the rows
	for (int k = 0; k < 2; k++) {
		int z = zl[k];
		z += __shfl_xor_sync(0xffffffffu, z, 1);
		z += __shfl_xor_sync(0xffffffffu, z, 2);
		if (tig == 0) {
			atomicAdd(&zs[k == 0 ? rA : rB], z);
		}
	}
	__syncthreads();
	for (int n = 0; n < ONPW; n++) {
		for (int e = 0; e < 4; e++) {
			int r = e < 2 ? rA : rB, d = (nw * ONPW + n) * 8 + tig * 2 + (e & 1);
			op[(((long)(t0 + r) * a.nq + h) * a.nchmax + c) * HD + d] = acc[n][e];
		}
	}
	if (tid < (uint)BRV) {
		zp[((long)(t0 + tid) * a.nq + h) * a.nchmax + c] = zs[tid];
	}
}

// arguments: int_bpv's, then the planes (Kp, Vp at the layer), Mp, the query mantissas and shifts
#define BPV2_MMA(NAME, HD)                                                                                                 \
	extern "C" __global__ void NAME(const int *S, const long *cmax, const int *Vc, const int *expa, long *zp, long *op,     \
	                                const IntParams *pr, BAttnArgs a, const unsigned char *Kp, const unsigned char *Vp,  \
	                                int Mp, const int *mq, const int *shq) {                                             \
		constexpr int BRV = HD == 64 ? 64 : 32;                                                                            \
		__shared__ __align__(16) unsigned char qp[2 * BRV * (HD + 16)];                                                    \
		__shared__ __align__(16) unsigned char kp[3 * 32 * (HD + 16)];                                                     \
		__shared__ __align__(16) unsigned char vp[3 * HD * 48];                                                            \
		__shared__ __align__(16) unsigned char wp[2 * BRV * 48];                                                           \
		__shared__ long Ms[BRV];                                                                                           \
		__shared__ int zs[BRV], shs[BRV];                                                                                  \
		bpv2_mma<HD>(cmax, expa, zp, op, Kp, Vp, Mp, mq, shq, pr->pos, pr->tok, a,                                        \
		             make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), threadIdx.x, qp, kp, vp, wp, Ms, zs, shs);            \
	}
BPV2_MMA(int_bpv2_mma64, 64)
BPV2_MMA(int_bpv2_mma128, 128)

// int_gemm on tensor cores, same arguments, grid (64 outputs x 32 rows a threadgroup) and results: per 64-group the
// activations' two byte planes (x = xh * 2^8 + xl) times the int8 weights, each an exact int32 mma sum, combined as
// 256 * (w . xh) + (w . xl) (|.| < 2^28), then scaled into 64 bits as int_gemm does.
extern "C" __global__ void int_gemm_mma(const signed char *q, const short *s, const short *x, long *out, const int *ex, int O,
                                        int K, int E, int einSlot, int to, int4 st) {
	constexpr int LD = 64 + 16;
	__shared__ __align__(16) unsigned char xp[2 * 32 * LD]; // xp[plane][row][k]
	__shared__ __align__(16) signed char wq[64 * LD];       // wq[out][k]
	__shared__ short sc[64];
	int xS = st.y, outS = st.z;
	int o0 = blockIdx.x * 64, t0 = blockIdx.y * 32;
	int tid = threadIdx.x, w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	int mt = w % 2, nt0 = (w / 2) * 2; // warp: rows mt * 16 .. +15, outputs (nt0 .. nt0 + 1) * 8 .. +7
	int ng = K / 64;
	long acc[2][4];
	for (int n = 0; n < 2; n++) {
		for (int e = 0; e < 4; e++) {
			acc[n][e] = 0;
		}
	}
	for (int g = 0; g < ng; g++) {
		__syncthreads();
		{ // activations: 32 rows x 64 shorts -> planes; weights: 64 rows x 64 bytes
			for (int i = tid; i < 32 * 64; i += 256) {
				int r = i / 64, k = i % 64;
				int v = x[(long)(t0 + r) * xS + g * 64 + k];
				xp[r * LD + k] = (unsigned char)(v & 255);
				xp[32 * LD + r * LD + k] = (unsigned char)(v >> 8);
			}
			int r = tid / 4, part = tid % 4;
			*(uint4 *)(wq + r * LD + part * 16) = *(const uint4 *)(q + (long)min(o0 + r, O - 1) * K + g * 64 + part * 16);
			if (tid < 64) {
				sc[tid] = s[(long)min(o0 + tid, O - 1) * ng + g];
			}
		}
		__syncthreads();
		int ch[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}}, cl[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
		int rA = mt * 16 + gid, rB = rA + 8;
#pragma unroll
		for (int kd = 0; kd < 64; kd += 32) {
			uint al[4], ah[4];
			const unsigned char *L = xp, *H = xp + 32 * LD;
			al[0] = *(const uint *)(L + rA * LD + kd + tig * 4);
			al[1] = *(const uint *)(L + rB * LD + kd + tig * 4);
			al[2] = *(const uint *)(L + rA * LD + kd + 16 + tig * 4);
			al[3] = *(const uint *)(L + rB * LD + kd + 16 + tig * 4);
			ah[0] = *(const uint *)(H + rA * LD + kd + tig * 4);
			ah[1] = *(const uint *)(H + rB * LD + kd + tig * 4);
			ah[2] = *(const uint *)(H + rA * LD + kd + 16 + tig * 4);
			ah[3] = *(const uint *)(H + rB * LD + kd + 16 + tig * 4);
			for (int n = 0; n < 2; n++) {
				int o = (nt0 + n) * 8 + gid;
				uint bf[2];
				bf[0] = *(const uint *)(wq + o * LD + kd + tig * 4);
				bf[1] = *(const uint *)(wq + o * LD + kd + 16 + tig * 4);
				mma8<false, true>(cl[n], al, bf);
				mma8<true, true>(ch[n], ah, bf);
			}
		}
		for (int n = 0; n < 2; n++) {
			for (int e = 0; e < 4; e++) {
				int o = (nt0 + n) * 8 + tig * 2 + (e & 1);
				int d = ch[n][e] * 256 + cl[n][e];
				acc[n][e] += (long)d * sc[o];
			}
		}
	}
	// element e of n-tile n: row (e < 2 ? rA : rB), output (nt0 + n) * 8 + tig * 2 + e % 2
	for (int n = 0; n < 2; n++) {
		for (int e = 0; e < 4; e++) {
			int r = mt * 16 + gid + (e < 2 ? 0 : 8), o = o0 + (nt0 + n) * 8 + tig * 2 + (e & 1), t = t0 + r;
			if (o < O) {
				out[(long)t * outS + o] = rsh(acc[n][e], to - (E + ex[4 * t + einSlot]));
			}
		}
	}
}

// ---- a step's attention on tensor cores (in place of int_attn1 / int_attn2, same grids and outputs) ----
#define ACH 256
struct AttnArgs {
	int nq, nkv, hd, maxlen, na, maxch;
};

// pass 1: a threadgroup per (kv head, chunk of 256 positions): the grp query heads (rows of an m16 tile) against the
// chunk's keys (warp w: keys 32w .. 32w+31), from the key planes; scores to S, the chunk's maxima to cmax.
template <int HD>
static __device__ __forceinline__ void dattn1(const int *mq, const int *shq, long *S, long *cmax, const IntParams *pr,
                                              const AttnArgs &a, const unsigned char *Kp, int Mp, unsigned char *qp,
                                              int *shs, long *pm) {
	constexpr int LD = HD + 16;
	int j = blockIdx.x, c = blockIdx.y, T = pr->pos + 1, grp = a.nq / a.nkv;
	if (c * ACH >= T) {
		return;
	}
	int tid = threadIdx.x, w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	for (int i = tid; i < 16 * HD; i += 256) {
		int g = i / HD, d = i % HD;
		int q = g < grp ? mq[(j * grp + g) * HD + d] : 0;
		qp[g * LD + d] = (unsigned char)(q & 255);
		qp[16 * LD + g * LD + d] = (unsigned char)(q >> 8);
	}
	if (tid < 16) {
		shs[tid] = tid < grp ? shq[j * grp + tid] : 0;
	}
	__syncthreads();
	long acc[4][4];
	for (int n = 0; n < 4; n++) {
		for (int e = 0; e < 4; e++) {
			acc[n][e] = 0;
		}
	}
	int kb = c * ACH + w * 32; // this warp's first key
#pragma unroll
	for (int pa = 0; pa < 2; pa++) {
#pragma unroll
		for (int pb = 0; pb < 3; pb++) {
			int cc[4][4] = {{0}};
			const unsigned char *A = qp + pa * 16 * LD, *B = Kp + ((long)j * 3 + pb) * Mp * HD;
#pragma unroll
			for (int kd = 0; kd < HD; kd += 32) {
				uint af[4];
				af[0] = *(const uint *)(A + gid * LD + kd + tig * 4);
				af[1] = *(const uint *)(A + (gid + 8) * LD + kd + tig * 4);
				af[2] = *(const uint *)(A + gid * LD + kd + 16 + tig * 4);
				af[3] = *(const uint *)(A + (gid + 8) * LD + kd + 16 + tig * 4);
				for (int n = 0; n < 4; n++) {
					const unsigned char *kr = B + (long)(kb + n * 8 + gid) * HD + kd + tig * 4;
					uint bf[2] = {*(const uint *)kr, *(const uint *)(kr + 16)};
					if (pa == 0 && pb < 2) {
						mma8<false, false>(cc[n], af, bf);
					} else if (pa == 0) {
						mma8<false, true>(cc[n], af, bf);
					} else if (pb < 2) {
						mma8<true, false>(cc[n], af, bf);
					} else {
						mma8<true, true>(cc[n], af, bf);
					}
				}
			}
			for (int n = 0; n < 4; n++) {
				for (int e = 0; e < 4; e++) {
					acc[n][e] += (long)cc[n][e] << (8 * pa + 8 * pb);
				}
			}
		}
	}
	// rows gid (heads; rows gid + 8 are padding): elements 0, 1 of each n-tile, keys kb + 8n + 2 tig + e
	long mx = LMIN;
	int g = gid;
	for (int n = 0; n < 4; n++) {
		for (int e = 0; e < 2; e++) {
			int t = kb + n * 8 + tig * 2 + e;
			if (g < grp && t < T) {
				long s = rsh(acc[n][e], 42 - shs[g]);
				S[(long)(j * grp + g) * a.maxlen + t] = s;
				mx = max(mx, s);
			}
		}
	}
	mx = max(mx, shx(mx, 1));
	mx = max(mx, shx(mx, 2));
	if (tig == 0) {
		pm[w * 8 + gid] = mx;
	}
	__syncthreads();
	if (tid < (uint)grp) {
		long m = LMIN;
		for (int k = 0; k < 8; k++) {
			m = max(m, pm[k * 8 + tid]);
		}
		cmax[(j * grp + tid) * a.maxch + c] = m;
	}
}

#define DATTN1(NAME, HD)                                                                                                   \
	extern "C" __global__ void NAME(const int *mq, const int *shq, const int *Kc, long *S, long *cmax, const IntParams *pr, \
	                                AttnArgs a, const unsigned char *Kp, int Mp) {                                       \
		__shared__ __align__(16) unsigned char qp[2 * 16 * (HD + 16)];                                                     \
		__shared__ int shs[16];                                                                                            \
		__shared__ long pm[64];                                                                                            \
		dattn1<HD>(mq, shq, S, cmax, pr, a, Kp, Mp, qp, shs, pm);                                                          \
	}
DATTN1(int_dattn1_64, 64)
DATTN1(int_dattn1_128, 128)

// pass 2: per (kv head, chunk): M per head from the chunk maxima; a chunk where every head's maximum is ez or more
// below M has only zero weights (EXPA is 0 from ez on): Z and O are 0. Otherwise weights w = EXPA[min(M - s, na - 1)]
// into planes (heads x 256 keys), Z, and O = w V on tensor cores (warp: dimension tiles; k = keys).
template <int HD, bool ATOM>
static __device__ __forceinline__ void dattn2(const long *S, const long *cmax, const int *expa, long *zpart, long *opart,
                                              const IntParams *pr, const AttnArgs &a, const unsigned char *Vp, int Mp, int ez,
                                              unsigned char *wp, long *Ms, int *zs, int *live) {
	constexpr int LW = ACH + 16, NPW = HD / 64; // n-tiles (8 dimensions) per warp
	int j = blockIdx.x, c = blockIdx.y, T = pr->pos + 1, grp = a.nq / a.nkv;
	if (c * ACH >= T) {
		return;
	}
	int nch = (T + ACH - 1) / ACH;
	int tid = threadIdx.x, w = tid / 32, lane = tid % 32, gid = lane / 4, tig = lane % 4;
	if (w < grp) {
		long mx = LMIN;
		for (int i = lane; i < nch; i += 32) {
			mx = max(mx, cmax[(j * grp + w) * a.maxch + i]);
		}
		mx = max(mx, shx(mx, 16));
		mx = max(mx, shx(mx, 8));
		mx = max(mx, shx(mx, 4));
		mx = max(mx, shx(mx, 2));
		mx = max(mx, shx(mx, 1));
		if (lane == 0) {
			Ms[w] = mx;
			zs[w] = 0;
		}
	}
	if (tid == 0) {
		*live = 0;
	}
	__syncthreads();
	if (tid < (uint)grp && Ms[tid] - cmax[(j * grp + tid) * a.maxch + c] < ez) {
		*live = 1;
	}
	__syncthreads();
	if (!*live) { // every weight in the chunk is 0
		if (ATOM) {
			return; // nothing to add
		}
		if (tid < (uint)grp) {
			zpart[(j * grp + tid) * a.maxch + c] = 0;
		}
		for (int i = tid; i < grp * HD; i += 256) {
			opart[((long)(j * grp + i / HD) * a.maxch + c) * HD + i % HD] = 0;
		}
		return;
	}
	// weights: thread = key, all heads (rows 8..15 and heads past grp are zero)
	int t = c * ACH + tid;
	for (int g = 0; g < 16; g++) {
		int wv = 0;
		if (g < grp && t < T) {
			long d = Ms[g] - S[(long)(j * grp + g) * a.maxlen + t];
			if (d < ez) {
				wv = expa[d];
			}
		}
		wp[g * LW + tid] = (unsigned char)(wv & 255);
		wp[16 * LW + g * LW + tid] = (unsigned char)(wv >> 8);
		if (g < grp) {
			int z = wv;
			for (int o = 16; o > 0; o >>= 1) {
				z += __shfl_xor_sync(0xffffffffu, z, o);
			}
			if (lane == 0) {
				atomicAdd(&zs[g], z);
			}
		}
	}
	__syncthreads();
	long acc[NPW][4];
	for (int n = 0; n < NPW; n++) {
		for (int e = 0; e < 4; e++) {
			acc[n][e] = 0;
		}
	}
#pragma unroll
	for (int pa = 0; pa < 2; pa++) {
#pragma unroll
		for (int pb = 0; pb < 3; pb++) {
			int cc[NPW][4] = {{0}};
			const unsigned char *A = wp + pa * 16 * LW;
			for (int kk = 0; kk < ACH; kk += 32) {
				uint af[4];
				af[0] = *(const uint *)(A + gid * LW + kk + tig * 4);
				af[1] = *(const uint *)(A + (gid + 8) * LW + kk + tig * 4);
				af[2] = *(const uint *)(A + gid * LW + kk + 16 + tig * 4);
				af[3] = *(const uint *)(A + (gid + 8) * LW + kk + 16 + tig * 4);
				for (int n = 0; n < NPW; n++) {
					int d = (w * NPW + n) * 8 + gid;
					const unsigned char *vr = Vp + (((long)j * 3 + pb) * HD + d) * Mp + c * ACH + kk + tig * 4;
					uint bf[2] = {*(const uint *)vr, *(const uint *)(vr + 16)};
					if (pb < 2) {
						mma8<false, false>(cc[n], af, bf);
					} else {
						mma8<false, true>(cc[n], af, bf);
					}
				}
			}
			for (int n = 0; n < NPW; n++) {
				for (int e = 0; e < 4; e++) {
					acc[n][e] += (long)cc[n][e] << (8 * pa + 8 * pb);
				}
			}
		}
	}
	// rows gid = heads: elements 0, 1 of each n-tile, dimensions (w * NPW + n) * 8 + 2 tig + e
	if (gid < grp) {
		for (int n = 0; n < NPW; n++) {
			for (int e = 0; e < 2; e++) {
				int d = (w * NPW + n) * 8 + tig * 2 + e;
				if (ATOM) { // the per-head totals, int_mergea's input (exact in any order)
					atomicAdd((unsigned long long *)opart + (long)(j * grp + gid) * HD + d, (unsigned long long)acc[n][e]);
				} else {
					opart[((long)(j * grp + gid) * a.maxch + c) * HD + d] = acc[n][e];
				}
			}
		}
	}
	if (tid < (uint)grp) {
		if (ATOM) {
			atomicAdd((unsigned long long *)zpart + j * grp + tid, (unsigned long long)zs[tid]);
		} else {
			zpart[(j * grp + tid) * a.maxch + c] = zs[tid];
		}
	}
}

#define DATTN2(NAME, HD, ATOM)                                                                                                   \
	extern "C" __global__ void NAME(const long *S, const long *cmax, const int *Vc, const int *expa, long *zpart,          \
	                                long *opart, const IntParams *pr, AttnArgs a, const unsigned char *Vp, int Mp, int ez) { \
		__shared__ __align__(16) unsigned char wp[2 * 16 * (ACH + 16)];                                                    \
		__shared__ long Ms[16];                                                                                            \
		__shared__ int zs[16], live;                                                                                       \
		dattn2<HD, ATOM>(S, cmax, expa, zpart, opart, pr, a, Vp, Mp, ez, wp, Ms, zs, &live);                                     \
	}
DATTN2(int_dattn2_64, 64, false)
DATTN2(int_dattn2_128, 128, false)
DATTN2(int_dattn2a_64, 64, true) // zpart, opart: the totals Zacc[nq], Oacc[nq * hd]
DATTN2(int_dattn2a_128, 128, true)

// int_matvec8: int_matvec (same arguments and results) with 8 lanes per row, 4 rows per warp, 32 rows a threadgroup:
// for the many matrices with short rows, a whole warp per row leaves lanes idle and pays the reduction per row.
extern "C" __global__ void int_matvec8(const signed char *q, const short *s, const short *x, long *out, const int *ex,
                                       int rows, int K, int E, int einSlot, int to) {
	int lane = threadIdx.x % 32, sub = lane % 8;
	int row = blockIdx.x * 32 + (threadIdx.x / 32) * 4 + lane / 8;
	bool live = row < rows;
	int ng = K / 64;
	const signed char *qr = q + (long)min(row, rows - 1) * K;
	const short *sr = s + (long)min(row, rows - 1) * ng;
	long acc = 0;
#pragma unroll 4
	for (int col = 16 * sub; col < K; col += 128) { // 8 lanes x 16 bytes; a 64-group is 4 lanes
		int4 xa = *(const int4 *)(x + col), xb = *(const int4 *)(x + col + 8); // 16 shorts
		int4 w = *(const int4 *)(qr + col);                                       // 16 bytes
		const short *xs = (const short *)&xa, *xt = (const short *)&xb;
		const signed char *wb = (const signed char *)&w;
		int d = 0;
#pragma unroll
		for (int i = 0; i < 8; i++) {
			d += wb[i] * xs[i] + wb[8 + i] * xt[i];
		}
		d += __shfl_xor_sync(0xffffffffu, d, 1);
		d += __shfl_xor_sync(0xffffffffu, d, 2);
		if ((sub & 3) == 0) {
			acc += (long)d * sr[col / 64];
		}
	}
	acc += shx(acc, 4);
	if (live && sub == 0) {
		out[row] = rsh(acc, to - (E + ex[einSlot]));
	}
}

#include "gpu_int_step.cu"
