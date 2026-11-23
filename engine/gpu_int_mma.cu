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
template <int HD, int QR, int KT>
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
			int v = K[(long)k0 * HD + i];
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
					S[((long)(t0 + r) * a.nq + j * grp + g) * a.smax + t] = (int)s;
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

#define BSCORES_MMA(NAME, HD, QR, KT)                                                                                      \
	extern "C" __global__ void NAME(const int *mq, const int *shq, const int *Kc, int *S, long *cmax, int *ovf,            \
	                                const IntParams *pr, BAttnArgs a) {                                                  \
		__shared__ __align__(16) unsigned char qp[2 * QR * (HD + 16)];                                                     \
		__shared__ __align__(16) unsigned char kp[3 * KT * (HD + 16)];                                                     \
		__shared__ int shs[QR];                                                                                            \
		__shared__ long pmx[(8 / (QR / 16)) * QR];                                                                         \
		bscores_mma<HD, QR, KT>(mq, shq, Kc, S, cmax, ovf, pr->pos, pr->tok, a,                                            \
		                        make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), threadIdx.x, qp, kp, shs, pmx);            \
	}
BSCORES_MMA(int_bscores_mma64, 64, 128, 64)
BSCORES_MMA(int_bscores_mma128, 128, 64, 32)

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
