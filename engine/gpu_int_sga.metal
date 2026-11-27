// Batched attention through simdgroup matrices (Apple GPU family 7 and later). Appended to gpu_int.metal (it uses
// its helpers, BAttnArgs, SIDX and KC); each kernel takes the arguments and grid of the gpu_int.metal kernel it
// replaces and produces the same integers.
//
// Exactness. Every simdgroup multiply-accumulate here has integer inputs that half holds exactly, so every product
// is an integer, and each accumulator starts at 0 and takes few enough terms that the sum of their magnitudes is at
// most 2^24. Every partial sum, in whatever order or grouping the hardware adds, is then an integer of magnitude
// <= 2^24, which float32 holds exactly: the accumulator ends as the exact integer sum. It is converted to an int and
// combined in integers.

// ---- scores: int_bscores (q . m for every query row and key, then the score shift) ----
//
// q: QQ = 12-bit (|q| <= 2047). A key mantissa m (KQ = 16 bits, |m| <= 32767) is two balanced byte planes:
// ml = ((m + 128) & 255) - 128 in [-128, 127], mh = (m - ml) / 256 in [-128, 128]. Over 64 dimensions each plane's
// sum of |q| |plane| is at most 2047 * 128 * 64 = 16,769,024 <= 2^24, so a float accumulator per plane per 64
// dimensions is exact. q . m = 256 (q . mh) + (q . ml); over 128 dimensions each plane's sum is below 2^25, an int.
//
// A threadgroup per (kv head, block of rows, key chunk), as int_bscores: QR query rows (grp heads x br rows, row
// qr = g * br + r) against the chunk's keys in tiles of KT. The 8 simdgroups split a tile as (QR / SR) x (KT / SK):
// simdgroup sg holds SR rows x SK keys as 8 x 8 accumulator tiles, two planes each.
#ifndef SCABL
#define SCABL 0 // timing experiments: 1 no multiply-accumulates, 2 no score stores
#endif
#define SGQS(HD) ((HD) + 8) // threadgroup row stride (halves) of the queries
#define SGKS 72             // ... of a 64-dimension slice of the key planes

template <int HD, int QR, int KT, int SR, int SK>
static inline void bscores_sg(device const int *mq, device const int *shq, device const short *Km, device int *S,
                              device long *cmax, device atomic_int *ovf, int base, int R, constant BAttnArgs &a, uint3 tg3,
                              uint tid, uint sg, uint lane, threadgroup half *qs, threadgroup half *kh,
                              threadgroup half *kl, threadgroup int *shs, threadgroup int *kss, threadgroup long *red) {
	constexpr int QS = SGQS(HD), NI = SR / 8, NJ = SK / 8, KG = KT / SK;
	int j = tg3.x, grp = a.nq / a.nkv, br = QR / grp;
	int t0 = tg3.y * br, c = tg3.z;
	int pmax = base + min(t0 + br, R) - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	if (!cvis(k0c, k0c + KC, base + t0 + 1, a.win, a.sink)) {
		return;
	}
	for (int i = tid; i < QR * HD; i += 256) {
		int qr = i / HD, d = i % HD, g = qr / br, r = qr % br;
		qs[qr * QS + d] = g < grp && t0 + r < R ? (half)mq[((long)(t0 + r) * a.nq + j * grp + g) * HD + d] : (half)0;
	}
	if (tid < QR) {
		int g = tid / br, r = tid % br;
		shs[tid] = g < grp && t0 + r < R ? shq[(t0 + r) * a.nq + j * grp + g] : 0;
	}
	device const short *K = Km + (long)j * a.maxlen * HD;
	device const int *Ksh = KSHR(Km, a.nkv, a.maxlen, HD) + (long)j * a.maxlen;
	int sr = (sg / KG) * SR, sk = (sg % KG) * SK;
	int qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
	long mx[NI];
	for (int i = 0; i < NI; i++) {
		mx[i] = LMIN;
	}
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		int ih[NI][NJ][2], il[NI][NJ][2];
		for (int i = 0; i < NI; i++) {
			for (int jj = 0; jj < NJ; jj++) {
				ih[i][jj][0] = ih[i][jj][1] = il[i][jj][0] = il[i][jj][1] = 0;
			}
		}
		for (int d0 = 0; d0 < HD; d0 += 64) {
			threadgroup_barrier(mem_flags::mem_threadgroup);
			for (int i = tid; i < KT * 16; i += 256) { // KT keys x 64 dimensions, 4 at a time
				int kk = i / 16, d = (i % 16) * 4;
				int t = min(k0 + kk, a.maxlen - 1);
				int4 m = int4(*(device const short4 *)(K + (long)t * HD + d0 + d));
				int4 ml = ((m + 128) & 255) - 128;
				*(threadgroup half4 *)(kh + kk * SGKS + d) = half4((m - ml) >> 8);
				*(threadgroup half4 *)(kl + kk * SGKS + d) = half4(ml);
			}
			if (d0 == 0 && tid < (uint)KT) {
				kss[tid] = Ksh[min(k0 + (int)tid, a.maxlen - 1)];
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			simdgroup_matrix<float, 8, 8> ch[NI][NJ], cl[NI][NJ];
			for (int i = 0; i < NI; i++) {
				for (int jj = 0; jj < NJ; jj++) {
					ch[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
					cl[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
				}
			}
			for (int k = 0; k < 64; k += 8) {
				simdgroup_matrix<half, 8, 8> bh[NJ], bl[NJ];
				for (int jj = 0; jj < NJ; jj++) { // keys transposed: (dimension, key)
					simdgroup_load(bh[jj], kh + (sk + 8 * jj) * SGKS + k, SGKS, ulong2(0, 0), true);
					simdgroup_load(bl[jj], kl + (sk + 8 * jj) * SGKS + k, SGKS, ulong2(0, 0), true);
				}
				for (int i = 0; i < NI; i++) {
					simdgroup_matrix<half, 8, 8> aq;
					simdgroup_load(aq, qs + (sr + 8 * i) * QS + d0 + k, QS);
					for (int jj = 0; jj < NJ; jj++) {
#if SCABL != 1
						simdgroup_multiply_accumulate(ch[i][jj], aq, bh[jj], ch[i][jj]);
						simdgroup_multiply_accumulate(cl[i][jj], aq, bl[jj], cl[i][jj]);
#endif
					}
				}
			}
			for (int i = 0; i < NI; i++) {
				for (int jj = 0; jj < NJ; jj++) {
					for (int e = 0; e < 2; e++) {
						ih[i][jj][e] += (int)ch[i][jj].thread_elements()[e];
						il[i][jj][e] += (int)cl[i][jj].thread_elements()[e];
					}
				}
			}
		}
		for (int i = 0; i < NI; i++) {
			int qr = sr + 8 * i + fm, g = qr / br, r = qr % br;
			if (g >= grp || t0 + r >= R) {
				continue;
			}
			int p = base + t0 + r;
			for (int jj = 0; jj < NJ; jj++) {
				for (int e = 0; e < 2; e++) {
					int kk = sk + 8 * jj + fn + e, t = k0 + kk;
					if (t <= p && kvis(t, p + 1, a.win, a.sink)) {
						long s = rsh(((long)ih[i][jj][e] << 8) + il[i][jj][e], 42 - shs[qr] - kss[kk]);
						if (s != (long)(int)s) {
							atomic_store_explicit(ovf, 1, memory_order_relaxed);
						}
#if SCABL != 2
						S[SIDX(t0 + r, j * grp + g, t, a)] = (int)s;
#endif
						mx[i] = max(mx[i], s);
					}
				}
			}
		}
	}
	// the chunk's maxima: lanes holding the same row (lane bits 0 and 3 pick the columns), then the KG simdgroups
	for (int i = 0; i < NI; i++) {
		long m = mx[i];
		m = max(m, shx(m, 1));
		m = max(m, shx(m, 8));
		if ((lane & 9) == 0) {
			red[(sr + 8 * i + fm) * KG + sg % KG] = m;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < (uint)QR) {
		int g = tid / br, r = tid % br;
		if (g < grp && t0 + r < R) {
			long m = LMIN;
			for (int k = 0; k < KG; k++) {
				m = max(m, red[tid * KG + k]);
			}
			cmax[((long)(t0 + r) * a.nq + j * grp + g) * a.nchmax + c] = m;
		}
	}
}

#define BSCORES_SG(NAME, HD, QR, KT, SR, SK)                                                                              \
	kernel void NAME(device const int *mq [[buffer(0)]], device const int *shq [[buffer(1)]],                              \
	                 device const short *Km [[buffer(2)]], device int *S [[buffer(3)]], device long *cmax [[buffer(4)]],  \
	                 device atomic_int *ovf [[buffer(5)]], device const IntParams *pr [[buffer(6)]],                     \
	                 constant BAttnArgs &a [[buffer(7)]], uint3 tg3 [[threadgroup_position_in_grid]],                    \
	                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],              \
	                 uint lane [[thread_index_in_simdgroup]]) {                                                           \
		threadgroup half qs[QR * SGQS(HD)];                                                                                \
		threadgroup half kh[KT * SGKS];                                                                                    \
		threadgroup half kl[KT * SGKS];                                                                                    \
		threadgroup int shs[QR];                                                                                           \
		threadgroup int kss[KT];                                                                                           \
		threadgroup long red[QR * (KT / SK)];                                                                              \
		bscores_sg<HD, QR, KT, SR, SK>(mq, shq, Km, S, cmax, ovf, pr->pos, pr->tok, a, tg3, tid, sg, lane, qs, kh, kl, shs, \
		                               kss, red);                                                                          \
	}
// the same QR as int_bscores64 / int_bscores128 (so the same grid)
BSCORES_SG(int_bscores_sg64, 64, 128, 32, 32, 16)
BSCORES_SG(int_bscores_sg128, 128, 64, 32, 16, 16)

// ---- weights x V: int_bpv (O = sum over keys of w V, and Z = sum of w, per row and chunk) ----
//
// A cached value is v = vm * 2^sv with |vm| <= 2^(VQ-1) - 1 = 2047 and one shift sv per key vector, so |v| < 2^22.
// The kernel doesn't need the stored sv: per key it takes s = max(0, bitlen(max_d |v_d|) - 11) <= sv, so every v_d is
// a multiple of 2^s and |v_d >> s| <= 2047. With smin the least s among a tile's keys that carry a weight,
// w v = 2^smin (w 2^(s - smin)) (v >> s): the shift moves into the weight, W = w 2^(s - smin) <= 2^15 * 2^11 = 2^26
// (w <= EXPA[0] = 2^15). W is split into balanced base-1024 digits p0 + 1024 p1 + 2^20 p2, each |p| <= 512 (two
// digits suffice when W <= 2^19, i.e. the tile's shifts span at most 4: nearly every tile on real text). Over 16 keys
// each digit's sum of |p| |vm| is at most 16 * 512 * 2047 = 16,769,024 <= 2^24: exact; a 32-key tile's two halves
// add as ints. O += (P0 + 1024 P1 + 2^20 P2) << smin, in 64 bits.
//
// A threadgroup per (query head, block of BRV rows, key chunk), as int_bpv. Weights exactly as int_bpv computes them
// (EXPA's head in threadgroup memory, the rest from the device; rows past the batch's end skipped); tiles whose
// weights are all 0 skipped, and 8 x 8 blocks of zero weights skip their multiply-accumulates. The 8 simdgroups split
// the BRV x HD outputs as (BRV / 32) x (HD / (BRV / 4)): 32 rows x 16 dimensions each (4 x 2 accumulator tiles).
#ifndef PVABL
#define PVABL 0 // timing experiments: 1 no multiply-accumulates, 2 no score loads, 3 no value loads
#endif
#define PVKT 32           // keys per tile
#define PVWS (PVKT + 8)   // threadgroup row stride (halves) of the weight digits
#define PVVS(HD) ((HD) + 8) // ... of the value tile

template <int HD, int BRV, int SR, int SDD, int ETS, int KH, bool FUSE>
static inline void bpv_sg(device const int *S, device const long *cmax, device const int *Vc, device const int *expa,
                          device long *zp, device long *op, int base, int R, constant BAttnArgs &a, uint3 tg3, uint tid,
                          uint sg, uint lane, threadgroup half *vm, threadgroup half *wp, threadgroup int *ks,
                          threadgroup uint *msk, threadgroup long *Ms, threadgroup ushort *et) {
	constexpr int KT = PVKT, VS = PVVS(HD), NRW = BRV / 8, DGS = HD / SDD, NI = SR / 8, NJ = SDD / 8, TPK = 256 / KT,
	              DPT = HD / TPK;
	static_assert((BRV / SR) * DGS == 8, "8 simdgroups of SR rows x SDD dimensions");
	int h = tg3.x, t0 = tg3.y * BRV, c = tg3.z, grp = a.nq / a.nkv, j = h / grp;
	int pmax = base + t0 + BRV - 1;
	int k0c = c * KC, k1c = min(k0c + KC, pmax + 1);
	int ez = a.ez;
	if (!cvis(k0c, k0c + KC, base + t0 + 1, a.win, a.sink)) {
		return;
	}
	int ezs = min(ez, ETS);
	for (int i = tid; i < ezs; i += 256) {
		et[i] = (ushort)expa[i];
	}
	if (tid < (uint)BRV) {
		int p = base + t0 + tid, nch = p / KC + 1;
		device const long *cm = cmax + ((long)(t0 + tid) * a.nq + h) * a.nchmax;
		long m = LMIN;
		for (int i = 0; i < nch; i++) {
			if (cvis(i * KC, i * KC + KC, p + 1, a.win, a.sink)) {
				m = max(m, cm[i]);
			}
		}
		Ms[tid] = m;
	}
	device const int *V = Vc + (long)j * a.maxlen * HD;
	int sr = (sg / DGS) * SR, sd = (sg % DGS) * SDD;
	int qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
	long od[NI][NJ][2];
	for (int i = 0; i < NI; i++) {
		for (int jj = 0; jj < NJ; jj++) {
			od[i][jj][0] = od[i][jj][1] = 0;
		}
	}
	int zl[NRW]; // row sg + 8i's share of Z (this lane's key), <= (KC / KT) * 2^15
	for (int i = 0; i < NRW; i++) {
		zl[i] = 0;
	}
	int vk = tid / TPK, vd = (tid % TPK) * DPT; // the value tile: TPK lanes per key, DPT dimensions each
	for (int k0 = k0c; k0 < k1c; k0 += KT) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		// weights: simdgroup sg does rows sg + 8i, lane = key
		int w[NRW];
		int t = k0 + (int)lane;
		uint any = 0;
		// SIDX in 32 bits (the score buffer is at most 2^28 ints); M - s in 32 bits: a used score and its row's
		// maximum are both int32 (a batch with a score outside int32 is redone by int_attnb*), M >= s
		device const int *St = S + ((uint)(t >> 4) * (uint)a.nq + (uint)h) * (uint)a.rows * 16u + (uint)(t & 15);
		for (int i = 0; i < NRW; i++) {
			int r = sg + 8 * i, p = base + t0 + r;
			int ww = 0;
			if (t <= p && t0 + r < R && kvis(t, p + 1, a.win, a.sink)) {
#if PVABL == 2
				uint d = (uint)(r + t) & 1023; // ablation: no score loads
#else
				uint d = (uint)(int)Ms[r] - (uint)St[(uint)(t0 + r) * 16u]; // unsigned: wraps, no overflow
#endif
				if (d < (uint)ezs) {
					ww = et[d];
				} else if (d < (uint)ez) {
					ww = expa[d];
				}
			}
			w[i] = ww;
			zl[i] += ww;
			uint b = (uint)(simd_vote::vote_t)simd_ballot(ww != 0); // lane = key
			any |= b;
			if (lane == 0) {
				msk[i * 8 + sg] = b; // row tile i, row sg within it
			}
		}
		if (lane == 0) {
			msk[NRW * 8 + sg] = any;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		uint keys = 0; // keys with a weight anywhere in the block
		for (int s = 0; s < 8; s++) {
			keys |= msk[NRW * 8 + s];
		}
		if (keys == 0) {
			continue;
		}
		// the value tile: each key's shift s, then v >> s as half
		{
			int tv = min(k0 + vk, a.maxlen - 1);
			device const int4 *src = (device const int4 *)(V + (long)tv * HD + vd);
			int4 vv[DPT / 4];
			uint orv = 0;
			for (int q = 0; q < DPT / 4; q++) {
#if PVABL == 3
				vv[q] = int4(vk + q, vd, 7, -9) << 3; // ablation: no value loads
#else
				vv[q] = src[q];
#endif
				uint4 av = uint4(abs(vv[q]));
				orv |= av.x | av.y | av.z | av.w;
			}
			for (ushort x = 1; x < TPK; x <<= 1) {
				orv |= simd_shuffle_xor(orv, x);
			}
			int s = max(0, (orv == 0 ? 0 : 32 - (int)clz(orv)) - (VQ - 1));
			for (int q = 0; q < DPT / 4; q++) {
				*(threadgroup half4 *)(vm + vk * VS + vd + 4 * q) = half4(vv[q] >> s);
			}
			if (vd == 0) {
				ks[vk] = s;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		int sl = ks[lane];
		bool kw = (keys >> lane) & 1;
		int smin = simd_min(kw ? sl : 99), smax = simd_max(kw ? sl : -1);
		// digits: base 2^DB with |p| <= 2^(DB-1); KH keys per float accumulator (KH * 2^(DB-1) * 2047 <= 2^24)
		constexpr int DB = KH == 16 ? 10 : 9, DH = 1 << (DB - 1), DM = (1 << DB) - 1;
		int nd = smax - smin <= DB * 2 - 16 ? 2 : 3;
		int p2[NRW];
		for (int i = 0; i < NRW; i++) {
			int W = w[i] == 0 ? 0 : w[i] << (sl - smin);
			int q0 = ((W + DH) & DM) - DH, W1 = (W - q0) >> DB;
			int q1 = ((W1 + DH) & DM) - DH;
			p2[i] = (W1 - q1) >> DB;
			if (nd == 2) {
				q1 = W1; // W <= 2^(2 DB - 1): |W1| <= 2^(DB-1)
			}
			int r = sg + 8 * i;
			wp[r * PVWS + lane] = (half)q0;
			wp[BRV * PVWS + r * PVWS + lane] = (half)q1;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		// 8 x 8 blocks of weights that are all 0: (row tile, key group) -> skip
		uint live[NI];
		for (int i = 0; i < NI; i++) {
			uint m = 0;
			for (int s = 0; s < 8; s++) {
				m |= msk[(sr / 8 + i) * 8 + s];
			}
			live[i] = m;
		}
		if (FUSE) {
			// both low digits in one pass over the keys (each value fragment loaded once), no zero-block skipping;
			// a third digit, when the shifts need one, in a second pass
			static_assert(!FUSE || KH == PVKT, "fused digits take whole tiles");
			simdgroup_matrix<float, 8, 8> c0[NI][NJ], c1[NI][NJ];
			for (int i = 0; i < NI; i++) {
				for (int jj = 0; jj < NJ; jj++) {
					c0[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
					c1[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
				}
			}
			for (int k = 0; k < KT; k += 8) {
				simdgroup_matrix<half, 8, 8> b[NJ];
				for (int jj = 0; jj < NJ; jj++) {
					simdgroup_load(b[jj], vm + k * VS + sd + 8 * jj, VS);
				}
				for (int i = 0; i < NI; i++) {
					simdgroup_matrix<half, 8, 8> a0, a1;
					simdgroup_load(a0, wp + (sr + 8 * i) * PVWS + k, PVWS);
					simdgroup_load(a1, wp + BRV * PVWS + (sr + 8 * i) * PVWS + k, PVWS);
					for (int jj = 0; jj < NJ; jj++) {
						simdgroup_multiply_accumulate(c0[i][jj], a0, b[jj], c0[i][jj]);
						simdgroup_multiply_accumulate(c1[i][jj], a1, b[jj], c1[i][jj]);
					}
				}
			}
			for (int i = 0; i < NI; i++) {
				for (int jj = 0; jj < NJ; jj++) {
					for (int e = 0; e < 2; e++) {
						od[i][jj][e] += ((long)(int)c0[i][jj].thread_elements()[e] << smin) +
						                ((long)(int)c1[i][jj].thread_elements()[e] << (DB + smin));
					}
				}
			}
			if (nd == 3) {
				threadgroup_barrier(mem_flags::mem_threadgroup);
				for (int i = 0; i < NRW; i++) {
					wp[(sg + 8 * i) * PVWS + lane] = (half)p2[i];
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
				for (int i = 0; i < NI; i++) {
					for (int jj = 0; jj < NJ; jj++) {
						c0[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
					}
				}
				for (int k = 0; k < KT; k += 8) {
					simdgroup_matrix<half, 8, 8> b[NJ];
					for (int jj = 0; jj < NJ; jj++) {
						simdgroup_load(b[jj], vm + k * VS + sd + 8 * jj, VS);
					}
					for (int i = 0; i < NI; i++) {
						simdgroup_matrix<half, 8, 8> a0;
						simdgroup_load(a0, wp + (sr + 8 * i) * PVWS + k, PVWS);
						for (int jj = 0; jj < NJ; jj++) {
							simdgroup_multiply_accumulate(c0[i][jj], a0, b[jj], c0[i][jj]);
						}
					}
				}
				for (int i = 0; i < NI; i++) {
					for (int jj = 0; jj < NJ; jj++) {
						for (int e = 0; e < 2; e++) {
							od[i][jj][e] += (long)(int)c0[i][jj].thread_elements()[e] << (2 * DB + smin);
						}
					}
				}
			}
			continue;
		}
		for (int dg = 0; dg < nd; dg++) {
			if (dg == 2) { // the third digit replaces the first
				threadgroup_barrier(mem_flags::mem_threadgroup);
				for (int i = 0; i < NRW; i++) {
					wp[(sg + 8 * i) * PVWS + lane] = (half)p2[i];
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}
			threadgroup const half *wd = wp + (dg == 1 ? BRV * PVWS : 0);
			int ti[NI][NJ][2]; // the digit's tile sum: two 16-key halves, each exact in float
			for (int k16 = 0; k16 < KT; k16 += KH) {
				simdgroup_matrix<float, 8, 8> cc[NI][NJ];
				for (int i = 0; i < NI; i++) {
					for (int jj = 0; jj < NJ; jj++) {
						cc[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
					}
				}
				for (int k = k16; k < k16 + KH; k += 8) {
					simdgroup_matrix<half, 8, 8> b[NJ];
					for (int jj = 0; jj < NJ; jj++) {
						simdgroup_load(b[jj], vm + k * VS + sd + 8 * jj, VS);
					}
					for (int i = 0; i < NI; i++) {
						if (((live[i] >> k) & 0xff) == 0) {
							continue;
						}
						simdgroup_matrix<half, 8, 8> aw;
						simdgroup_load(aw, wd + (sr + 8 * i) * PVWS + k, PVWS);
#if PVABL != 1
						for (int jj = 0; jj < NJ; jj++) {
							simdgroup_multiply_accumulate(cc[i][jj], aw, b[jj], cc[i][jj]);
						}
#endif
					}
				}
				for (int i = 0; i < NI; i++) {
					for (int jj = 0; jj < NJ; jj++) {
						for (int e = 0; e < 2; e++) {
							int v = (int)cc[i][jj].thread_elements()[e];
							ti[i][jj][e] = k16 == 0 ? v : ti[i][jj][e] + v;
						}
					}
				}
			}
			int sh = DB * dg + smin;
			for (int i = 0; i < NI; i++) {
				for (int jj = 0; jj < NJ; jj++) {
					for (int e = 0; e < 2; e++) {
						od[i][jj][e] += (long)ti[i][jj][e] << sh;
					}
				}
			}
		}
	}
	for (int i = 0; i < NI; i++) {
		int r = sr + 8 * i + fm;
		long pi = ((long)(t0 + r) * a.nq + h) * a.nchmax + c;
		for (int jj = 0; jj < NJ; jj++) {
			for (int e = 0; e < 2; e++) {
				op[pi * HD + sd + 8 * jj + fn + e] = od[i][jj][e];
			}
		}
	}
	for (int i = 0; i < NRW; i++) {
		int z = simd_sum(zl[i]);
		if (lane == 0) {
			zp[((long)(t0 + sg + 8 * i) * a.nq + h) * a.nchmax + c] = z;
		}
	}
}

#define BPV_SG(NAME, HD, BRV, SR, SDD, ETS, KH, FUSE)                                                                                       \
	kernel void NAME(device const int *S [[buffer(0)]], device const long *cmax [[buffer(1)]],                            \
	                 device const int *Vc [[buffer(2)]], device const int *expa [[buffer(3)]], device long *zp [[buffer(4)]], \
	                 device long *op [[buffer(5)]], device const IntParams *pr [[buffer(6)]],                            \
	                 constant BAttnArgs &a [[buffer(7)]], uint3 tg3 [[threadgroup_position_in_grid]],                    \
	                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],              \
	                 uint lane [[thread_index_in_simdgroup]]) {                                                           \
		threadgroup half vm[PVKT * PVVS(HD)];                                                                              \
		threadgroup half wp[2 * BRV * PVWS];                                                                               \
		threadgroup int ks[PVKT];                                                                                          \
		threadgroup uint msk[(BRV / 8 + 1) * 8];                                                                           \
		threadgroup long Ms[BRV];                                                                                          \
		threadgroup ushort et[ETS];                                                                                        \
		bpv_sg<HD, BRV, SR, SDD, ETS, KH, FUSE>(S, cmax, Vc, expa, zp, op, pr->pos, pr->tok, a, tg3, tid, sg, lane, vm, wp, ks, msk, Ms, et);  \
	}
// the same BRV as int_bpv64 / int_bpv128 (so the same grid)
// head 64: 32-row blocks, simdgroups 16 rows x 16 dimensions; head 128: 16-row blocks, 16 x 16 (fewer registers
// than 32-row blocks, which matters more than the value tile's reuse)
BPV_SG(int_bpv_sg64, 64, 32, 16, 16, 7168, 32, true)
BPV_SG(int_bpv_sg128, 128, 16, 16, 16, 7168, 32, true)
#ifdef SGA_EXPERIMENTS // tests only: the unfused digit loop, for comparison
BPV_SG(int_bpv_sg64_nf, 64, 32, 16, 16, 7168, 32, false)
BPV_SG(int_bpv_sg128_nf, 128, 32, 32, 16, 7168, 32, false)
#endif
