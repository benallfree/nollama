// int_gemm through simdgroup matrices (Apple GPU family 7 and later: every Apple silicon Mac). Same arguments, grid
// (64 outputs x 64 rows per threadgroup, 256 threads) and integers as gpu_int.metal's int_gemm, which it replaces when
// it compiles.
//
// Exactness: weights |q| <= 127 and activations |x| <= 2^(AQ-1) - 1 are integers that half holds exactly. With
// AQ <= 12, every product and every partial sum of a 64-term group is an integer below 127 * 2047 * 64 = 16,637,888 <
// 2^24, which float32 holds exactly, so the simdgroup multiply-accumulates give the exact integer group sum whatever
// order or grouping the hardware uses. The group sum is then scaled in integers exactly as int_gemm does.
#include <metal_stdlib>
using namespace metal;

// AQ comes from the Go side (intActBits)
#if AQ > 12
#error "int_gemm_sg needs activations of at most 12 bits"
#endif

static inline long rsh(long x, int n) { return n <= 0 ? (x << (-n)) : ((x + (1L << (n - 1))) >> n); }

#define SBO 64 // outputs per threadgroup
#define SBT 64 // rows per threadgroup (intGemmRows)
#define SKP 72 // threadgroup row stride in halves (padded)

// 8 simdgroups as 2 (rows) x 4 (outputs); each holds 32 rows x 16 outputs as 4 x 2 accumulator tiles of 8 x 8.
kernel void int_gemm_sg(device const char *q [[buffer(0)]], device const short *s [[buffer(1)]],
                        device const short *x [[buffer(2)]], device long *out [[buffer(3)]], device const int *ex [[buffer(4)]],
                        constant int &O [[buffer(5)]], constant int &K [[buffer(6)]], constant int &E [[buffer(7)]],
                        constant int &einSlot [[buffer(8)]], constant int &to [[buffer(9)]], constant int4 &st [[buffer(10)]],
                        uint2 tg2 [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
	threadgroup half W[SBO * SKP];
	threadgroup half X[SBT * SKP];
	threadgroup short S[SBO];
	int xS = st.y, outS = st.z;
	int o0 = tg2.x * SBO, t0 = tg2.y * SBT;
	int sr = (sg / 4) * 32, so = (sg % 4) * 16; // this simdgroup's rows and outputs within the tile
	// an 8 x 8 simdgroup matrix's elements: this lane holds (fm, fn) and (fm, fn + 1)
	int qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
	long acc[4][2][2];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++) {
			acc[i][j][0] = acc[i][j][1] = 0;
		}
	}
	int ng = K / 64;
	int wr = tid >> 2, wc = (tid & 3) * 16; // each thread copies 16 weights and 16 activations per group
	device const char *wsrc = q + (long)min(o0 + wr, O - 1) * K + wc;
	device const short *xsrc = x + (long)(t0 + wr) * xS + wc;
	for (int g = 0; g < ng; g++) {
		{
			device const char4 *src = (device const char4 *)(wsrc + g * 64);
			threadgroup half4 *dst = (threadgroup half4 *)(W + wr * SKP + wc);
			device const short4 *xs = (device const short4 *)(xsrc + g * 64);
			threadgroup half4 *xd = (threadgroup half4 *)(X + wr * SKP + wc);
			for (int i = 0; i < 4; i++) {
				dst[i] = half4(src[i]);
				xd[i] = half4(xs[i]);
			}
		}
		if (tid < SBO) {
			S[tid] = s[(long)min(o0 + (int)tid, O - 1) * ng + g];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		simdgroup_matrix<float, 8, 8> c[4][2];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 2; j++) {
				c[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
			}
		}
		for (int k = 0; k < 64; k += 8) {
			simdgroup_matrix<half, 8, 8> b[2];
			for (int j = 0; j < 2; j++) { // weights transposed: (k, output)
				simdgroup_load(b[j], W + (so + 8 * j) * SKP + k, SKP, ulong2(0, 0), true);
			}
			for (int i = 0; i < 4; i++) {
				simdgroup_matrix<half, 8, 8> a;
				simdgroup_load(a, X + (sr + 8 * i) * SKP + k, SKP);
				for (int j = 0; j < 2; j++) {
					simdgroup_multiply_accumulate(c[i][j], a, b[j], c[i][j]);
				}
			}
		}
		for (int j = 0; j < 2; j++) {
			int sc0 = S[so + 8 * j + fn], sc1 = S[so + 8 * j + fn + 1];
			for (int i = 0; i < 4; i++) {
				for (int e = 0; e < 2; e++) {
					int d = (int)c[i][j].thread_elements()[e];
					int sc = e == 0 ? sc0 : sc1;
					acc[i][j][e] += ((long)((d >> 16) * sc) << 16) + (long)((d & 0xffff) * sc);
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	for (int i = 0; i < 4; i++) {
		int t = t0 + sr + 8 * i + fm;
		int sh = to - (E + ex[4 * t + einSlot]);
		for (int j = 0; j < 2; j++) {
			for (int e = 0; e < 2; e++) {
				int o = o0 + so + 8 * j + fn + e;
				if (o < O) {
					out[(long)t * outS + o] = rsh(acc[i][j][e], sh);
				}
			}
		}
	}
}
