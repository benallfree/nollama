// nollama's int8 tensor-core prompt kernels (mma.sync m16n8k32 s8, compute 8.0+), compiled to their own PTX module
// (gpu_cuda_mma.ptx), loaded only on GPUs that have them; older GPUs keep gpu.cu's half-precision kernels.
//
// Q4_K prompt matmuls quantize the activations once per input (quant_act_mma: 32-element blocks of int8 with a float
// scale, as the CPU's 8-bit path does) and multiply them with the weights' nibbles as 0..15, their sub-block scales
// and mins applied after each block's integer sums. (Q8_0 weights measured no faster this way than on the half
// kernels on an A16, so they stay there.)
#include <cuda_fp16.h>
#include <stdint.h>

typedef unsigned char uchar;
typedef unsigned int uint;

struct MatmulArgs {
	uint rows, nb, rowBytes, cols, tokens;
};

__device__ __forceinline__ float f16(const uchar *p) { return __half2float(__ushort_as_half(*(const unsigned short *)p)); }

// scale_min_k4x2 is gpu.cu's: Q4_K sub-blocks k and k+1 (k even) from a block's first 16 bytes h.
__device__ __forceinline__ float4 scale_min_k4x2(uint k, uint4 h) {
	uint sh = 8 * (k % 4), y = h.y >> sh, z = h.z >> sh, sc, mn;
	if (k < 4) {
		sc = y & 0x3f3f, mn = z & 0x3f3f;
	} else {
		uint w = h.w >> sh;
		sc = (w & 0x0f0f) | ((y >> 2) & 0x3030), mn = ((w >> 4) & 0x0f0f) | ((z >> 2) & 0x3030);
	}
	return make_float4(sc & 0xff, (sc >> 8) & 0xff, mn & 0xff, (mn >> 8) & 0xff);
}

// quant_act_mma quantizes x ([tokens][cols] floats, nblk = tokens*cols/32 blocks) to q (int8, same layout) and s
// (float2 per block: the scale d = max|x|/127 and d * the sum of the block's quants). Eight lanes per block, four
// elements each. Grid (nblk/32), 256 threads.
extern "C" __global__ void quant_act_mma(const float *x, signed char *q, float2 *s, uint nblk) {
	uint blk0 = blockIdx.x * 32 + threadIdx.x / 8, l = threadIdx.x % 8;
	uint blk = min(blk0, nblk - 1); // past the end: redo the last block (all lanes stay in the shuffles), don't store
	float4 v = *(const float4 *)(x + (size_t)blk * 32 + 4 * l);
	float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
	for (int o = 1; o < 8; o *= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o, 8));
	float d = m / 127, id = m > 0 ? 127 / m : 0;
	int a = __float2int_rn(v.x * id), b = __float2int_rn(v.y * id), c = __float2int_rn(v.z * id), e = __float2int_rn(v.w * id);
	if (blk0 < nblk) *(char4 *)(q + (size_t)blk * 32 + 4 * l) = make_char4(a, b, c, e);
	int sum = a + b + c + e;
#pragma unroll
	for (int o = 1; o < 8; o *= 2) sum += __shfl_xor_sync(0xffffffff, sum, o, 8);
	if (l == 0 && blk0 < nblk) s[blk] = make_float2(d, d * sum);
}

// matmul_mmq: Y[t][r] = W row r . X row t from int8 quants (Q4_K weights). A block computes 128 rows x 64 tokens with 8 warps; warp
// w owns rows 32*(w%4).. and tokens 32*(w/4)..: 2x4 m16n8k32 tiles per 32-element block. Each K step of 128 stages
// the weights' quants [128][128] and the activations' [64][128] in shared memory (rows padded to 144 bytes, so a
// fragment's 8 rows x 4 words fall in 32 different banks), with per-(row, block) (scale, min) and per-(token, block)
// (d, d*sum) pairs; after each block's integer MMAs, acc += sum * scale * d - min * d*sum. a.cols % 128 == 0.
// Each K step of 128 is half a Q4_K block: thread pairs take a row's two 64-element chunks.
// Grid (rows/128, tokens/64), 256 threads.
#define MQ_K 128
#define MQ_LD (MQ_K + 16)
__device__ void matmul_mmq(const uchar *w, const signed char *xq, const float2 *xs, float *y, MatmulArgs a) {
	__shared__ __align__(16) signed char sa[128 * MQ_LD], sb[64 * MQ_LD];
	__shared__ float2 sad[128][4], sbd[64][4];
	uint tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q4 = lane % 4;
	uint r0 = blockIdx.x * 128, t0 = blockIdx.y * 64;
	uint wr = (warp % 4) * 32, wt = (warp / 4) * 32;
	float acc[2][4][4];
#pragma unroll
	for (int i = 0; i < 2; i++)
#pragma unroll
		for (int j = 0; j < 4; j++)
#pragma unroll
			for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
	uint xnb = a.cols / 32;
	for (uint k0 = 0; k0 < a.cols; k0 += MQ_K) {
		{ // weights: 128 rows x 2 chunks of 64 (sub-blocks 2c, 2c+1 share their quant bytes), one per thread
			uint l = tid / 2, cc = tid % 2, c = ((k0 / 128) % 2) * 2 + cc, r = r0 + l;
			int *lo = (int *)(sa + l * MQ_LD + cc * 64), *hi = lo + 8;
			if (r < a.rows) {
				const uchar *blk = w + (size_t)r * a.rowBytes + (size_t)(k0 / 256) * 144;
				uint4 hd = *(const uint4 *)blk;
				float4 sm = scale_min_k4x2(2 * c, hd);
				float2 dm = __half22float2(*(const __half2 *)&hd.x);
				const uint4 *qs = (const uint4 *)(blk + 16 + 32 * c); // 16-byte aligned
				uint4 u0 = qs[0], u1 = qs[1];
				uint v[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
				for (int i = 0; i < 8; i++) {
					lo[i] = v[i] & 0x0F0F0F0F;
					hi[i] = (v[i] >> 4) & 0x0F0F0F0F;
				}
				sad[l][2 * cc] = make_float2(dm.x * sm.x, dm.y * sm.z);
				sad[l][2 * cc + 1] = make_float2(dm.x * sm.y, dm.y * sm.w);
			} else {
#pragma unroll
				for (int i = 0; i < 8; i++) lo[i] = hi[i] = 0;
				sad[l][2 * cc] = sad[l][2 * cc + 1] = make_float2(0.f, 0.f);
			}
		}
		{ // activations: 64 tokens x 4 blocks, one per thread
			uint l = tid / 4, kb = tid % 4, t = t0 + l;
			uint4 *dst = (uint4 *)(sb + l * MQ_LD + kb * 32);
			if (t < a.tokens) {
				const uint4 *src = (const uint4 *)(xq + (size_t)t * a.cols + k0 + kb * 32);
				dst[0] = src[0];
				dst[1] = src[1];
				sbd[l][kb] = xs[(size_t)t * xnb + k0 / 32 + kb];
			} else {
				dst[0] = dst[1] = make_uint4(0, 0, 0, 0);
				sbd[l][kb] = make_float2(0.f, 0.f);
			}
		}
		__syncthreads();
#pragma unroll
		for (int kb = 0; kb < 4; kb++) {
			int c[2][4][4];
			uint fa[2][4], fb[4][2];
#pragma unroll
			for (int i = 0; i < 2; i++) {
				const signed char *pa = sa + (wr + 16 * i + g) * MQ_LD + kb * 32 + 4 * q4;
				fa[i][0] = *(const uint *)pa;
				fa[i][1] = *(const uint *)(pa + 8 * MQ_LD);
				fa[i][2] = *(const uint *)(pa + 16);
				fa[i][3] = *(const uint *)(pa + 8 * MQ_LD + 16);
			}
#pragma unroll
			for (int j = 0; j < 4; j++) {
				const signed char *pb = sb + (wt + 8 * j + g) * MQ_LD + kb * 32 + 4 * q4;
				fb[j][0] = *(const uint *)pb;
				fb[j][1] = *(const uint *)(pb + 16);
			}
#pragma unroll
			for (int i = 0; i < 2; i++)
#pragma unroll
				for (int j = 0; j < 4; j++) {
					asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
					             "{%10,%11,%12,%13};"
					             : "=r"(c[i][j][0]), "=r"(c[i][j][1]), "=r"(c[i][j][2]), "=r"(c[i][j][3])
					             : "r"(fa[i][0]), "r"(fa[i][1]), "r"(fa[i][2]), "r"(fa[i][3]), "r"(fb[j][0]), "r"(fb[j][1]),
					               "r"(0), "r"(0), "r"(0), "r"(0));
				}
			// c[i][j][e]: row wr+16i+g (+8 for e >= 2), token wt+8j+2*q4 (+1 for odd e)
			float2 ad[2][2], bd[4][2];
#pragma unroll
			for (int i = 0; i < 2; i++) {
				ad[i][0] = sad[wr + 16 * i + g][kb];
				ad[i][1] = sad[wr + 16 * i + g + 8][kb];
			}
#pragma unroll
			for (int j = 0; j < 4; j++) {
				bd[j][0] = sbd[wt + 8 * j + 2 * q4][kb];
				bd[j][1] = sbd[wt + 8 * j + 2 * q4 + 1][kb];
			}
#pragma unroll
			for (int i = 0; i < 2; i++)
#pragma unroll
				for (int j = 0; j < 4; j++)
#pragma unroll
					for (int e = 0; e < 4; e++) {
						float2 A = ad[i][e / 2], B = bd[j][e % 2];
						acc[i][j][e] += (float)c[i][j][e] * A.x * B.x - A.y * B.y;
					}
		}
		__syncthreads();
	}
#pragma unroll
	for (int i = 0; i < 2; i++)
#pragma unroll
		for (int j = 0; j < 4; j++)
#pragma unroll
			for (int e = 0; e < 4; e++) {
				uint r = r0 + wr + 16 * i + g + 8 * (e / 2), t = t0 + wt + 8 * j + 2 * q4 + e % 2;
				if (r < a.rows && t < a.tokens) y[(size_t)t * a.rows + r] = acc[i][j][e];
			}
}

extern "C" __global__ void __launch_bounds__(256) matmul_q4k_mmq(const uchar *w, const signed char *xq, const float2 *xs, float *y,
                                                                 MatmulArgs a) {
	matmul_mmq(w, xq, xs, y, a);
}
