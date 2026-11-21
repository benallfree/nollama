// nollama's Metal kernels. Compiled at runtime from this source (see gpu.go), so builds need no Xcode.
#include <metal_stdlib>
using namespace metal;

// Decodes the 6-bit scale and min of sub-block j from a Q4_K scales array (as scaleMinK4 in quant.go).
static inline uchar2 scale_min_k4(int j, device const uchar *q) {
	if (j < 4) {
		return uchar2(q[j] & 63, q[j + 4] & 63);
	}
	return uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

struct MatvecArgs {
	uint rows;     // output rows
	uint nb;       // 256-element blocks per row
	uint rowBytes; // bytes per weight row
	uint cols;     // input length (nb*256)
};

// y[t*rows + r] = dot(W row r, x[t*cols:]) for Q4_K weights and float activations.
// One simdgroup per row; lane l handles elements 64j+l and 64j+32+l of every block, so
// both the weight bytes and the activations are read coalesced. Grid: (row groups, tokens).
kernel void matvec_q4k_f32(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant MatvecArgs &a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r = tg.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	device const uchar *row = w + (ulong)r * a.rowBytes;
	device const float *xt = x + (ulong)tg.y * a.cols;
	float sum = 0;
	for (uint b = 0; b < a.nb; b++) {
		device const uchar *blk = row + b * 144;
		float d = float(*(device const half *)blk);
		float dmin = float(*(device const half *)(blk + 2));
		device const uchar *sc = blk + 4;
		device const uchar *qs = blk + 16;
		device const float *xb = xt + b * 256;
		for (int j = 0; j < 4; j++) {
			uchar2 sm1 = scale_min_k4(2 * j, sc);
			uchar2 sm2 = scale_min_k4(2 * j + 1, sc);
			uchar q = qs[32 * j + lane];
			float x1 = xb[64 * j + lane];
			float x2 = xb[64 * j + 32 + lane];
			sum += (d * sm1.x * float(q & 0xF) - dmin * sm1.y) * x1;
			sum += (d * sm2.x * float(q >> 4) - dmin * sm2.y) * x2;
		}
	}
	sum = simd_sum(sum);
	if (lane == 0) {
		y[(ulong)tg.y * a.rows + r] = sum;
	}
}

// Same contract for Q8_0 weights (34-byte blocks: d f16, qs[32] int8); a.nb counts 32-element blocks.
// Lanes split into 4 groups of 8; group g takes blocks g, g+4, ...; each lane does 4 elements of its block.
kernel void matvec_q8_0_f32(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant MatvecArgs &a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
	uint r = tg.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	device const uchar *row = w + (ulong)r * a.rowBytes;
	device const float *xt = x + (ulong)tg.y * a.cols;
	uint e = (lane % 8) * 4;
	float sum = 0;
	for (uint b = lane / 8; b < a.nb; b += 4) {
		device const uchar *blk = row + b * 34;
		device const char *qs = (device const char *)(blk + 2 + e);
		device const float *xb = xt + b * 32 + e;
		float s = float(qs[0]) * xb[0] + float(qs[1]) * xb[1] + float(qs[2]) * xb[2] + float(qs[3]) * xb[3];
		sum += float(*(device const half *)blk) * s;
	}
	sum = simd_sum(sum);
	if (lane == 0) {
		y[(ulong)tg.y * a.rows + r] = sum;
	}
}
