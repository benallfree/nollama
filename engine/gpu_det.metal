// nollama's deterministic Metal kernels: each computes exactly what the CPU's reference code computes, bit for bit
// (deterministic mode, LoadOptions.Deterministic). Compiled with fast math off (IEEE division and square roots) and,
// from here to the end of the source, no fusing of a multiply and an add into one multiply-add unless written as
// fma(): the CPU rounds each product on its own (see fma32 in detmath.go).
//
// Each kernel names the Go function whose order it follows. Sums of integers are exact, so their order is free;
// every float sum runs in the reference's order.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// tree32 is sum32 (quant.go) over a simdgroup's 32 lanes: lane i adds lane i+s for s = 16, 8, 4, 2, 1. Every lane
// gets the total.
static inline float tree32(float v) {
	for (ushort s = 16; s > 0; s >>= 1) {
		v += simd_shuffle_down(v, s);
	}
	return simd_broadcast(v, 0);
}

// expf_ref is expf (kernels.go).
static inline float expf_ref(float x) {
	if (x < -87.0f) {
		return 0;
	}
	if (x > 88.0f) {
		x = 88.0f;
	}
	// n = floor(t + 0.5) with t + 0.5 exact, as Go computes it in float64: t - floor(t) is exact here (Sterbenz)
	float t = x * 1.44269504088896341f;
	float n = floor(t);
	if (t - n >= 0.5f) {
		n += 1;
	}
	float r = x - n * 0.693359375f - n * -2.12194440e-4f;
	float p = 1.9875691500e-4f;
	p = p * r + 1.3981999507e-3f;
	p = p * r + 8.3334519073e-3f;
	p = p * r + 4.1665795894e-2f;
	p = p * r + 1.6666665459e-1f;
	p = p * r + 5.0000001201e-1f;
	p = p * r * r + r + 1;
	return p * as_type<float>(uint(int(n) + 127) << 23);
}

struct DetNormArgs {
	uint n;
	float eps;
};

// rmsnorm_det is rmsnorm (forward.go): one simdgroup (32 threads) per row; lane l sums x[i]^2 for i = l mod 32.
kernel void rmsnorm_det(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant DetNormArgs &a [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
	device const float *xr = x + (ulong)row * a.n;
	device float *orow = out + (ulong)row * a.n;
	float acc = 0;
	for (uint i = lane; i < a.n; i += 32) {
		acc = fma(xr[i], xr[i], acc);
	}
	float ss = tree32(acc);
	float scale = 1.0f / sqrt(ss / float(a.n) + a.eps);
	for (uint i = lane; i < a.n; i += 32) {
		orow[i] = xr[i] * scale * w[i];
	}
}

// swiglu_det is forward's SwiGLU: out = gate / (1 + expf(-gate)) * up.
kernel void swiglu_det(device const float *gate [[buffer(0)]],
                       device const float *up [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant uint &n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
	if (i < n) {
		float g = gate[i];
		out[i] = g / (1.0f + expf_ref(-g)) * up[i];
	}
}

// quant_q8_0_det is quantizeQ8_0 (kernels.go): one simdgroup per 32-element block, one element per lane; out is
// 36-byte blocks (d f32, qs[32]).
kernel void quant_q8_0_det(device const float *x [[buffer(0)]],
                           device uchar *out [[buffer(1)]],
                           constant uint &nblocks [[buffer(2)]],
                           uint g [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint b = g * nsg + sg;
	if (b >= nblocks) {
		return;
	}
	float v = x[(ulong)b * 32 + lane];
	float amax = simd_max(fabs(v));
	float d = 0, id = 0;
	if (amax >= 0x1p-64f) { // qTiny (detmath.go): a block of tiny values quantizes to zero
		d = amax / 127.0f;
		id = 1.0f / d;
	}
	device uchar *ob = out + (ulong)b * 36;
	ob[4 + lane] = uchar(char(int(rint(v * id))));
	if (lane == 0) {
		*(device float *)ob = d;
	}
}

// quant_q8_k_det is quantizeQ8K (quant.go): one simdgroup per 256-element block, lane l takes elements 8l..8l+7;
// out is ggml's block_q8_K (d f32, qs[256] int8, bsums[16] int16), 292 bytes.
kernel void quant_q8_k_det(device const float *x [[buffer(0)]],
                           device uchar *out [[buffer(1)]],
                           constant uint &nblocks [[buffer(2)]],
                           uint g [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint b = g * nsg + sg;
	if (b >= nblocks) {
		return;
	}
	device const float *xb = x + (ulong)b * 256 + lane * 8;
	float v[8];
	// the first element with the largest magnitude, as quantizeQ8K's strict > picks it
	float amax = 0, mx = 0;
	uint at = 0xFFFFFFFF;
	for (uint i = 0; i < 8; i++) {
		v[i] = xb[i];
		float a = fabs(v[i]);
		if (a > amax) {
			amax = a, mx = v[i], at = lane * 8 + i;
		}
	}
	for (ushort s = 16; s > 0; s >>= 1) {
		float oa = simd_shuffle_xor(amax, s), om = simd_shuffle_xor(mx, s);
		uint oat = simd_shuffle_xor(at, s);
		if (oa > amax || (oa == amax && oat < at)) {
			amax = oa, mx = om, at = oat;
		}
	}
	device uchar *ob = out + (ulong)b * 292;
	if (amax < 0x1p-64f) { // qTiny: zeros, or values too small for a finite -128/max
		for (uint i = 0; i < 8; i++) {
			ob[4 + lane * 8 + i] = 0;
		}
		if (lane < 16) {
			*(device short *)(ob + 260 + 2 * lane) = 0;
		}
		if (lane == 0) {
			*(device float *)ob = 0;
		}
		return;
	}
	float iscale = -128.0f / mx;
	int s = 0;
	for (uint i = 0; i < 8; i++) {
		int q = clamp(int(rint(iscale * v[i])), -128, 127);
		ob[4 + lane * 8 + i] = uchar(char(q));
		s += q;
	}
	s += simd_shuffle_down(s, 1); // bsums[j] covers elements 16j..16j+15: lanes 2j and 2j+1
	if (lane % 2 == 0) {
		*(device short *)(ob + 260 + lane) = short(s);
	}
	if (lane == 0) {
		*(device float *)ob = 1.0f / iscale;
	}
}

struct DetMatvecArgs {
	uint rows;     // output rows
	uint nb;       // blocks per row
	uint rowBytes; // bytes per weight row
	uint actBytes; // bytes per quantized activation row
};

// matvec_q8_0_det is dotQ8_0Go (kernels.go): y[t*rows + r] = W row r . activation row t (36-byte Q8_0 blocks).
// One simdgroup per row; four 8-lane groups take blocks b0..b0+3, 4 elements per lane, and every lane runs the
// per-block multiply-add chain in block order. Grid: (row groups, tokens).
kernel void matvec_q8_0_det(device const uchar *w [[buffer(0)]],
                            device const uchar *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant DetMatvecArgs &a [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
	uint r = tg.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	device const uchar *row = w + (ulong)r * a.rowBytes;
	device const uchar *xa = x + (ulong)tg.y * a.actBytes;
	uint e = (lane % 8) * 4, grp = lane / 8;
	float sum = 0;
	for (uint b0 = 0; b0 < a.nb; b0 += 4) {
		uint b = b0 + grp;
		int s = 0;
		float d = 0;
		if (b < a.nb) {
			device const char *wq = (device const char *)(row + b * 34 + 2 + e);
			device const char *xq = (device const char *)(xa + b * 36 + 4 + e);
			s = int(wq[0]) * int(xq[0]) + int(wq[1]) * int(xq[1]) + int(wq[2]) * int(xq[2]) + int(wq[3]) * int(xq[3]);
			d = float(*(device const half *)(row + b * 34)) * *(device const float *)(xa + b * 36);
		}
		s += simd_shuffle_xor(s, 1);
		s += simd_shuffle_xor(s, 2);
		s += simd_shuffle_xor(s, 4);
		uint nk = min(4u, a.nb - b0);
		for (uint k = 0; k < nk; k++) {
			sum = fma(simd_shuffle(d, k * 8), float(simd_shuffle(s, k * 8)), sum);
		}
	}
	if (lane == 0) {
		y[(ulong)tg.y * a.rows + r] = sum;
	}
}

// Decodes the 6-bit scale and min of sub-block j from a Q4_K scales array (scaleMinK4 in quant.go).
static inline uchar2 det_scale_min_k4(int j, device const uchar *q) {
	if (j < 4) {
		return uchar2(q[j] & 63, q[j + 4] & 63);
	}
	return uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// matvec_q4k_det is dotQ4KQ8KGo (quant.go) against block_q8_K activations. One simdgroup per row; lane l takes
// elements 64j+l and 64j+32+l of each super-block (exact integer totals), then every lane runs the two multiply-adds
// per super-block in order.
kernel void matvec_q4k_det(device const uchar *w [[buffer(0)]],
                           device const uchar *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant DetMatvecArgs &a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
	uint r = tg.x * nsg + sg;
	if (r >= a.rows) {
		return;
	}
	device const uchar *row = w + (ulong)r * a.rowBytes;
	device const uchar *xa = x + (ulong)tg.y * a.actBytes;
	float sum = 0;
	for (uint b = 0; b < a.nb; b++) {
		device const uchar *blk = row + b * 144;
		device const uchar *ab = xa + b * 292;
		device const uchar *sc = blk + 4;
		device const uchar *qs = blk + 16;
		device const char *q8 = (device const char *)(ab + 4);
		device const short *bs = (device const short *)(ab + 260);
		int part = 0, mins = 0;
		for (int j = 0; j < 4; j++) {
			uchar2 sm1 = det_scale_min_k4(2 * j, sc);
			uchar2 sm2 = det_scale_min_k4(2 * j + 1, sc);
			uchar q = qs[32 * j + lane];
			part += int(q & 0xF) * int(q8[64 * j + lane]) * int(sm1.x) + int(q >> 4) * int(q8[64 * j + 32 + lane]) * int(sm2.x);
			mins += int(sm1.y) * (int(bs[4 * j]) + int(bs[4 * j + 1])) + int(sm2.y) * (int(bs[4 * j + 2]) + int(bs[4 * j + 3]));
		}
		int sumi = simd_sum(part);
		float yd = *(device const float *)ab;
		float d = float(*(device const half *)blk) * yd;
		float dmin = float(*(device const half *)(blk + 2)) * yd;
		sum = fma(d, float(sumi), sum);
		sum = fma(-dmin, float(mins), sum);
	}
	if (lane == 0) {
		y[(ulong)tg.y * a.rows + r] = sum;
	}
}

struct DetAttnArgs {
	uint heads;
	uint kvHeads;
	uint headDim;
	uint ctxP;
	uint pos;
	float scale;
};

// attention_det is forward's attention over the f32 KV cache (scores4Go, softmax, wsum4Go), one threadgroup per
// (query head, token). Scores: one multiply-add per d, in order; softmax: expf, the sum in position order; output:
// one multiply-add per position, in order. Layouts as attention_f32 (gpu.metal). Threadgroup memory: headDim + 2
// floats.
kernel void attention_det(device const float *q [[buffer(0)]],
                          device const float *kT [[buffer(1)]],
                          device const float *v [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          device float *sc [[buffer(4)]],
                          device const uint2 *mask [[buffer(5)]],
                          constant DetAttnArgs &a [[buffer(6)]],
                          threadgroup float *shm [[threadgroup(0)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 ntg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
	uint nt = ntg.x;
	uint h = tg.x, t = tg.y;
	uint hd = a.headDim, kvh = h / (a.heads / a.kvHeads), kvd = a.kvHeads * hd;
	uint ctx = a.pos + t + 1;
	uint2 m = mask[t];
	threadgroup float *qs = shm;       // [headDim]
	threadgroup float *red = shm + hd; // [nsg] partial maxima, then [0] = 1/sum
	device float *s = sc + ((ulong)t * a.heads + h) * a.ctxP;
	device const float *qh = q + ((ulong)t * a.heads + h) * hd;
	for (uint d = tid; d < hd; d += nt) {
		qs[d] = qh[d];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	device const float *kh = kT + (ulong)kvh * hd * a.ctxP;
	float mx = -INFINITY;
	for (uint j = tid; j < ctx; j += nt) {
		float x = -INFINITY;
		if (j < m.x || j >= m.y) {
			x = 0;
			for (uint d = 0; d < hd; d++) {
				x = fma(qs[d], kh[(ulong)d * a.ctxP + j], x);
			}
			x = x * a.scale;
		}
		s[j] = x;
		mx = max(mx, x);
	}
	mx = simd_max(mx);
	if (lane == 0) {
		red[sg] = mx;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	mx = red[0];
	for (uint i = 1; i < nsg; i++) {
		mx = max(mx, red[i]);
	}
	for (uint j = tid; j < ctx; j += nt) {
		s[j] = expf_ref(s[j] - mx);
	}
	threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
	if (tid == 0) {
		float sum = 0;
		for (uint j = 0; j < ctx; j++) {
			sum += s[j];
		}
		red[0] = 1.0f / sum;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float inv = red[0];
	for (uint j = tid; j < ctx; j += nt) {
		s[j] = s[j] * inv;
	}
	threadgroup_barrier(mem_flags::mem_device);
	device const float *vh = v + kvh * hd;
	for (uint d = tid; d < hd; d += nt) {
		float o = 0;
		for (uint j = 0; j < ctx; j++) {
			o = fma(s[j], vh[(ulong)j * kvd + d], o);
		}
		out[((ulong)t * a.heads + h) * hd + d] = o;
	}
}
