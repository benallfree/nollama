// What gpu_int.cu needs to read like gpu_int.metal: Metal's scalar and vector types, its simdgroup functions (as
// warp functions; both are 32 lanes) and its address-space words. Integer arithmetic only, so a kernel computes the
// same integers on either GPU.
#pragma once

// long is 64 bits on the CUDA targets (LP64 Linux, and nvcc matches the host), as in Metal.
typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define threadgroup

// vectors: Metal's int4 / short4 / char4 / uint2, with the operators the kernels use
struct __align__(4) mchar4 {
	signed char x, y, z, w;
};
struct __align__(8) mshort4 {
	short x, y, z, w;
};
struct __align__(16) mint4 {
	int x, y, z, w;
	__device__ mint4() {}
	__device__ mint4(int v) : x(v), y(v), z(v), w(v) {}
	__device__ mint4(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {}
	__device__ mint4(mchar4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
	__device__ mint4(mshort4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
	__device__ int &operator[](int i) { return (&x)[i]; }
	__device__ int operator[](int i) const { return (&x)[i]; }
};
#define char4 mchar4
#define short4 mshort4
#define int4 mint4

// int_gemm's float group sums (exact: integers below 2^24): Metal stores the operands as half, here as float
struct __align__(16) mfloat4 {
	float x, y, z, w;
	__device__ mfloat4() {}
	__device__ mfloat4(mchar4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
	__device__ mfloat4(mshort4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
};
#define half float
#define half4 mfloat4
#define float4 mfloat4
__device__ __forceinline__ float dot(mfloat4 a, mfloat4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__device__ __forceinline__ mint4 operator*(mint4 a, mint4 b) { return mint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ mint4 operator*(int a, mint4 b) { return mint4(a * b.x, a * b.y, a * b.z, a * b.w); }
__device__ __forceinline__ mint4 operator+(mint4 a, mint4 b) { return mint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ mint4 &operator+=(mint4 &a, mint4 b) {
	a = a + b;
	return a;
}
__device__ __forceinline__ mint4 operator>>(mint4 a, int s) { return mint4(a.x >> s, a.y >> s, a.z >> s, a.w >> s); }
__device__ __forceinline__ mint4 operator&(mint4 a, int m) { return mint4(a.x & m, a.y & m, a.z & m, a.w & m); }

// simdgroup functions on warps
__device__ __forceinline__ int simd_shuffle_xor(int v, ushort m) { return __shfl_xor_sync(0xffffffffu, v, m); }
__device__ __forceinline__ long shx(long v, ushort m) { return (long)__shfl_xor_sync(0xffffffffu, (long long)v, m); }
__device__ __forceinline__ int simd_sum(int v) {
	for (int o = 16; o > 0; o >>= 1) {
		v += __shfl_xor_sync(0xffffffffu, v, o);
	}
	return v;
}
__device__ __forceinline__ bool simd_any(bool b) { return __any_sync(0xffffffffu, b); }

// math
__device__ __forceinline__ int clz(ulong v) { return __clzll((long long)v); }
template <typename T> __device__ __forceinline__ T clamp(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

#define threadgroup_barrier(x) __syncthreads()
