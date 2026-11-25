//go:build amd64

package engine

import (
	"os"
	"unsafe"
)

// An AVX-512BW Q4_K prompt kernel shaped like an outer product,
// 4 weight rows x 16 tokens per call. Each 32-bit lane of a 512-bit register holds one (row, token) pair: the four
// 128-bit lanes are the 4 rows, the four dwords inside one are 4 tokens. One unpacked weight vector (16 values of
// 4 rows) is split by VPSHUFD into 4 groups of 4 values and each is reused for 4 token groups; 8 VPMADDUBSW
// products are summed in int16 (max 8*2*15*128 = 30720, no saturation) before one VPMADDWD applies the sub-block
// scale. The integer totals are exact, and the float steps follow the reference order of dotQ4KQ8KGo.

// no4x16 (NOLLAMA_NO4X16 set) keeps the 2x4 prompt kernels, for comparing against it.
var no4x16 = os.Getenv("NOLLAMA_NO4X16") != ""

//go:noescape
func gemmQ4K4x16AVX512(w, y unsafe.Pointer, gstride, nb int, out *float32, ostride int)

//go:noescape
func gemmQ4K4x16VNNI(w, y unsafe.Pointer, gstride, nb int, out *float32, ostride int)

//go:noescape
func gemmQ6K4x16AVX512(w, y unsafe.Pointer, gstride, nb int, out *float32, ostride int)

//go:noescape
func gemmQ6K4x16VNNI(w, y unsafe.Pointer, gstride, nb int, out *float32, ostride int)

// hasAVX512VNNI: CPUID.7.0:ECX bit 11, on top of AVX-512BW.
func hasAVX512VNNI() bool {
	if !hasAVX512BW() {
		return false
	}
	_, _, ecx7, _ := cpuid(7, 0)
	return ecx7&(1<<11) != 0
}

// gemmQ4K4x16Asm runs one of the two asm versions (gemmQ4K4x16 describes the arguments).
func gemmQ4K4x16Asm(w, y []byte, gstride, nb int, out []float32, ostride int, vnni bool) {
	_ = w[nb*q4k4x16Bytes-1]
	_ = y[3*gstride+nb*q8k16GBytes-1]
	_ = out[15*ostride+3]
	if vnni {
		gemmQ4K4x16VNNI(unsafe.Pointer(&w[0]), unsafe.Pointer(&y[0]), gstride, nb, &out[0], 4*ostride)
		return
	}
	gemmQ4K4x16AVX512(unsafe.Pointer(&w[0]), unsafe.Pointer(&y[0]), gstride, nb, &out[0], 4*ostride)
}

// gemmQ6K4x16Asm is gemmQ4K4x16Asm for Q6_K rows packed by q6kPack4x16.
func gemmQ6K4x16Asm(w, y []byte, gstride, nb int, out []float32, ostride int, vnni bool) {
	_ = w[nb*q6k4x16Bytes-1]
	_ = y[3*gstride+nb*q8k16GBytes-1]
	_ = out[15*ostride+3]
	if vnni {
		gemmQ6K4x16VNNI(unsafe.Pointer(&w[0]), unsafe.Pointer(&y[0]), gstride, nb, &out[0], 4*ostride)
		return
	}
	gemmQ6K4x16AVX512(unsafe.Pointer(&w[0]), unsafe.Pointer(&y[0]), gstride, nb, &out[0], 4*ostride)
}
