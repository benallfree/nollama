//go:build arm64

package engine

import (
	"os"
	"unsafe"
)

//go:noescape
func gemmQ4K2x4Asm(w0, w1, s0, s1, y unsafe.Pointer, nb int, out *float32)

//go:noescape
func gemmF16x2x4Asm(w0, w1, y unsafe.Pointer, stride, n int, out *float32)

//go:noescape
func gemmQ6K1x4Asm(w, y unsafe.Pointer, nb int, out *float32)

//go:noescape
func q4kScalesAsm(w unsafe.Pointer, nb int, dst unsafe.Pointer)

// noGemm (AGENT_ENGINE_NOGEMM set) keeps the 4-column prompt kernels, for
// comparing against them.
var noGemm = os.Getenv("AGENT_ENGINE_NOGEMM") != ""

// archFloatAsm turns on the 2-row x 4-column F16 prompt kernel.
func archFloatAsm() {
	if noGemm {
		return
	}
	gemmF16x2x4 = func(w0, w1 []byte, y []float32, stride int, out *[8]float32) {
		n := len(w0) / 2
		if n%16 != 0 {
			for r, w := range [][]byte{w0, w1} {
				for c := 0; c < 4; c++ {
					out[r*4+c] = dotF16F32Go(w, y[c*stride:c*stride+n])
				}
			}
			return
		}
		_, _ = w1[2*n-1], y[3*stride+n-1]
		gemmF16x2x4Asm(bptr(w0), bptr(w1), fptr(y), stride*4, n, &out[0])
	}
}

// archIntAsm turns on the Q4_K (2-row) and Q6_K (1-row) 4-column prompt kernels.
func archIntAsm() {
	if noGemm {
		return
	}
	q4kScales = func(w, dst []byte, nb int) {
		_, _ = w[nb*q4kBytes-1], dst[nb*q4kScaleBytes-1]
		q4kScalesAsm(bptr(w), nb, bptr(dst))
	}
	gemmQ6K1x4 = func(w, y []byte, nb int, out *[4]float32) {
		_, _ = w[nb*q6kBytes-1], y[nb*q8k6Bytes-1]
		gemmQ6K1x4Asm(bptr(w), bptr(y), nb, &out[0])
	}
	gemmQ4K2x4 = func(w0, w1, s0, s1, y []byte, nb int, out *[8]float32) {
		_, _, _ = w0[nb*q4kBytes-1], w1[nb*q4kBytes-1], y[nb*q8k4Bytes-1]
		_, _ = s0[nb*q4kScaleBytes-1], s1[nb*q4kScaleBytes-1]
		gemmQ4K2x4Asm(bptr(w0), bptr(w1), bptr(s0), bptr(s1), bptr(y), nb, &out[0])
	}
}
