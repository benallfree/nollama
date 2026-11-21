//go:build arm64

package engine

import "unsafe"

//go:noescape
func gemmQ4K2x4Asm(w0, w1, s0, s1, y unsafe.Pointer, nb int, out *float32)

//go:noescape
func q4kScalesAsm(w unsafe.Pointer, nb int, dst unsafe.Pointer)

// archIntAsm turns on the 2-row x 4-column Q4_K prompt kernel.
func archIntAsm() {
	q4kScales = func(w, dst []byte, nb int) {
		_, _ = w[nb*q4kBytes-1], dst[nb*q4kScaleBytes-1]
		q4kScalesAsm(bptr(w), nb, bptr(dst))
	}
	gemmQ4K2x4 = func(w0, w1, s0, s1, y []byte, nb int, out *[8]float32) {
		_, _, _ = w0[nb*q4kBytes-1], w1[nb*q4kBytes-1], y[nb*q8k4Bytes-1]
		_, _ = s0[nb*q4kScaleBytes-1], s1[nb*q4kScaleBytes-1]
		gemmQ4K2x4Asm(bptr(w0), bptr(w1), bptr(s0), bptr(s1), bptr(y), nb, &out[0])
	}
}
