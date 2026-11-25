//go:build amd64

package engine

import (
	"encoding/binary"
	"math"
)

//go:noescape
func expSubAVX2(x *float32, n int, m float32)

//go:noescape
func swigluAVX2(dst, gate, up *float32, n int)

//go:noescape
func scaleMaxAVX2(x *float32, n int, s float32) float32

//go:noescape
func mulFTZAVX2(x *float32, n int, s float32)

//go:noescape
func rmsSumSqAVX2(x *float32, n int, acc *[32]float32)

//go:noescape
func rmsScaleAVX2(out, x, w *float32, n int, s float32)

//go:noescape
func q8kAbsMaxAVX2(x *float32) (amax, max float32)

//go:noescape
func q8kQuantAVX2(x *float32, iscale float32, ob *byte)

// useElementwiseAVX2 plugs in the AVX2 row steps (elementwise_amd64.s); tails of fewer than 8 go through Go.
func useElementwiseAVX2() {
	expSubRow = func(x []float32, m float32) {
		n := len(x) &^ 7
		if n > 0 {
			expSubAVX2(&x[0], n, m)
		}
		expSubRowGo(x[n:], m)
	}
	swigluRow = func(h, g, u []float32) {
		_, _ = g[:len(h)], u[:len(h)]
		n := len(h) &^ 7
		if n > 0 {
			swigluAVX2(&h[0], &g[0], &u[0], n)
		}
		swigluRowGo(h[n:], g[n:], u[n:])
	}
	scaleMaxRow = func(x []float32, s float32) float32 {
		n := len(x) &^ 7
		if n == 0 {
			return scaleMaxRowGo(x, s)
		}
		return max(scaleMaxAVX2(&x[0], n, s), scaleMaxRowGo(x[n:], s))
	}
	rmsSumSq = func(x []float32, acc *[32]float32) {
		if len(x) > 0 {
			rmsSumSqAVX2(&x[0], len(x), acc)
		}
	}
	rmsScale = func(out, x, w []float32, s float32) {
		_, _ = out[len(x)-1], w[len(x)-1]
		rmsScaleAVX2(&out[0], &x[0], &w[0], len(x), s)
	}
	quantizeQ8K = func(x []float32, out []byte) {
		_ = out[len(x)/qk*q8kBytes-1]
		for b := 0; b < len(x)/qk; b++ {
			xb := x[b*qk : b*qk+qk]
			ob := out[b*q8kBytes : b*q8kBytes+q8kBytes]
			amax, max := q8kAbsMaxAVX2(&xb[0])
			if amax < qTiny {
				clear(ob)
				continue
			}
			iscale := -128 / max
			q8kQuantAVX2(&xb[0], iscale, &ob[0])
			binary.LittleEndian.PutUint32(ob, math.Float32bits(1/iscale))
		}
	}
	mulFTZRow = func(x []float32, s float32) {
		n := len(x) &^ 7
		if n > 0 {
			mulFTZAVX2(&x[0], n, s)
		}
		mulFTZRowGo(x[n:], s)
	}
}
