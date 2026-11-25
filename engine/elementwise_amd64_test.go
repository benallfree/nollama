//go:build amd64

package engine

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// ewInputs: special values, the clamp edges, subnormals, a fine grid over the exp range and a sweep of float32 bit
// patterns (NaNs dropped: their payloads aren't part of the contract).
func ewInputs() []float32 {
	xs := []float32{0, float32(math.Copysign(0, -1)), 1, -1, -87, 88, -87.0001, 88.0001, -86.9999, 87.9999, -88, 89, -1e30, 1e30,
		float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(1), math.Float32frombits(0x807fffff),
		math.Float32frombits(0x00800000), math.Float32frombits(0x80800000), math.MaxFloat32, -math.MaxFloat32}
	for v := float32(-100); v <= 100; v += 1.0 / 64 {
		xs = append(xs, v, v+1e-5)
	}
	for b := uint64(0); b < 1<<32; b += 4093 {
		if v := math.Float32frombits(uint32(b)); v == v {
			xs = append(xs, v)
		}
	}
	return xs
}

func ewSameBits(t *testing.T, what string, got, want []float32, in []float32) {
	t.Helper()
	bad := 0
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			if bad < 5 {
				t.Errorf("%s[%d] (in %g, %#08x): got %g (%#08x), want %g (%#08x)", what, i, in[i], math.Float32bits(in[i]),
					got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d differ", what, bad, len(want))
	}
}

func TestElementwiseAVX2(t *testing.T) {
	if !hasAVX2FMA() {
		t.Skip("no AVX2")
	}
	xs := ewInputs()
	n := len(xs) &^ 7
	xs = xs[:n]
	rng := rand.New(rand.NewSource(1))

	for _, m := range []float32{0, 3.5, -2, 1e-3} {
		want := append([]float32(nil), xs...)
		expSubRowGo(want, m)
		got := append([]float32(nil), xs...)
		expSubAVX2(&got[0], n, m)
		ewSameBits(t, "expSub", got, want, xs)
	}

	u := make([]float32, n)
	for i := range u {
		u[i] = xs[rng.Intn(n)]
	}
	want := make([]float32, n)
	swigluRowGo(want, xs, u)
	got := make([]float32, n)
	swigluAVX2(&got[0], &xs[0], &u[0], n)
	ewSameBits(t, "swiglu", got, want, xs)

	for _, s := range []float32{0.0883883, 1, 1e-30, 3e-39} {
		want := append([]float32(nil), xs...)
		mulFTZRowGo(want, s)
		got := append([]float32(nil), xs...)
		mulFTZAVX2(&got[0], n, s)
		ewSameBits(t, "mulFTZ", got, want, xs)
	}

	// scaleMax on finite scores, every length and tail through the wrappers
	saved := [4]any{expSubRow, swigluRow, scaleMaxRow, mulFTZRow}
	useElementwiseAVX2()
	defer func() {
		expSubRow, swigluRow = saved[0].(func([]float32, float32)), saved[1].(func(h, g, u []float32))
		scaleMaxRow, mulFTZRow = saved[2].(func([]float32, float32) float32), saved[3].(func([]float32, float32))
	}()
	for l := 0; l <= 70; l++ {
		x := make([]float32, l)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 20)
		}
		w, g := append([]float32(nil), x...), append([]float32(nil), x...)
		mw, mg := scaleMaxRowGo(w, 0.125), scaleMaxRow(g, 0.125)
		ewSameBits(t, "scaleMax", g, w, x)
		if math.Float32bits(mw) != math.Float32bits(mg) {
			t.Errorf("scaleMax len %d: max %g, want %g", l, mg, mw)
		}
		expSubRowGo(w, mw)
		expSubRow(g, mg)
		ewSameBits(t, "expSubRow", g, w, x)
		mulFTZRowGo(w, 1/3.0)
		mulFTZRow(g, 1/3.0)
		ewSameBits(t, "mulFTZRow", g, w, x)
		h1, h2 := make([]float32, l), make([]float32, l)
		swigluRowGo(h1, x, w)
		swigluRow(h2, x, w)
		ewSameBits(t, "swigluRow", h2, h1, x)
	}
}

func BenchmarkElementwise(b *testing.B) {
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%300) * -0.05
	}
	u := append([]float32(nil), x...)
	h := make([]float32, len(x))
	b.Run("expSubGo", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			y := append(h[:0], x...)
			expSubRowGo(y, 0.5)
		}
	})
	b.Run("expSubAVX2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			y := append(h[:0], x...)
			expSubAVX2(&y[0], len(y), 0.5)
		}
	})
	b.Run("swigluGo", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			swigluRowGo(h, x, u)
		}
	})
	b.Run("swigluAVX2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			swigluAVX2(&h[0], &x[0], &u[0], len(h))
		}
	})
}

func TestRmsnormQuantizeAVX2(t *testing.T) {
	if !hasAVX2FMA() {
		t.Skip("no AVX2")
	}
	sq, sc, qz := rmsSumSq, rmsScale, quantizeQ8K
	useElementwiseAVX2()
	fastSq, fastSc, fastQ := rmsSumSq, rmsScale, quantizeQ8K
	defer func() { rmsSumSq, rmsScale, quantizeQ8K = sq, sc, qz }()
	rng := rand.New(rand.NewSource(2))
	sub := math.Float32frombits(0x00400000)
	for trial := 0; trial < 300; trial++ {
		n := 256 * (1 + trial%4)
		x, w := make([]float32, n), make([]float32, n)
		mag := math.Pow(10, float64(rng.Intn(60)-40))
		for i := range x {
			x[i] = float32(rng.NormFloat64() * mag)
			w[i] = float32(rng.NormFloat64())
			switch rng.Intn(40) {
			case 0:
				x[i] = sub
			case 1:
				w[i] = -sub
			case 2:
				x[i] = float32(rng.Intn(9)-4) * 0.5 // halves: round-to-even cases after scaling
			}
		}
		if trial%5 == 0 { // a tie in |x| with opposite signs: the first one sets the scale
			x[7], x[200] = -3*float32(mag), 3*float32(mag)
		}
		if trial%7 == 0 {
			for i := 0; i < 256; i++ {
				x[i] *= 1e-30
			}
		}
		og, of := make([]float32, n), make([]float32, n)
		rmsSumSq, rmsScale = nil, nil
		rmsnorm(og, x, w, 1e-6)
		rmsSumSq, rmsScale = fastSq, fastSc
		rmsnorm(of, x, w, 1e-6)
		ewSameBits(t, "rmsnorm", of, og, x)

		qg, qf := make([]byte, n/qk*q8kBytes), make([]byte, n/qk*q8kBytes)
		quantizeQ8KGo(x, qg)
		fastQ(x, qf)
		if !bytes.Equal(qg, qf) {
			for i := range qg {
				if qg[i] != qf[i] {
					t.Fatalf("trial %d: quantizeQ8K byte %d (block %d, off %d): %d, want %d", trial, i, i/q8kBytes, i%q8kBytes, qf[i], qg[i])
				}
			}
		}
	}
}
