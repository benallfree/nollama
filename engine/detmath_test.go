package engine

import (
	"math"
	"os"
	"testing"
)

// TestDetMath prints checksums of the float64 math the engine and the teacher-forcing hook use, to compare across GOARCH.
func TestDetMath(t *testing.T) {
	if os.Getenv("DET_KERNELS") == "" {
		t.Skip("DET_KERNELS not set")
	}
	h := func(name string, f func(i int) float64, n int) {
		var c uint64
		var c32 uint64
		for i := 0; i < n; i++ {
			v := f(i)
			c = c*1099511628211 ^ math.Float64bits(v)
			c32 = c32*1099511628211 ^ uint64(math.Float32bits(float32(v)))
		}
		t.Logf("%-10s f64 %016x  as f32 %016x", name, c, c32)
	}
	inv := make([]float64, 64)
	for i := range inv {
		inv[i] = math.Pow(1000000, -2*float64(i)/128)
	}
	h("pow", func(i int) float64 { return inv[i] }, 64)
	h("cos", func(i int) float64 { return math.Cos(float64(i/64) * inv[i%64]) }, 40960*64)
	h("sin", func(i int) float64 { return math.Sin(float64(i/64) * inv[i%64]) }, 40960*64)
	h("exp", func(i int) float64 { return math.Exp(-float64(i) * 1e-4) }, 400000)
	h("log", func(i int) float64 { return math.Log(1 + float64(i)*0.37) }, 400000)
}

func TestSincos(t *testing.T) {
	var worst float64
	var c uint64
	for i := 0; i < 40960*64; i++ {
		x := float64(i/64) * math.Pow(1000000, -2*float64(i%64)/128)
		s, co := sincos(x)
		ws, wc := math.Sincos(x)
		worst = math.Max(worst, math.Max(math.Abs(s-ws), math.Abs(co-wc)))
		c = c*1099511628211 ^ math.Float64bits(s) ^ math.Float64bits(co)<<1
	}
	if worst > 1e-12 {
		t.Fatalf("sincos off by %g", worst)
	}
	t.Logf("sincos max error vs math.Sincos %.3g, checksum %016x", worst, c)
}
