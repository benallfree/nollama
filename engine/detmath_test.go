package engine

import (
	"math"
	"math/rand/v2"
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
		x := float64(i/64) * float64(float32(math.Pow(1000000, -2*float64(i%64)/128)))
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

func TestExpLog64(t *testing.T) {
	var we, wl float64
	for i := -800000; i <= 800000; i++ {
		x := float64(i) * 1e-3
		if g, w := exp64(x), math.Exp(x); w != 0 && !math.IsInf(w, 0) {
			we = math.Max(we, math.Abs(g-w)/w)
		}
	}
	for i := 1; i <= 1000000; i++ {
		x := float64(i) * 0.731
		g, w := log64(x), math.Log(x)
		wl = math.Max(wl, math.Abs(g-w)/math.Max(1, math.Abs(w)))
		x = 1 / x
		g, w = log64(x), math.Log(x)
		wl = math.Max(wl, math.Abs(g-w)/math.Max(1, math.Abs(w)))
	}
	// math.Exp's amd64 assembly is off by up to ~4e-14 relative; on arm64 (pure Go) exp64 matches it to 2e-16
	if we > 1e-13 || wl > 4e-16 {
		t.Fatalf("exp64 rel err %g, log64 err %g", we, wl)
	}
	for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0, 1e-300, 710, -746} {
		ge, we := exp64(x), math.Exp(x)
		if !(ge == we || (ge != ge && we != we)) {
			t.Errorf("exp64(%g) = %g, want %g", x, ge, we)
		}
	}
	for _, x := range []float64{math.NaN(), math.Inf(1), 0, -1, 1} { // not subnormals: amd64's math.Log gets them wrong
		gl, wl := log64(x), math.Log(x)
		if !(gl == wl || (gl != gl && wl != wl)) {
			t.Errorf("log64(%g) = %g, want %g", x, gl, wl)
		}
	}
	t.Logf("exp64 max rel err %.3g, log64 max err %.3g", we, wl)
}

// TestDetMathGolden: the portable functions give the same bits on every architecture. The checksums were computed on
// arm64 and match amd64 (GOAMD64 v1 and v3).
func TestDetMathGolden(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 5)) // integer-only generator: Float64 is exact, the same everywhere
	sum := func(f func() uint64, n int) uint64 {
		var c uint64
		for i := 0; i < n; i++ {
			c = c*1099511628211 ^ f()
		}
		return c
	}
	got := map[string]uint64{
		"sincos": sum(func() uint64 {
			s, c := sincos(r.Float64() * 41000)
			return math.Float64bits(s) ^ math.Float64bits(c)<<1
		}, 200000),
		"exp64": sum(func() uint64 { return math.Float64bits(exp64(float64(r.Float64()*1400) - 720)) }, 200000),
		"log64": sum(func() uint64 { return math.Float64bits(log64(math.Ldexp(r.Float64(), r.IntN(200)-100))) }, 200000),
		"expf":  sum(func() uint64 { return uint64(math.Float32bits(expf(float32(float64(r.Float64()*180) - 90)))) }, 200000),
		"rope": sum(func() uint64 {
			v := []float32{float32(r.Float64()), float32(r.Float64())}
			rope(v, []float32{float32(r.Float64())}, []float32{float32(r.Float64())})
			return uint64(math.Float32bits(v[0]))<<32 | uint64(math.Float32bits(v[1]))
		}, 200000),
		// fma32 last: it draws from r after the others, so their checksums stay put
		"fma32": sum(func() uint64 {
			a, b, c := math.Float32frombits(r.Uint32()), math.Float32frombits(r.Uint32()), math.Float32frombits(r.Uint32())
			f := fma32(a, b, c)
			if f != f {
				return 1 // NaN bit patterns differ by CPU (arm64's default NaN is positive, x86's negative)
			}
			return uint64(math.Float32bits(f))
		}, 200000),
	}
	for _, k := range []string{"sincos", "exp64", "log64", "expf", "fma32", "rope"} {
		w, ok := detMathGolden[k]
		t.Logf("%-6s %016x", k, got[k])
		if ok && w != got[k] {
			t.Errorf("%s: checksum %016x, want %016x (not portable on this CPU)", k, got[k], w)
		}
	}
}

var detMathGolden = map[string]uint64{
	"sincos": 0xce46a4fc25c8bbfb,
	"exp64":  0xd0656db02952d589,
	"log64":  0xeaf7e439eda877dc,
	"expf":   0x7fa6d21f096245bd,
	"fma32":  0x65c40c6c40cd67f1,
	"rope":   0xc478218e3d231971,
}

//go:noinline
func hwFMA32(a, b, c float32) float32 { return a*b + c } // fused into one instruction on arm64

func TestFMA32(t *testing.T) {
	if !fusesFMA() {
		t.Skip("this GOARCH doesn't fuse a*b+c; the golden checksum covers fma32 here")
	}
	r := rand.New(rand.NewPCG(11, 13))
	bad := 0
	for i := 0; i < 5_000_000; i++ {
		a := math.Float32frombits(r.Uint32())
		b := math.Float32frombits(r.Uint32())
		var c float32
		switch i % 3 {
		case 0:
			c = math.Float32frombits(r.Uint32())
		case 1: // near cancellation: c ≈ -a*b
			c = -float32(float64(a)*float64(b)) * (1 + float32(r.IntN(9)-4)*1e-7)
		default:
			c = float32(r.NormFloat64())
			a, b = float32(r.NormFloat64()), float32(r.NormFloat64()*1000)
		}
		g, w := fma32(a, b, c), hwFMA32(a, b, c)
		if math.Float32bits(g) != math.Float32bits(w) && !(g != g && w != w) {
			if bad++; bad < 5 {
				t.Errorf("fma32(%g, %g, %g) = %g, hardware %g", a, b, c, g, w)
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%d mismatches", bad)
	}
}

// fusesFMA reports whether the compiler fused hwFMA32 (it does on arm64).
func fusesFMA() bool {
	// a*a = 1 + 2^-11 + 2^-24 exactly. Unfused, the product rounds (a tie, to even) to 1 + 2^-11 and the sum is 0;
	// fused, the sum is 2^-24.
	a := float32(1 + 1.0/(1<<12))
	return hwFMA32(a, a, -(1+1.0/(1<<11))) != 0
}
