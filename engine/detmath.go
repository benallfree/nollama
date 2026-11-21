package engine

import "math"

// Portable float math. The Go compiler may fuse x*y+z into one FMA instruction on arm64, ppc64le and s390x (and
// amd64 with GOAMD64=v3) but not on plain amd64, so the same Go source can round differently per architecture. An
// explicit conversion, float64(x*y) or float32(x*y), rounds the product and blocks the fusion. The standard library's
// math.Sin/Cos/Exp/Log/Pow are written without such conversions (and some have per-architecture assembly), so their
// results also differ in the last bit between arm64 and amd64. The functions here give the same bits everywhere.

// sincos is math.Sincos for 0 <= x < 2^29 (Cephes, as in Go's math/sin.go), with every product rounded.
func sincos(x float64) (sin, cos float64) {
	const (
		pi4a = 7.85398125648498535156e-1  // 0x3fe921fb40000000, Pi/4 split into three parts
		pi4b = 3.77489470793079817668e-8  // 0x3e64442d00000000,
		pi4c = 2.69515142907905952645e-15 // 0x3ce8469898cc5170,
	)
	j := uint64(float64(x * (4 / 3.14159265358979323846264338327950288419716939937510582097494459)))
	y := float64(j)
	if j&1 == 1 {
		j++
		y++
	}
	j &= 7
	z := ((x - float64(y*pi4a)) - float64(y*pi4b)) - float64(y*pi4c)
	sinSign, cosSign := false, false
	if j > 3 {
		j -= 4
		sinSign, cosSign = !sinSign, !cosSign
	}
	if j > 1 {
		cosSign = !cosSign
	}
	zz := float64(z * z)
	pc := -1.13585365213876817300e-11
	pc = float64(pc*zz) + 2.08757008419747316778e-9
	pc = float64(pc*zz) + -2.75573141792967388112e-7
	pc = float64(pc*zz) + 2.48015872888517045348e-5
	pc = float64(pc*zz) + -1.38888888888730564116e-3
	pc = float64(pc*zz) + 4.16666666666665929218e-2
	ps := 1.58962301576546568060e-10
	ps = float64(ps*zz) + -2.50507477628578072866e-8
	ps = float64(ps*zz) + 2.75573136213857245213e-6
	ps = float64(ps*zz) + -1.98412698295895385996e-4
	ps = float64(ps*zz) + 8.33333333332211858878e-3
	ps = float64(ps*zz) + -1.66666666666666307295e-1
	cos = 1.0 - float64(0.5*zz) + float64(float64(zz*zz)*pc)
	sin = z + float64(float64(z*zz)*ps)
	if j == 1 || j == 2 {
		sin, cos = cos, sin
	}
	if cosSign {
		cos = -cos
	}
	if sinSign {
		sin = -sin
	}
	return sin, cos
}

// exp64 is math.Exp (FreeBSD's e_exp.c, as in Go's math/exp.go) with every product rounded. NaN and ±Inf follow
// math.Exp; results that would be subnormal flush through math.Ldexp as there.
func exp64(x float64) float64 {
	const (
		ln2hi = 6.93147180369123816490e-01
		ln2lo = 1.90821492927058770002e-10
		log2e = 1.44269504088896338700e+00
		over  = 7.09782712893383973096e+02
		under = -7.45133219101941108420e+02
		p1    = 1.66666666666666657415e-01
		p2    = -2.77777777770155933842e-03
		p3    = 6.61375632143793436117e-05
		p4    = -1.65339022054652515390e-06
		p5    = 4.13813679705723846039e-08
	)
	switch {
	case x != x || x > over:
		return x * math.Inf(1) // NaN stays NaN, overflow is +Inf
	case x < under:
		return 0
	case -3.725290298461914e-09 < x && x < 3.725290298461914e-09:
		return 1 + x
	}
	var k int
	if x < 0 {
		k = int(float64(log2e*x) - 0.5)
	} else {
		k = int(float64(log2e*x) + 0.5)
	}
	hi := x - float64(float64(k)*ln2hi)
	lo := float64(float64(k) * ln2lo) // a product kept in a variable can still fuse with the next add: convert
	r := hi - lo
	t := float64(r * r)
	c := p5
	c = float64(c*t) + p4
	c = float64(c*t) + p3
	c = float64(c*t) + p2
	c = float64(c*t) + p1
	c = r - float64(t*c)
	y := 1 - ((lo - float64(r*c)/(2-c)) - hi)
	return math.Ldexp(y, k)
}

// log64 is math.Log (FreeBSD's e_log.c, as in Go's math/log.go) with every product rounded.
func log64(x float64) float64 {
	const (
		ln2hi = 6.93147180369123816490e-01
		ln2lo = 1.90821492927058770002e-10
		l1    = 6.666666666666735130e-01
		l2    = 3.999999999940941908e-01
		l3    = 2.857142874366239149e-01
		l4    = 2.222219843214978396e-01
		l5    = 1.818357216161805012e-01
		l6    = 1.531383769920937332e-01
		l7    = 1.479819860511658591e-01
	)
	switch {
	case x != x || math.IsInf(x, 1):
		return x
	case x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(-1)
	}
	f1, ki := math.Frexp(x)
	if f1 < math.Sqrt2/2 {
		f1 *= 2
		ki--
	}
	f := f1 - 1
	k := float64(ki)
	s := f / (2 + f)
	s2 := float64(s * s)
	s4 := float64(s2 * s2)
	t1 := float64(s2 * (l1 + float64(s4*(l3+float64(s4*(l5+float64(s4*l7)))))))
	t2 := float64(s4 * (l2 + float64(s4*(l4+float64(s4*l6)))))
	r := t1 + t2
	hfsq := float64(float64(0.5*f) * f)
	return float64(k*ln2hi) - ((hfsq - (float64(s*(hfsq+r)) + float64(k*ln2lo))) - f)
}

// ftz32 flushes a subnormal (below 2^-126 in magnitude) to a zero of the same sign. GPUs do this to every float32
// input and result (Metal does even with fast math off), so the reference flushes where small values arise in practice:
// rmsnorm, SwiGLU, the softmax probabilities, and activation blocks of only tiny values (qTiny).
func ftz32(v float32) float32 {
	if b := math.Float32bits(v); b&0x7F800000 == 0 {
		return math.Float32frombits(b & 0x80000000)
	}
	return v
}

// qTiny is the largest magnitude below which an activation block quantizes to all zeros (Q8_0 and Q8_K), so that its
// scale times any weight scale (even a subnormal f16) stays far above 2^-126, where a GPU would flush it.
const qTiny = 0x1p-64

// fma32 is a*b+c rounded once to float32, as the arm64 FMADD/FMLA and amd64 VFMADD instructions compute it. The
// product is exact in float64; the sum is rounded to float64 "to odd" (an inexact result gets its lowest bit set), and
// with 29 spare bits that makes the final rounding to float32 correct. The quantized dot products' reference order is
// written with it, so the assembly kernels can match the Go ones bit for bit.
func fma32(a, b, c float32) float32 {
	p := float64(a) * float64(b) // exact: 24+24 bits
	s := p + float64(c)
	if s-s != 0 { // Inf or NaN
		return float32(s)
	}
	bb := s - p // TwoSum: e is the exact rounding error of s
	e := (p - (s - bb)) + (float64(c) - bb)
	if e != 0 {
		if u := math.Float64bits(s); u&1 == 0 {
			if (e > 0) == (s > 0) {
				u++
			} else {
				u--
			}
			s = math.Float64frombits(u)
		}
	}
	return float32(s)
}
