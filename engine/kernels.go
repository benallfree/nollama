package engine

import (
	"encoding/binary"
	"math"
)

const (
	q80Bytes  = 34 // weight block: d f16, qs[32]
	q80aBytes = 36 // activation block: d f32, qs[32]
)

// Kernel entry points; arm64 with dotprod replaces them with NEON assembly.
var (
	dotQ4K     = func(x, y []byte, nb int) float32 { return dotQ4KQ8KGo(x, y, nb) }
	dotQ6K     = func(x, y []byte, nb int) float32 { return dotQ6KQ8KGo(x, y, nb) }
	dotQ80     = func(x, y []byte, nb int) float32 { return dotQ8_0Go(x, y, nb) }
	dotF16     = dotF16F32Go
	dotF32     = dotF32Go
	axpy       = axpyGo
	kernelName = "go"

	// 4-column variants: out[c] = dot(x, y[c*stride:]). Nil when not accelerated.
	dotQ4Kx4 func(x, y []byte, stride, nb int, out *[4]float32)
	dotQ6Kx4 func(x, y []byte, stride, nb int, out *[4]float32)
	dotF16x4 func(x []byte, y []float32, stride int, out *[4]float32) // stride in floats
	dotQ80x4 func(x, y []byte, nb int, out *[4]float32)               // y in q80x4 layout
)

// q80x4Bytes is one block of 4 interleaved activation rows: d[4] f32, qs[4][32].
const q80x4Bytes = 16 + 4*32

// interleaveQ8_0 copies row lane (0..3) of a group from 36-byte blocks into the q80x4 layout.
func interleaveQ8_0(src []byte, dst []byte, lane, nb int) {
	for b := 0; b < nb; b++ {
		s := src[b*q80aBytes:]
		d := dst[b*q80x4Bytes:]
		copy(d[4*lane:4*lane+4], s[:4])
		copy(d[16+32*lane:16+32*lane+32], s[4:36])
	}
}

// quantizeQ8_0 quantizes x (len multiple of 32) into 36-byte activation blocks.
func quantizeQ8_0(x []float32, out []byte) {
	for b := 0; b < len(x)/32; b++ {
		xb := x[b*32 : b*32+32]
		ob := out[b*q80aBytes : b*q80aBytes+q80aBytes]
		var amax float32
		for _, v := range xb {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d, id := float32(0), float32(0)
		if amax >= qTiny { // a block of only tiny values (or zeros) quantizes to zero: no huge or infinite 1/d
			d = amax / 127
			id = 1 / d
		}
		for j, v := range xb {
			ob[4+j] = byte(int8(math.RoundToEven(float64(v * id))))
		}
		binary.LittleEndian.PutUint32(ob, math.Float32bits(d))
	}
}

func dotQ8_0Go(x, y []byte, nb int) float32 {
	var sum float32
	for i := 0; i < nb; i++ {
		xb := x[i*q80Bytes:]
		yb := y[i*q80aBytes:]
		var s int32
		for j := 0; j < 32; j++ {
			s += int32(int8(xb[2+j])) * int32(int8(yb[4+j]))
		}
		d := f16(xb) * math.Float32frombits(binary.LittleEndian.Uint32(yb))
		sum = fma32(d, float32(s), sum) // reference order, as in dotQ4KQ8KGo: exact total, one multiply-add per block
	}
	return sum
}

// swiglu is silu(g)·u, flushing subnormals as a GPU does (swiglu_det in gpu_det.metal).
func swiglu(g, u float32) float32 {
	g = ftz32(g)
	return ftz32(ftz32(g/(1+expf(-g))) * ftz32(u))
}

// Elementwise row steps of softmax and SwiGLU. The Go versions define the results; the asm versions (amd64 AVX2)
// apply the same float32 operations to each element, so they are bit-identical.
var (
	expSubRow   = expSubRowGo   // x[j] = expf(x[j] - m)
	swigluRow   = swigluRowGo   // h[i] = swiglu(g[i], u[i])
	scaleMaxRow = scaleMaxRowGo // x[j] *= s; returns max(x) (-Inf when empty)
	mulFTZRow   = mulFTZRowGo   // x[j] = ftz32(x[j] * s)
)

func expSubRowGo(x []float32, m float32) {
	for j, v := range x {
		x[j] = expf(v - m)
	}
}

func swigluRowGo(h, g, u []float32) {
	g, u = g[:len(h)], u[:len(h)]
	for i := range h {
		h[i] = swiglu(g[i], u[i])
	}
}

func scaleMaxRowGo(x []float32, s float32) float32 {
	mx := float32(math.Inf(-1))
	for j := range x {
		x[j] *= s
		mx = max(mx, x[j])
	}
	return mx
}

func mulFTZRowGo(x []float32, s float32) {
	for j, v := range x {
		x[j] = ftz32(v * s)
	}
}

// expf is a fast float32 exp (max rel. error ~2e-7 on [-87, 88]).
func expf(x float32) float32 {
	if x < -87 {
		return 0
	}
	if x > 88 {
		x = 88
	}
	const log2e = 1.44269504088896341
	n := float32(math.Floor(float64(float32(x*log2e) + 0.5)))
	r := x - float32(n*0.693359375) - float32(n*-2.12194440e-4)
	// exp(r) on [-ln2/2, ln2/2]
	p := float32(1.9875691500e-4)
	p = float32(p*r) + 1.3981999507e-3
	p = float32(p*r) + 8.3334519073e-3
	p = float32(p*r) + 4.1665795894e-2
	p = float32(p*r) + 1.6666665459e-1
	p = float32(p*r) + 5.0000001201e-1
	p = float32(p*r*r) + r + 1
	return float32(p * math.Float32frombits(uint32(int32(n)+127)<<23)) // rounded: callers add to it (1+expf, sums)
}

// Four-query attention kernels (see forward): scores4 computes
// out[i*ostride+p] = Σ_d q[i*hd+d]·kT[d*dstride+p] for p < 16*nblk, hd = len(q)/4;
// wsum4 computes out[i*ostride+d] = Σ_{j<4*n4} p[i*pstride+j]·v[j*vstride+d] for d < hd.
var (
	scores4 = scores4Go
	wsum4   = wsum4Go
)

func scores4Go(q, kT []float32, dstride, nblk int, out []float32, ostride int) {
	hd := len(q) / 4
	n := 16 * nblk
	for i := 0; i < 4; i++ {
		o := out[i*ostride : i*ostride+n]
		clear(o)
		for d := 0; d < hd; d++ {
			axpyGo(q[i*hd+d], kT[d*dstride:d*dstride+n], o)
		}
	}
}

func wsum4Go(p []float32, pstride int, v []float32, vstride, n4 int, out []float32, ostride, hd int) {
	for i := 0; i < 4; i++ {
		o := out[i*ostride : i*ostride+hd]
		clear(o)
		for j := 0; j < 4*n4; j++ {
			axpyGo(p[i*pstride+j], v[j*vstride:j*vstride+hd], o)
		}
	}
}
