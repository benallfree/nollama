package engine

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// testDetBackend opens the first GPU backend with deterministic kernels (no weights), or skips.
func testDetBackend(t *testing.T) gpuBackend {
	t.Helper()
	var why []string
	for _, open := range gpuOpeners {
		b, err := open(true)
		if err == nil {
			t.Cleanup(b.close)
			return b
		}
		why = append(why, err.Error())
	}
	t.Skipf("no deterministic GPU backend here: %v", why)
	return nil
}

// sameBits reports the first index where got and want differ in any bit (NaNs of any payload count as equal).
func sameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		g, w := got[i], want[i]
		if math.Float32bits(g) != math.Float32bits(w) && !(g != g && w != w) {
			t.Errorf("%s: [%d] = %v (%08x), CPU %v (%08x)", what, i, g, math.Float32bits(g), w, math.Float32bits(w))
			return
		}
	}
}

// detInput is n rows of k activations with the cases quantization must agree on: an all-zero block, magnitude ties
// of both signs (the first one sets a Q8_K block's scale), and values from tiny to large.
func detInput(r *rand.Rand, n, k int) []float32 {
	x := randFloats(r, n*k)
	for i := range x {
		x[i] *= float32(math.Exp(r.NormFloat64() * 2))
	}
	if n*k >= 512 {
		clear(x[256:512]) // one zero Q8_K block (and eight zero Q8_0 blocks)
	}
	if n*k >= 1024 {
		for i := 768; i < 1024; i++ { // a Q8_K block of tiny values, around qTiny
			x[i] = float32(r.NormFloat64() * 0x1p-65)
		}
		x[600], x[700] = -3.5, 3.5 // a tie between signs: the first decides the Q8_K scale's sign
		x[900], x[901] = 2.25, 2.25
	}
	return x
}

// cpuMatmul is out[t][r] = the CPU's dot product of weight row r and activation row t (its quantized forms).
func cpuMatmul(w *weight, x []float32, n int) []float32 {
	a := newActs(n, w.cols)
	a.n = n
	copy(a.f, x)
	a.quantize(nil, w)
	dot := rowDot(w, a)
	out := make([]float32, n*w.rows)
	for t := 0; t < n; t++ {
		for row := 0; row < w.rows; row++ {
			out[t*w.rows+row] = dot(w.row(row), t)
		}
	}
	return out
}

// TestGPUDetMatmul checks the deterministic matmul (quantize, then the integer matvec) against the CPU, bit for bit.
func TestGPUDetMatmul(t *testing.T) {
	b := testDetBackend(t)
	r := rand.New(rand.NewPCG(7, 8))
	const rows = 77
	type tc struct {
		typ          uint32
		nb, bs, bb   int
		scaleOffsets []int
	}
	cases := []tc{{typeQ4_K, 8, qk, q4kBytes, []int{0, 2}}, {typeQ8_0, 64, 32, q80Bytes, []int{0}}, {typeQ8_0, 3, 32, q80Bytes, []int{0}}}
	// one weights buffer holding every case, loaded once
	var offs []int
	size := 0
	for _, c := range cases {
		offs = append(offs, size)
		size += (rows*c.nb*c.bb + 63) &^ 63
	}
	file := alignedBytes(size)
	for i, c := range cases {
		copy(file[offs[i]:], randBlocks(r, rows*c.nb, c.bb, c.scaleOffsets))
	}
	if err := b.loadWeights(file); err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		w := &weight{typ: c.typ, cols: c.nb * c.bs, rows: rows, rowBytes: c.nb * c.bb, data: file[offs[i] : offs[i]+rows*c.nb*c.bb]}
		const n = 45
		x := detInput(r, n, w.cols)
		want := cpuMatmul(w, x, n)
		xr, or := upload(t, b, x), zeros(t, b, n*w.rows)
		act := zeros(t, b, gpuActBytes(n, w.cols)/4)
		for _, nt := range []int{1, 5, n} {
			bt := b.batch()
			bt.matmul(w, xr, or, nt, act)
			submit(t, bt)
			sameBits(t, fmt.Sprintf("%s %dx%d, %d tokens", typeName(w.typ), w.rows, w.cols, nt), download(b, or, nt*w.rows), want[:nt*w.rows])
		}
	}
}

// TestGPUDetElementwise checks rmsnorm and SwiGLU against the CPU, bit for bit.
func TestGPUDetElementwise(t *testing.T) {
	b := testDetBackend(t)
	r := rand.New(rand.NewPCG(9, 10))
	for _, c := range []struct{ rows, n int }{{3, 2048}, {16, 128}, {2, 100}} {
		x := detInput(r, c.rows, c.n)
		w := randFloats(r, c.n)
		want := make([]float32, len(x))
		for i := 0; i < c.rows; i++ {
			rmsnorm(want[i*c.n:(i+1)*c.n], x[i*c.n:(i+1)*c.n], w, 1e-6)
		}
		xb, wb := upload(t, b, x), upload(t, b, w)
		bt := b.batch()
		bt.rmsnorm(xb, wb, xb, c.rows, c.n, 1e-6) // in place, as forward runs QK-norm
		submit(t, bt)
		sameBits(t, fmt.Sprintf("rmsnorm %dx%d", c.rows, c.n), download(b, xb, len(x)), want)
	}

	const n = 1 << 16
	g, u := detInput(r, 1, n), randFloats(r, n)
	for i := 0; i < 400; i++ { // the edges of expf: clamped, underflowing, and around its rounding steps
		g[i] = float32(i-200) * 0.5
	}
	want := make([]float32, n)
	for i := range g {
		want[i] = swiglu(g[i], u[i])
	}
	gb, ub, ob := upload(t, b, g), upload(t, b, u), zeros(t, b, n)
	bt := b.batch()
	bt.swiglu(gb, ub, ob, n)
	submit(t, bt)
	sameBits(t, "swiglu", download(b, ob, n), want)
}

// cpuAttention is forward's attention for one (token, head): scores4Go's order, softmax as forward computes it,
// wsum4Go's order. ctx positions, the hidden range [lo, hi).
func cpuAttention(q, kT, v []float32, hd, ctxP, kvd, ctx, lo, hi int, scale float32) []float32 {
	row := make([]float32, ctx)
	for d := 0; d < hd; d++ {
		axpyGo(q[d], kT[d*ctxP:d*ctxP+ctx], row)
	}
	mx := float32(math.Inf(-1))
	for j := range row {
		row[j] *= scale
		if j >= lo && j < hi {
			row[j] = float32(math.Inf(-1))
		}
		mx = max(mx, row[j])
	}
	var sum float32
	for j := range row {
		row[j] = expf(row[j] - mx)
		sum += row[j]
	}
	inv := 1 / sum
	out := make([]float32, hd)
	for j := range row {
		row[j] = ftz32(row[j] * inv)
		axpyGo(row[j], v[j*kvd:j*kvd+hd], out)
	}
	return out
}

// TestGPUDetAttention checks causal attention (grouped heads, a packed-sequence mask) against the CPU, bit for bit.
func TestGPUDetAttention(t *testing.T) {
	b := testDetBackend(t)
	r := rand.New(rand.NewPCG(11, 12))
	const heads, kvHeads, hd, ctxP, pos, n = 8, 2, 128, 160, 100, 37
	kvd := kvHeads * hd
	scale := float32(1 / math.Sqrt(hd))
	q := randFloats(r, n*heads*hd)
	kc, vc := randFloats(r, kvd*ctxP), randFloats(r, ctxP*kvd)
	for i := range q {
		q[i] *= 3
	}
	for _, masked := range []bool{false, true} {
		mask := make([]uint32, 2*n)
		if masked {
			for t := 0; t < n; t++ {
				mask[2*t], mask[2*t+1] = 10, uint32(40+t)
			}
		}
		raw := u32(mask...)
		mb, err := b.alloc(len(raw))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(mb.release)
		b.write(gpuRef{buf: mb}, raw)
		qb, kb, vb := upload(t, b, q), upload(t, b, kc), upload(t, b, vc)
		ob, sb := zeros(t, b, n*heads*hd), zeros(t, b, n*heads*ctxP)
		bt := b.batch()
		bt.attention(gpuAttention{q: qb, out: ob, k: kb, v: vb, scratch: sb, mask: gpuRef{buf: mb}, kvType: typeF32,
			n: n, pos: pos, heads: heads, kvHeads: kvHeads, headDim: hd, ctxP: ctxP})
		submit(t, bt)
		got := download(b, ob, n*heads*hd)
		want := make([]float32, 0, len(got))
		for tk := 0; tk < n; tk++ {
			for h := 0; h < heads; h++ {
				kvh := h / (heads / kvHeads)
				want = append(want, cpuAttention(q[(tk*heads+h)*hd:(tk*heads+h+1)*hd], kc[kvh*hd*ctxP:], vc[kvh*hd:],
					hd, ctxP, kvd, pos+tk+1, int(mask[2*tk]), int(mask[2*tk+1]), scale)...)
			}
		}
		sameBits(t, fmt.Sprintf("attention, masked=%v", masked), got, want)
	}
}
