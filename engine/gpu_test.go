package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unsafe"
)

// testGGUF returns GGUF, or the cached qwen3-1.7b, or skips.
func testGGUF(t *testing.T) string {
	if p := os.Getenv("GGUF"); p != "" {
		return p
	}
	if dir, err := os.UserCacheDir(); err == nil {
		p := filepath.Join(dir, "nollama", "models", "qwen3-1.7b.gguf")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no model: set GGUF")
	return ""
}

func putFloats(b []byte, x []float32) {
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
}

func getFloat(b []byte, i int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
}

// testBackend opens the first GPU backend that works here (with no weights), or skips.
func testBackend(t *testing.T) gpuBackend {
	t.Helper()
	var why []string
	for _, open := range gpuOpeners {
		b, err := open(false)
		if err == nil {
			t.Cleanup(b.close)
			return b
		}
		why = append(why, err.Error())
	}
	skipGPU(t, errors.New(strings.Join(why, "; ")))
	return nil
}

// alignedBytes returns n zero bytes starting on a page boundary (loadWeights can wrap them without copying).
func alignedBytes(n int) []byte {
	page := os.Getpagesize()
	b := make([]byte, n+2*page)
	off := (page - int(uintptr(unsafe.Pointer(&b[0]))%uintptr(page))) % page
	return b[off : off+n : off+n+page]
}

func upload(t *testing.T, b gpuBackend, x []float32) gpuRef {
	t.Helper()
	buf, err := b.alloc(4 * max(len(x), 1))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(buf.release)
	raw := make([]byte, 4*len(x))
	putFloats(raw, x)
	r := gpuRef{buf: buf}
	b.write(r, raw)
	return r
}

func zeros(t *testing.T, b gpuBackend, n int) gpuRef { return upload(t, b, make([]float32, n)) }

func download(b gpuBackend, r gpuRef, n int) []float32 {
	raw := make([]byte, 4*n)
	b.read(r, raw)
	return floatsOf(raw, n)
}

func submit(t *testing.T, bt gpuBatch) {
	t.Helper()
	if err := bt.submit(); err != nil {
		t.Fatal(err)
	}
}

// refMatmul is out[t][r] = W row r · x[t] in float64 over the dequantized weights, and each output's
// sum of |products| (the scale its rounding error is measured against).
func refMatmul(w *weight, x []float32, n int) (out, mag []float64) {
	out, mag = make([]float64, n*w.rows), make([]float64, n*w.rows)
	tmp := make([]float32, w.cols)
	for row := 0; row < w.rows; row++ {
		dequantRow(w.typ, w.row(row), tmp)
		for tk := 0; tk < n; tk++ {
			for i, v := range tmp {
				p := float64(v) * float64(x[tk*w.cols+i])
				out[tk*w.rows+row] += p
				mag[tk*w.rows+row] += math.Abs(p)
			}
		}
	}
	return out, mag
}

// checkMatmul runs the backend's matmul for 1, 5 and 45 tokens of random input (one-token and tiled kernels)
// and checks it against refMatmul.
func checkMatmul(t *testing.T, b gpuBackend, w *weight, r *rand.Rand) {
	t.Helper()
	const n = 45
	x := randFloats(r, n*w.cols)
	xr, or := upload(t, b, x), zeros(t, b, n*w.rows)
	act := zeros(t, b, gpuActBytes(n, w.cols)/4)
	ref, mag := refMatmul(w, x, n)
	for _, nt := range []int{1, 5, n} {
		bt := b.batch()
		bt.matmul(w, xr, or, nt, act)
		submit(t, bt)
		got := download(b, or, nt*w.rows)
		var worst float64
		for i := range got {
			worst = max(worst, math.Abs(float64(got[i])-ref[i])/max(mag[i], 1e-6))
		}
		t.Logf("%s %dx%d, %d token(s): worst error %.1e of the row's magnitude", typeName(w.typ), w.rows, w.cols, nt, worst)
		if lim := promptTol(b, nt, 1e-5, 1e-4); worst > lim {
			t.Errorf("%s %dx%d, %d tokens: error %.2e", typeName(w.typ), w.rows, w.cols, nt, worst)
		}
	}
}

// TestGPUMatmulKQuantsSynthetic checks Q4_K, Q6_K and F16 on random blocks (the test model is Q8_0).
func TestGPUMatmulKQuantsSynthetic(t *testing.T) {
	b := testBackend(t)
	for _, c := range []struct {
		typ      uint32
		size     int
		f16s     []int // offsets of the block's f16 scales
		rows, nb int
	}{{typeQ4_K, q4kBytes, []int{0, 2}, 77, 8}, {typeQ6_K, q6kBytes, []int{208}, 77, 8}, {typeF16, 512, f16Offsets(256), 77, 8}} {
		if !b.supports(c.typ) {
			t.Logf("%s has no %s", b.name(), typeName(c.typ))
			continue
		}
		r := rand.New(rand.NewPCG(5, 6))
		file := alignedBytes(c.rows * c.nb * c.size)
		copy(file, randBlocks(r, c.rows*c.nb, c.size, c.f16s))
		if err := b.loadWeights(file); err != nil {
			t.Fatal(err)
		}
		checkMatmul(t, b, &weight{typ: c.typ, cols: c.nb * qk, rows: c.rows, rowBytes: c.nb * c.size, data: file}, r)
	}
}

// openModelGPU loads the test model and opens a backend with its weights, or skips.
func openModelGPU(t *testing.T) (*Model, gpuBackend) {
	m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	b, err := openGPU(m.g, false)
	if err != nil {
		skipGPU(t, err)
	}
	t.Cleanup(b.close)
	return m, b
}

// TestGPUMatmul checks matmul on a real model's weights (one-token and tiled kernels) and times one decode
// step's matrix work against the CPU, and a 512-token prompt's.
func TestGPUMatmul(t *testing.T) {
	m, b := openModelGPU(t)
	t.Logf("%s, file %d MB, unified=%v", b.name(), len(m.g.data)>>20, b.unified())
	var ws []*weight
	for _, l := range m.layers {
		for _, w := range []*weight{l.wq, l.wk, l.wv, l.wo, l.wgate, l.wup, l.wdown} {
			if b.supports(w.typ) {
				ws = append(ws, w)
			}
		}
	}
	if len(ws) == 0 {
		t.Skip("no weights of a type " + b.name() + " handles")
	}
	r := rand.New(rand.NewPCG(3, 4))
	l := m.layers[len(m.layers)-1]
	for _, w := range []*weight{l.wk, l.wdown} {
		if b.supports(w.typ) {
			checkMatmul(t, b, w, r)
		}
	}
	if testing.Short() {
		return
	}
	if b.supports(m.out.typ) {
		ws = append(ws, m.out)
	}
	maxRows, maxCols, bytes := 0, 0, 0
	for _, w := range ws {
		maxRows, maxCols = max(maxRows, w.rows), max(maxCols, w.cols)
		bytes += len(w.data)
	}
	const prompt = 512
	xr, or := upload(t, b, randFloats(r, prompt*maxCols)), zeros(t, b, prompt*maxRows)
	act := zeros(t, b, gpuActBytes(prompt, maxCols)/4)
	timeIt := func(n int, ws []*weight) time.Duration {
		run := func() {
			bt := b.batch()
			for _, w := range ws {
				bt.matmul(w, xr, or, n, act)
			}
			submit(t, bt)
		}
		run()
		t0 := time.Now()
		const iters = 3
		for i := 0; i < iters; i++ {
			run()
		}
		return time.Since(t0) / iters
	}
	gpuT := timeIt(1, ws)
	a := newActs(1, maxCols)
	out := make([]float32, maxRows)
	t0 := time.Now()
	for _, w := range ws {
		a.k, a.n = w.cols, 1
		a.quantize(m.pool, w)
		m.matmul(m.pool, a, 0, mmSeg{w, out, w.rows})
	}
	cpuT := time.Since(t0)
	gb := func(d time.Duration) float64 { return float64(bytes) / d.Seconds() / 1e9 }
	t.Logf("decode step (%d matrices, %d MB): GPU %v (%.0f GB/s), CPU %v (%.0f GB/s, %d threads)",
		len(ws), bytes>>20, gpuT.Round(time.Microsecond), gb(gpuT), cpuT.Round(time.Microsecond), gb(cpuT), m.pool.n)
	layers := ws[:len(ws)-1]
	if !b.supports(m.out.typ) {
		layers = ws
	}
	var flops float64
	for _, w := range layers {
		flops += 2 * float64(w.rows) * float64(w.cols) * prompt
	}
	d := timeIt(prompt, layers)
	t.Logf("%d-token prompt, layer matrices: %v (%.1f TFLOPS)", prompt, d.Round(time.Millisecond), flops/d.Seconds()/1e12)
}

func randFloats(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

func floatsOf(b []byte, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = getFloat(b, i)
	}
	return x
}

func maxRelDiff(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	var worst float64
	for i := range want {
		e := math.Abs(float64(got[i]-want[i])) / math.Max(1, math.Abs(float64(want[i])))
		worst = max(worst, e)
	}
	t.Logf("%s: worst error %.1e", what, worst)
	if worst > tol {
		t.Errorf("%s: error %.2e > %.0e", what, worst, tol)
	}
}

// TestGPUElementwise checks rmsnorm (whole rows and per-head), RoPE, SwiGLU and add against the CPU code.
func TestGPUElementwise(t *testing.T) {
	b := testBackend(t)
	r := rand.New(rand.NewPCG(7, 8))
	buf := func(x []float32) gpuRef { return upload(t, b, x) }

	// rmsnorm: 3 rows of 2048 (hidden state), then 3*16 rows of 128 (QK-norm per head), in place.
	for _, c := range []struct{ rows, n int }{{3, 2048}, {48, 128}} {
		x, w := randFloats(r, c.rows*c.n), randFloats(r, c.n)
		xb, wb := buf(x), buf(w)
		bt := b.batch()
		bt.rmsnorm(xb, wb, xb, c.rows, c.n, 1e-6)
		submit(t, bt)
		want := make([]float32, len(x))
		for i := 0; i < c.rows; i++ {
			rmsnorm(want[i*c.n:(i+1)*c.n], x[i*c.n:(i+1)*c.n], w, 1e-6)
		}
		maxRelDiff(t, fmt.Sprintf("rmsnorm %dx%d", c.rows, c.n), download(b, xb, len(x)), want, 1e-5)
	}

	// RoPE: 3 tokens x 16 heads x 128.
	const tokens, heads, hd = 3, 16, 128
	x := randFloats(r, tokens*heads*hd)
	cs, sn := make([]float32, tokens*hd/2), make([]float32, tokens*hd/2)
	for t := 0; t < tokens; t++ {
		for i := 0; i < hd/2; i++ {
			th := float64(40+t) * math.Pow(1e6, -2*float64(i)/hd)
			cs[t*hd/2+i], sn[t*hd/2+i] = float32(math.Cos(th)), float32(math.Sin(th))
		}
	}
	xb, cb, sb := buf(x), buf(cs), buf(sn)
	bt := b.batch()
	bt.rope(xb, cb, sb, tokens, heads, hd)
	submit(t, bt)
	want := append([]float32(nil), x...)
	for tk := 0; tk < tokens; tk++ {
		for h := 0; h < heads; h++ {
			v := want[(tk*heads+h)*hd : (tk*heads+h+1)*hd]
			rope(v, cs[tk*hd/2:(tk+1)*hd/2], sn[tk*hd/2:(tk+1)*hd/2])
		}
	}
	maxRelDiff(t, "rope", download(b, xb, len(x)), want, 1e-6)

	// SwiGLU and add over 6144*3.
	n := 3 * 6144
	g, up := randFloats(r, n), randFloats(r, n)
	gb, ub, ob := buf(g), buf(up), buf(make([]float32, n))
	bt = b.batch()
	bt.swiglu(gb, ub, ob, n)
	bt.add(gb, ub, n)
	submit(t, bt)
	sw, ad := make([]float32, n), make([]float32, n)
	for i := range sw {
		sw[i] = g[i] / (1 + expf(-g[i])) * up[i]
		ad[i] = g[i] + up[i]
	}
	maxRelDiff(t, "swiglu", download(b, ob, n), sw, 1e-5)
	maxRelDiff(t, "add", download(b, gb, n), ad, 0)
}

// TestGPUAttention checks attention over the f32 cache against a float64 reference, with and without a
// packed-sequence mask, for a batch of tokens appended after earlier positions (GQA: 16 query heads on 8 KV heads).
// The second case is a short batch over a long context, which runs split attention.
func TestGPUAttention(t *testing.T) {
	b := testBackend(t)
	for _, c := range []struct{ pos, n int }{{37, 5}, {700, 3}, {90, 45}} {
		t.Run(fmt.Sprintf("pos=%d,n=%d", c.pos, c.n), func(t *testing.T) { testGPUAttention(t, b, c.pos, c.n) })
	}
}

func testGPUAttention(t *testing.T, b gpuBackend, pos, n int) {
	r := rand.New(rand.NewPCG(11, 12))
	const heads, kvHeads, hd = 16, 8, 128
	split := gpuAttention{n: n, pos: pos, headDim: hd, ctxP: (pos + n + 15) &^ 15}
	_, splits, ok := split.split()
	t.Logf("split attention: %v (%d splits)", ok, splits)
	ctxP := (pos + n + 15) &^ 15
	kvd := kvHeads * hd
	q := randFloats(r, n*heads*hd)
	kT := randFloats(r, kvHeads*hd*ctxP)
	v := randFloats(r, ctxP*kvd)
	buf := func(x []float32) gpuRef { return upload(t, b, x) }
	qb, kb, vb := buf(q), buf(kT), buf(v)
	ob, sb := buf(make([]float32, n*heads*hd)), buf(make([]float32, n*heads*ctxP))
	for _, masked := range []bool{false, true} {
		mask := make([]byte, 8*n)
		lo, hi := make([]int, n), make([]int, n)
		if masked { // tokens 2.. start a new packed sequence at position pos+2; positions [10, pos+2) are another's
			for tk := 2; tk < n; tk++ {
				lo[tk], hi[tk] = 10, pos+2
				binary.LittleEndian.PutUint32(mask[8*tk:], 10)
				binary.LittleEndian.PutUint32(mask[8*tk+4:], uint32(pos+2))
			}
		}
		mb := zeros(t, b, 2*n)
		b.write(mb, mask)
		bt := b.batch()
		bt.attention(gpuAttention{q: qb, out: ob, k: kb, v: vb, scratch: sb, mask: mb, kvType: typeF32,
			n: n, pos: pos, heads: heads, kvHeads: kvHeads, headDim: hd, ctxP: ctxP})
		submit(t, bt)
		want := make([]float32, n*heads*hd)
		for tk := 0; tk < n; tk++ {
			for h := 0; h < heads; h++ {
				kvh := h / (heads / kvHeads)
				ctx := pos + tk + 1
				sc := make([]float64, ctx)
				mx := math.Inf(-1)
				for j := 0; j < ctx; j++ {
					if j >= lo[tk] && j < hi[tk] {
						sc[j] = math.Inf(-1)
						continue
					}
					for d := 0; d < hd; d++ {
						sc[j] += float64(q[(tk*heads+h)*hd+d]) * float64(kT[(kvh*hd+d)*ctxP+j])
					}
					sc[j] /= math.Sqrt(hd)
					mx = math.Max(mx, sc[j])
				}
				var sum float64
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				for d := 0; d < hd; d++ {
					var o float64
					for j := range sc {
						o += sc[j] / sum * float64(v[j*kvd+kvh*hd+d])
					}
					want[(tk*heads+h)*hd+d] = float32(o)
				}
			}
		}
		maxRelDiff(t, fmt.Sprintf("attention, masked=%v", masked), download(b, ob, len(want)), want, 1e-5)
	}
}

// TestGPUBackends reports which GPU backends open here and why the others don't.
func TestGPUBackends(t *testing.T) {
	for i, open := range gpuOpeners {
		b, err := open(false)
		if err != nil {
			t.Logf("backend %d: %v", i, err)
			continue
		}
		t.Logf("backend %d: %s, unified=%v", i, b.name(), b.unified())
		b.close()
	}
}

// f16Offsets lists every f16 in a block of n F16 values, so randBlocks makes them all finite.
func f16Offsets(n int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = 2 * i
	}
	return o
}

// promptTol is f32 for f32 kernels, or half when n tokens run on half-precision prompt kernels.
func promptTol(b gpuBackend, n int, f32, half float64) float64 {
	if n >= matmulTiledMin && b.halfPrompts() {
		return half
	}
	return f32
}

// skipGPU skips a test for want of a GPU, but fails it when a GPU is there and our kernels don't compile (which
// would otherwise skip every GPU test without a word).
func skipGPU(t *testing.T, err error) {
	t.Helper()
	if strings.Contains(err.Error(), "metal: compile") {
		t.Fatal(err)
	}
	t.Skip(err)
}

// TestGPUGateUp checks the fused gate/up/SwiGLU step against the separate matmuls and swiglu, for each weight type and
// for both the fused (small batch) and unfused (prompt) paths.
func TestGPUGateUp(t *testing.T) {
	b := testBackend(t)
	for _, c := range []struct {
		typ        uint32
		size, bs   int // block bytes and elements
		f16s       []int
		rows, cols int
	}{{typeQ4_K, q4kBytes, qk, []int{0, 2}, 77, 2048}, {typeQ6_K, q6kBytes, qk, []int{208}, 77, 2048},
		{typeF16, 512, 256, f16Offsets(256), 77, 2048}, {typeQ8_0, 34, 32, []int{0}, 77, 2048}} {
		if !b.supports(c.typ) {
			continue
		}
		r := rand.New(rand.NewPCG(7, 8))
		nb := c.cols / c.bs
		wbytes := c.rows * nb * c.size
		file := alignedBytes(2 * wbytes)
		copy(file, randBlocks(r, 2*c.rows*nb, c.size, c.f16s))
		if err := b.loadWeights(file); err != nil {
			t.Fatal(err)
		}
		wg := &weight{typ: c.typ, cols: c.cols, rows: c.rows, rowBytes: nb * c.size, data: file[:wbytes]}
		wu := &weight{typ: c.typ, cols: c.cols, rows: c.rows, rowBytes: nb * c.size, data: file[wbytes:]}
		const n = 12
		x := upload(t, b, randFloats(r, n*c.cols))
		act := zeros(t, b, gpuActBytes(n, c.cols)/4)
		gate, up, want, got := zeros(t, b, n*c.rows), zeros(t, b, n*c.rows), zeros(t, b, n*c.rows), zeros(t, b, n*c.rows)
		for _, nt := range []int{1, 3, n} {
			bt := b.batch()
			bt.matmul(wg, x, gate, nt, act)
			bt.matmul(wu, x, up, nt, act)
			bt.swiglu(gate, up, want, nt*c.rows)
			bt.gateUp(wg, wu, x, gate, up, got, nt, act)
			submit(t, bt)
			maxRelDiff(t, fmt.Sprintf("%s gateUp, %d tokens", typeName(c.typ), nt), download(b, got, nt*c.rows), download(b, want, nt*c.rows), 1e-5)
		}
	}
}

// TestGPUNormRopeKV checks normRopeKV against normRope then storeKV.
func TestGPUNormRopeKV(t *testing.T) {
	b := testBackend(t)
	r := rand.New(rand.NewPCG(9, 10))
	const tokens, heads, hd, ctxP, pos = 3, 4, 128, 32, 5
	kvd := heads * hd
	x0 := randFloats(r, tokens*kvd)
	w := upload(t, b, randFloats(r, hd))
	cs, sn := upload(t, b, randFloats(r, tokens*hd/2)), upload(t, b, randFloats(r, tokens*hd/2))
	xa, xb := upload(t, b, x0), upload(t, b, x0)
	kca, kcb := zeros(t, b, kvd*ctxP), zeros(t, b, kvd*ctxP)
	vdummy, vc := zeros(t, b, tokens*kvd), zeros(t, b, kvd*ctxP)
	bt := b.batch()
	bt.normRope(xa, w, cs, sn, tokens, heads, hd, 1e-6)
	bt.storeKV(xa, vdummy, kca, vc, tokens, pos, kvd, ctxP)
	bt.normRopeKV(xb, w, cs, sn, kcb, tokens, heads, hd, pos, ctxP, 1e-6)
	submit(t, bt)
	maxRelDiff(t, "normRopeKV x", download(b, xb, tokens*kvd), download(b, xa, tokens*kvd), 0)
	maxRelDiff(t, "normRopeKV cache", download(b, kcb, kvd*ctxP), download(b, kca, kvd*ctxP), 0)
}
