package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
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

// checkMatvec runs u.matvecDispatch for 1 and n tokens of random input and checks it against a float64
// reference over the dequantized weights. It returns the worst error relative to the row's magnitude.
func checkMatvec(t *testing.T, u *gpu, w *weight, r *rand.Rand) {
	t.Helper()
	const n = 5
	x := make([]float32, n*w.cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	xb, _ := u.dev.NewBuffer(4 * len(x))
	ob, _ := u.dev.NewBuffer(4 * n * w.rows)
	defer xb.Release()
	defer ob.Release()
	putFloats(xb.Bytes(), x)
	tmp := make([]float32, w.cols)
	for _, nt := range []int{1, n} {
		if err := u.dev.Run(u.matvecDispatch(w, xb, ob, nt)); err != nil {
			t.Fatal(err)
		}
		var worst float64
		for tk := 0; tk < nt; tk++ {
			for row := 0; row < w.rows; row++ {
				dequantRow(w.typ, w.row(row), tmp)
				var ref, mag float64
				for i, v := range tmp {
					p := float64(v) * float64(x[tk*w.cols+i])
					ref += p
					mag += math.Abs(p)
				}
				got := float64(getFloat(ob.Bytes(), tk*w.rows+row))
				if e := math.Abs(got-ref) / max(mag, 1e-6); e > worst {
					worst = e
				}
			}
		}
		t.Logf("%s %dx%d, %d token(s): worst error %.1e of the row's magnitude", typeName(w.typ), w.rows, w.cols, nt, worst)
		if worst > 1e-5 {
			t.Errorf("%s %dx%d, %d tokens: error %.2e", typeName(w.typ), w.rows, w.cols, nt, worst)
		}
	}
}

// TestGPUMatvecQ4KSynthetic checks the Q4_K kernel on random blocks (the default models are Q8_0).
func TestGPUMatvecQ4KSynthetic(t *testing.T) {
	dev, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	ps, err := dev.Compile(gpuSource, gpuKernels...)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(5, 6))
	const rows, nb = 77, 8
	blocks := randBlocks(r, rows*nb, q4kBytes, []int{0, 2})
	wb, _ := dev.NewBuffer(len(blocks))
	copy(wb.Bytes(), blocks)
	u := &gpu{dev: dev, ps: ps, weights: wb, base: uintptr(unsafe.Pointer(unsafe.SliceData(wb.Bytes())))}
	defer u.close()
	checkMatvec(t, u, &weight{typ: typeQ4_K, cols: nb * qk, rows: rows, rowBytes: nb * q4kBytes, data: wb.Bytes()}, r)
}

// TestGPUMatvec checks the matvec kernels on a real model's weights, wrapped zero-copy from the
// mmapped file, and compares one decode step's matrix work on the GPU with the CPU.
func TestGPUMatvec(t *testing.T) {
	m, err := Load(testGGUF(t), LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	u, err := openGPU(m.g)
	if err != nil {
		t.Skip(err)
	}
	defer u.close()
	t.Logf("device %s, file %d MB wrapped without copying", u.dev.Name, len(m.g.data)>>20)

	var ws []*weight
	for _, l := range m.layers {
		for _, w := range []*weight{l.wq, l.wk, l.wv, l.wo, l.wgate, l.wup, l.wdown} {
			if gpuMatvec(w.typ) {
				ws = append(ws, w)
			}
		}
	}
	if gpuMatvec(m.out.typ) {
		ws = append(ws, m.out)
	}
	if len(ws) == 0 {
		t.Skip("no weights of a type the GPU handles")
	}
	r := rand.New(rand.NewPCG(3, 4))
	l := m.layers[len(m.layers)-1]
	for _, w := range []*weight{l.wk, l.wdown} {
		if gpuMatvec(w.typ) {
			checkMatvec(t, u, w, r)
		}
	}

	// Speed: every layer matrix plus the output projection once, as in one decode step.
	maxRows, maxCols, bytes := 0, 0, 0
	for _, w := range ws {
		maxRows, maxCols = max(maxRows, w.rows), max(maxCols, w.cols)
		bytes += len(w.data)
	}
	xb, _ := u.dev.NewBuffer(4 * maxCols)
	ob, _ := u.dev.NewBuffer(4 * maxRows)
	defer xb.Release()
	defer ob.Release()
	ds := make([]metal.Dispatch, len(ws))
	for i, w := range ws {
		ds[i] = u.matvecDispatch(w, xb, ob, 1)
	}
	const iters = 10
	_ = u.dev.Run(ds...)
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if err := u.dev.Run(ds...); err != nil {
			t.Fatal(err)
		}
	}
	gpuT := time.Since(t0) / iters

	a := newActs(1, maxCols)
	out := make([]float32, maxRows)
	cpu := func() {
		for _, w := range ws {
			a.k, a.n = w.cols, 1
			a.quantize(m.pool, w)
			m.matmul(m.pool, a, 0, mmSeg{w, out, w.rows})
		}
	}
	cpu()
	t0 = time.Now()
	for i := 0; i < iters; i++ {
		cpu()
	}
	cpuT := time.Since(t0) / iters
	gb := func(d time.Duration) float64 { return float64(bytes) / d.Seconds() / 1e9 }
	t.Logf("decode-step matrices (%d, %d MB): GPU %v (%.0f GB/s), CPU %v (%.0f GB/s, %d threads)",
		len(ws), bytes>>20, gpuT.Round(time.Microsecond), gb(gpuT), cpuT.Round(time.Microsecond), gb(cpuT), m.pool.n)
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
	dev, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	ps, err := dev.Compile(gpuSource, gpuKernels...)
	if err != nil {
		t.Fatal(err)
	}
	u := &gpu{dev: dev, ps: ps}
	defer dev.Close()
	r := rand.New(rand.NewPCG(7, 8))
	buf := func(x []float32) *metal.Buffer {
		b, _ := dev.NewBuffer(4 * len(x))
		putFloats(b.Bytes(), x)
		return b
	}

	// rmsnorm: 3 rows of 2048 (hidden state), then 3*16 rows of 128 (QK-norm per head), in place.
	for _, c := range []struct{ rows, n int }{{3, 2048}, {48, 128}} {
		x, w := randFloats(r, c.rows*c.n), randFloats(r, c.n)
		xb, wb := buf(x), buf(w)
		if err := dev.Run(u.rmsnormDispatch(metal.Arg{Buf: xb}, metal.Arg{Buf: wb}, metal.Arg{Buf: xb}, c.rows, c.n, 1e-6)); err != nil {
			t.Fatal(err)
		}
		want := make([]float32, len(x))
		for i := 0; i < c.rows; i++ {
			rmsnorm(want[i*c.n:(i+1)*c.n], x[i*c.n:(i+1)*c.n], w, 1e-6)
		}
		maxRelDiff(t, fmt.Sprintf("rmsnorm %dx%d", c.rows, c.n), floatsOf(xb.Bytes(), len(x)), want, 1e-5)
		xb.Release()
		wb.Release()
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
	if err := dev.Run(u.ropeDispatch(metal.Arg{Buf: xb}, metal.Arg{Buf: cb}, metal.Arg{Buf: sb}, tokens, heads, hd)); err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), x...)
	for tk := 0; tk < tokens; tk++ {
		for h := 0; h < heads; h++ {
			v := want[(tk*heads+h)*hd : (tk*heads+h+1)*hd]
			rope(v, cs[tk*hd/2:(tk+1)*hd/2], sn[tk*hd/2:(tk+1)*hd/2])
		}
	}
	maxRelDiff(t, "rope", floatsOf(xb.Bytes(), len(x)), want, 1e-6)

	// SwiGLU and add over 6144*3.
	n := 3 * 6144
	g, up := randFloats(r, n), randFloats(r, n)
	gb, ub, ob := buf(g), buf(up), buf(make([]float32, n))
	if err := dev.Run(u.elementwise("swiglu_f32", n, metal.Arg{Buf: gb}, metal.Arg{Buf: ub}, metal.Arg{Buf: ob}),
		u.elementwise("add_f32", n, metal.Arg{Buf: gb}, metal.Arg{Buf: ub})); err != nil {
		t.Fatal(err)
	}
	sw, ad := make([]float32, n), make([]float32, n)
	for i := range sw {
		sw[i] = g[i] / (1 + expf(-g[i])) * up[i]
		ad[i] = g[i] + up[i]
	}
	maxRelDiff(t, "swiglu", floatsOf(ob.Bytes(), n), sw, 1e-5)
	maxRelDiff(t, "add", floatsOf(gb.Bytes(), n), ad, 0)
}
