package engine

// Determinism probes. Not part of the normal suite: each test skips unless its env vars are set.
//
//	DET_OUT=file GGUF=model DET_TEXT=file DET_THREADS=n DET_MODE=serial|batch|batchtail  go test -run TestDetDump
//	DET_A=file DET_B=file go test -run TestDetCompare
//	DET_KERNELS=1 go test -run TestDetKernels

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"
)

const detN = 64 // positions dumped

// TestDetDump writes, for each of detN positions, the full logits (float32 bits) the model gives after reading the
// text's tokens up to that position. serial reads one token per forward (decode, what a decompressor must do);
// batch reads the prefix in one forward per position (prefill); batchtail prefills the first half in one batch, then
// decodes the rest one token at a time.
func TestDetDump(t *testing.T) {
	out := os.Getenv("DET_OUT")
	if out == "" {
		t.Skip("DET_OUT not set")
	}
	th, _ := strconv.Atoi(os.Getenv("DET_THREADS"))
	m, err := Load(os.Getenv("GGUF"), LoadOptions{Threads: th})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if os.Getenv("DET_PERTURB") != "" { // one ulp on one weight of layer 0's attention norm: how far does it spread?
		w := m.layers[0].attnNorm
		w[0] = math.Nextafter32(w[0], float32(math.Inf(1)))
	}
	txt, err := os.ReadFile(os.Getenv("DET_TEXT"))
	if err != nil {
		t.Fatal(err)
	}
	ids := m.Tokenize(string(txt))
	if len(ids) < detN {
		t.Fatalf("text too short: %d tokens", len(ids))
	}
	ids = ids[:detN]
	mode := os.Getenv("DET_MODE")
	V := m.hp.vocab
	buf := make([]byte, 0, detN*V*4)
	put := func(l []float32) {
		for _, v := range l {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
	}
	switch mode {
	case "serial":
		s := m.newState(detN+1, 1)
		for i := range ids {
			m.forward(s, ids[i:i+1], i)
			put(s.logits)
		}
	case "batch":
		for i := range ids {
			s := m.newState(detN+1, i+1)
			m.forward(s, ids[:i+1], 0)
			put(s.logits)
		}
	case "batchtail":
		h := detN / 2
		s := m.newState(detN+1, h)
		for i := 0; i < h-1; i++ { // positions before h-1 aren't available from one batch: pad with serial runs
			s2 := m.newState(detN+1, 1)
			for j := 0; j <= i; j++ {
				m.forward(s2, ids[j:j+1], j)
			}
			put(s2.logits)
		}
		m.forward(s, ids[:h], 0)
		put(s.logits)
		for i := h; i < detN; i++ {
			m.forward(s, ids[i:i+1], i)
			put(s.logits)
		}
	default:
		t.Fatalf("DET_MODE %q", mode)
	}
	if err := os.WriteFile(out, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s: kernels %s, threads %d, mode %s, %d positions x %d", out, kernelName, m.pool.n, mode, detN, V)
}

func readDump(t *testing.T, p string) []float32 {
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return f
}

func logSoftmax(l []float32) []float64 {
	mx := float64(math.Inf(-1))
	for _, v := range l {
		mx = math.Max(mx, float64(v))
	}
	var s float64
	for _, v := range l {
		s += math.Exp(float64(v) - mx)
	}
	ls := math.Log(s)
	o := make([]float64, len(l))
	for i, v := range l {
		o[i] = float64(v) - mx - ls
	}
	return o
}

// TestDetCompare reports where two dumps first differ and by how much.
func TestDetCompare(t *testing.T) {
	pa, pb := os.Getenv("DET_A"), os.Getenv("DET_B")
	if pa == "" {
		t.Skip("DET_A not set")
	}
	a, b := readDump(t, pa), readDump(t, pb)
	if len(a) != len(b) {
		t.Fatalf("sizes differ")
	}
	V := len(a) / detN
	first := -1
	var maxAbs, maxLP float64
	argmaxFlips, diffPos := 0, 0
	for p := 0; p < detN; p++ {
		la, lb := a[p*V:(p+1)*V], b[p*V:(p+1)*V]
		nd := 0
		var ma float64
		ia, ib := 0, 0
		for i := range la {
			if math.Float32bits(la[i]) != math.Float32bits(lb[i]) {
				nd++
				ma = math.Max(ma, math.Abs(float64(la[i]-lb[i])))
			}
			if la[i] > la[ia] {
				ia = i
			}
			if lb[i] > lb[ib] {
				ib = i
			}
		}
		if nd == 0 {
			continue
		}
		diffPos++
		if ia != ib {
			argmaxFlips++
		}
		sa, sb := logSoftmax(la), logSoftmax(lb)
		var mlp float64
		for i := range sa {
			if sa[i] > -20 { // tokens with p > 2e-9
				mlp = math.Max(mlp, math.Abs(sa[i]-sb[i]))
			}
		}
		if os.Getenv("DET_VERBOSE") != "" {
			t.Logf("  pos %2d: %6d differ, max |dlogit| %.3g, max |dlogprob| %.3g, argmax %v", p, nd, ma, mlp, ia == ib)
		}
		if first < 0 {
			first = p
			t.Logf("first difference at position %d: %d of %d logits differ, max |dlogit| %.3g, max |dlogprob| %.3g", p, nd, V, ma, mlp)
		}
		maxAbs, maxLP = math.Max(maxAbs, ma), math.Max(maxLP, mlp)
	}
	if first < 0 {
		t.Logf("IDENTICAL: all %d positions bit-equal", detN)
		return
	}
	t.Logf("%d of %d positions differ; max |dlogit| %.3g, max |dlogprob| %.3g (p>2e-9), argmax flips %d", diffPos, detN, maxAbs, maxLP, argmaxFlips)
}

// TestDetKernels counts bit mismatches between kernel variants on random data: asm vs Go, and 4-column vs 1-column.
func TestDetKernels(t *testing.T) {
	if os.Getenv("DET_KERNELS") == "" {
		t.Skip("DET_KERNELS not set")
	}
	r := rand.New(rand.NewPCG(7, 9))
	const trials = 2000
	nb := 4 // 1024 elements, Qwen3-0.6B's dim
	report := func(name string, bad int) {
		t.Logf("%-28s %4d / %d differ (kernels %s)", name, bad, trials, kernelName)
	}

	mkq8k := func() []byte {
		x := make([]float32, nb*qk)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		y := make([]byte, nb*q8kBytes)
		quantizeQ8K(x, y)
		return y
	}
	var bad4, badG, bad6, bad6G, bad80G, badF16G, badF16x4, badF32G int
	for i := 0; i < trials; i++ {
		w := randBlocks(r, nb, q4kBytes, []int{0, 2})
		ys := [4][]byte{mkq8k(), mkq8k(), mkq8k(), mkq8k()}
		yy := append(append(append(append([]byte{}, ys[0]...), ys[1]...), ys[2]...), ys[3]...)
		one := dotQ4K(w, ys[1], nb)
		if math.Float32bits(one) != math.Float32bits(dotQ4KQ8KGo(w, ys[1], nb)) {
			badG++
		}
		if dotQ4Kx4 != nil {
			var o [4]float32
			dotQ4Kx4(w, yy, nb*q8kBytes, nb, &o)
			if math.Float32bits(o[1]) != math.Float32bits(one) {
				bad4++
			}
		}
		w6 := randBlocks(r, nb, q6kBytes, []int{208})
		one6 := dotQ6K(w6, ys[2], nb)
		if math.Float32bits(one6) != math.Float32bits(dotQ6KQ8KGo(w6, ys[2], nb)) {
			bad6G++
		}
		if dotQ6Kx4 != nil {
			var o [4]float32
			dotQ6Kx4(w6, yy, nb*q8kBytes, nb, &o)
			if math.Float32bits(o[2]) != math.Float32bits(one6) {
				bad6++
			}
		}
		x := make([]float32, nb*qk)
		for j := range x {
			x[j] = float32(r.NormFloat64())
		}
		q80 := make([]byte, nb*qk/32*q80aBytes)
		quantizeQ8_0(x, q80)
		w8 := randBlocks(r, nb*qk/32, q80Bytes, []int{0})
		if math.Float32bits(dotQ80(w8, q80, nb*qk/32)) != math.Float32bits(dotQ8_0Go(w8, q80, nb*qk/32)) {
			bad80G++
		}
		wf := make([]byte, 2*len(x))
		for j := range x {
			binary.LittleEndian.PutUint16(wf[2*j:], uint16(0x3000+r.IntN(0x1800))|uint16(r.IntN(2))<<15)
		}
		xf := make([]float32, 4*len(x))
		for j := range xf {
			xf[j] = float32(r.NormFloat64())
		}
		f1 := dotF16(wf, xf[len(x):2*len(x)])
		if math.Float32bits(f1) != math.Float32bits(dotF16F32Go(wf, xf[len(x):2*len(x)])) {
			badF16G++
		}
		if dotF16x4 != nil {
			var o [4]float32
			dotF16x4(wf, xf, len(x), &o)
			if math.Float32bits(o[1]) != math.Float32bits(f1) {
				badF16x4++
			}
		}
		if math.Float32bits(dotF32(x, xf[:len(x)])) != math.Float32bits(dotF32Go(x, xf[:len(x)])) {
			badF32G++
		}
	}
	report("Q4_K asm vs Go", badG)
	report("Q4_K x4 vs x1", bad4)
	report("Q6_K asm vs Go", bad6G)
	report("Q6_K x4 vs x1", bad6)
	report("Q8_0 asm vs Go", bad80G)
	report("F16 asm vs Go", badF16G)
	report("F16 x4 vs x1", badF16x4)
	report("F32 asm vs Go", badF32G)

	// pure-Go float helpers whose results depend on FMA fusion: print a checksum to compare across GOARCH
	var hx, hr uint64
	for i := 0; i < 100000; i++ {
		v := float32(r.Float64()*40 - 20)
		hx = hx*1099511628211 ^ uint64(math.Float32bits(expf(v)))
	}
	v := make([]float32, 128)
	cs, sn := make([]float32, 64), make([]float32, 64)
	for i := 0; i < 1000; i++ {
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		for j := range cs {
			cs[j], sn[j] = float32(r.Float64()), float32(r.Float64())
		}
		rope(v, cs, sn)
		for _, x := range v {
			hr = hr*1099511628211 ^ uint64(math.Float32bits(x))
		}
	}
	t.Log(fmt.Sprintf("expf checksum %016x, rope checksum %016x", hx, hr))
}
