package engine

import (
	"math"
	"testing"
	"time"
)

func detok(m *Model, ids []int) string {
	var b []byte
	for _, id := range ids {
		b = append(b, m.tok.tokenBytes(id)...)
	}
	return string(b)
}

// logitDiff is the largest |a-b| relative to a's spread (max-min).
func logitDiff(a, b []float32) float64 {
	lo, hi := a[0], a[0]
	var worst float64
	for i := range a {
		lo, hi = min(lo, a[i]), max(hi, a[i])
		worst = max(worst, math.Abs(float64(a[i]-b[i])))
	}
	return worst / float64(hi-lo)
}

// TestGPUForward checks gpuModel.forward against the CPU. The GPU keeps activations in f32, so it is held tightly
// to the CPU's float reference (floatRef) on the prompt; the fast CPU path quantizes activations to 8 bits, which
// moves logits by about 1% of their spread, so against it the test checks the top token and a loose bound, while
// decoding 8 tokens. The GPU reads the prompt in two batches, so the second starts mid-cache.
func TestGPUForward(t *testing.T) {
	m, err := Load(testGGUF(t), LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := newGPUModel(m)
	if err != nil {
		t.Skip(err)
	}
	defer g.close()
	t.Logf("%s", g.b.name())

	ids := m.Tokenize(ChatPrompt([]Message{{Role: "user", Content: "Name three rivers in Europe and say which country each one flows through."}}))
	sc, sg := m.newState(512, 64), m.newState(512, 64)
	m.forward(sc, ids, 0)
	sf := m.newState(512, 64)
	floatRef = true
	m.forward(sf, ids, 0)
	floatRef = false
	split := len(ids) * 2 / 3
	if err := g.forward(sg, ids[:split], 0); err != nil {
		t.Fatal(err)
	}
	if err := g.forward(sg, ids[split:], split); err != nil {
		t.Fatal(err)
	}
	d := logitDiff(sf.logits, sg.logits)
	t.Logf("prompt: GPU vs float reference %.1e of the logits' spread (fast CPU path: %.1e)", d, logitDiff(sf.logits, sc.logits))
	if d > 1e-4 {
		t.Errorf("GPU vs float reference: %.2e", d)
	}
	check := func(step string) int {
		d := logitDiff(sc.logits, sg.logits)
		a, b := argmax(sc.logits), argmax(sg.logits)
		t.Logf("%s: logits differ by %.1e of their spread; top token CPU %d %q, GPU %d", step, d, a, detok(m, []int{a}), b)
		if d > 5e-2 || a != b {
			t.Errorf("%s: GPU disagrees (diff %.2e, top %d vs %d)", step, d, a, b)
		}
		return a
	}
	tok := check("prompt")
	out := []int{tok}
	for i := 0; i < 8; i++ {
		p := len(ids) + i
		m.forward(sc, []int{tok}, p)
		if err := g.forward(sg, []int{tok}, p); err != nil {
			t.Fatal(err)
		}
		tok = check("decode")
		out = append(out, tok)
	}
	t.Logf("text: %q", detok(m, out))

	// The KV cache the GPU wrote is the state's own memory: the CPU sees the same keys.
	// (positions 0..len(ids)+8 of every key channel in layer 0; the CPU's keys carry its activation rounding)
	var kd, kmax float64
	for i := 0; i < len(sc.kc[0]); i++ {
		kd = max(kd, math.Abs(float64(sc.kc[0][i]-sg.kc[0][i])))
		kmax = max(kmax, math.Abs(float64(sc.kc[0][i])))
	}
	t.Logf("layer-0 key cache, CPU vs GPU: max diff %.1e (largest key %.1f)", kd, kmax)
	if kd > 0.05*kmax {
		t.Errorf("key caches differ by %.2e", kd)
	}
}

// TestGPUForwardSpeed times decode and a 512-token prompt on the GPU and the CPU.
func TestGPUForwardSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	m, err := Load(testGGUF(t), LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := newGPUModel(m)
	if err != nil {
		t.Skip(err)
	}
	defer g.close()
	prompt := make([]int, 512)
	for i := range prompt {
		prompt[i] = 1000 + i%5000
	}
	run := func(name string, fwd func(s *state, toks []int, pos int)) {
		s := m.newState(1024, 512)
		fwd(s, prompt[:16], 0) // warm up
		t0 := time.Now()
		fwd(s, prompt, 0)
		pre := time.Since(t0)
		t0 = time.Now()
		const dec = 32
		for i := 0; i < dec; i++ {
			fwd(s, []int{argmax(s.logits)}, len(prompt)+i)
		}
		d := time.Since(t0)
		t.Logf("%s: prompt %d tokens in %v (%.0f tok/s); decode %.1f tok/s", name, len(prompt), pre.Round(time.Millisecond),
			float64(len(prompt))/pre.Seconds(), dec/d.Seconds())
	}
	run("GPU "+g.b.name(), func(s *state, toks []int, pos int) {
		if err := g.forward(s, toks, pos); err != nil {
			t.Fatal(err)
		}
	})
	run("CPU", m.forward)
}
