package engine

import (
	"math"
	"slices"
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
	m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := newGPUModel(m, false)
	if err != nil {
		skipGPU(t, err)
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
	if d > promptTol(g.b, split, 1e-4, 2e-3) {
		t.Errorf("GPU vs float reference: %.2e", d)
	}
	check := func(step string) int {
		d := logitDiff(sc.logits, sg.logits)
		a, b := argmax(sc.logits), argmax(sg.logits)
		t.Logf("%s: logits differ by %.1e of their spread; top token CPU %d %q, GPU %d", step, d, a, detok(m, []int{a}), b)
		// The CPU's 8-bit path can flip a near tie: a different top token is fine when the CPU itself scores
		// the GPU's pick within the two paths' difference of its own.
		lo, hi := slices.Min(sc.logits), slices.Max(sc.logits)
		tie := float64(sc.logits[a]-sc.logits[b]) <= 2*d*float64(hi-lo)
		if d > 5e-2 || a != b && !tie {
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
	m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := newGPUModel(m, false)
	if err != nil {
		skipGPU(t, err)
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

// TestGPUScoreEmbed checks the Score path (logits for every position) and the Embed path (the last hidden state)
// against the CPU's float reference.
func TestGPUScoreEmbed(t *testing.T) {
	m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := newGPUModel(m, false)
	if err != nil {
		skipGPU(t, err)
	}
	defer g.close()
	ids := m.Tokenize("The quick brown fox jumps over the lazy dog.")
	n, vocab, dim := len(ids), m.hp.vocab, m.hp.dim
	sf, sg := m.newState(64, 32), m.newState(64, 32)
	sf.allLogits, sg.allLogits = make([]float32, 32*vocab), make([]float32, 32*vocab)
	floatRef = true
	m.forward(sf, ids, 0)
	floatRef = false
	if err := g.forward(sg, ids, 0); err != nil {
		t.Fatal(err)
	}
	var worst float64
	for p := 0; p < n; p++ {
		worst = max(worst, logitDiff(sf.allLogits[p*vocab:(p+1)*vocab], sg.allLogits[p*vocab:(p+1)*vocab]))
	}
	t.Logf("Score, %d positions: worst %.1e of the logits' spread vs the float reference", n, worst)
	if worst > promptTol(g.b, n, 1e-4, 5e-3) {
		t.Errorf("Score: %.2e", worst)
	}

	ef, eg := m.newState(64, 32), m.newState(64, 32)
	ef.hidden, eg.hidden = make([]float32, dim), make([]float32, dim)
	floatRef = true
	m.forward(ef, ids, 0)
	floatRef = false
	if err := g.forward(eg, ids, 0); err != nil {
		t.Fatal(err)
	}
	maxRelDiff(t, "Embed hidden state vs float reference", eg.hidden, ef.hidden, promptTol(g.b, n, 1e-3, 5e-2))
}

// TestGPUForwardDiscrete runs the discrete-GPU KV path (device cache, keys and values copied back to kc/vc,
// re-upload when a call doesn't continue the last) on this GPU, in lockstep with the unified path: prompt in two
// batches, decode, then a rewind into the prompt (which forces the upload). Logits and caches must match.
func TestGPUForwardDiscrete(t *testing.T) {
	m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	gu, err := newGPUModel(m, false)
	if err != nil {
		skipGPU(t, err)
	}
	defer gu.close()
	gpuForceDiscrete = true
	gd, err := newGPUModel(m, false)
	gpuForceDiscrete = false
	if err != nil {
		t.Fatal(err)
	}
	defer gd.close()
	if gd.shared {
		t.Fatal("not in discrete mode")
	}
	ids := m.Tokenize(ChatPrompt([]Message{{Role: "user", Content: "Write a haiku about rain on a tin roof."}}))
	su, sd := m.newState(256, 64), m.newState(256, 64)
	step := func(name string, toks []int, pos int) {
		t.Helper()
		if err := gu.forward(su, toks, pos); err != nil {
			t.Fatal(err)
		}
		if err := gd.forward(sd, toks, pos); err != nil {
			t.Fatal(err)
		}
		if d := logitDiff(su.logits, sd.logits); d > 1e-6 {
			t.Errorf("%s: unified vs discrete logits differ by %.2e", name, d)
		}
	}
	split := len(ids) / 2
	step("prompt 1", ids[:split], 0)
	step("prompt 2", ids[split:], split)
	pos := len(ids)
	for i := 0; i < 4; i++ {
		step("decode", []int{argmax(su.logits)}, pos)
		pos++
	}
	// Rewind: re-read the last 3 prompt tokens. The discrete state must upload kc/vc (its device cache
	// holds the decoded tokens too, and the call doesn't continue the last one).
	step("rewind", ids[len(ids)-3:], len(ids)-3)
	step("decode after rewind", []int{argmax(su.logits)}, len(ids))
	var kd float64
	for li := range su.kc {
		for i := range su.kc[li] {
			kd = max(kd, math.Abs(float64(su.kc[li][i]-sd.kc[li][i])), math.Abs(float64(su.vc[li][i]-sd.vc[li][i])))
		}
	}
	t.Logf("caches, unified vs discrete: max diff %.1e", kd)
	if kd > 1e-4 {
		t.Errorf("caches differ by %.2e", kd)
	}
}

// TestGPUSplitAttention decodes over a long context with split attention and with the one-pass kernel.
func TestGPUSplitAttention(t *testing.T) {
	m, err := Load(testGGUF(t), LoadOptions{GPU: "off"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := newGPUModel(m, false)
	if err != nil {
		skipGPU(t, err)
	}
	defer g.close()
	prompt := make([]int, 600)
	for i := range prompt {
		prompt[i] = 1000 + i*7%5000
	}
	decode := func(off bool) []float32 {
		attnSplitOff = off
		defer func() { attnSplitOff = false }()
		s := m.newState(1024, 600)
		for _, f := range []func() error{
			func() error { return g.forward(s, prompt, 0) },
			func() error { return g.forward(s, []int{42}, len(prompt)) },
			func() error { return g.forward(s, []int{43, 44, 45}, len(prompt)+1) }, // a short batch, as speculative decoding sends
		} {
			if err := f(); err != nil {
				t.Fatal(err)
			}
		}
		g.forget(s)
		return append([]float32(nil), s.logits...)
	}
	split, one := decode(false), decode(true)
	d := logitDiff(split, one)
	t.Logf("split vs one-pass attention: logits differ by %.1e of their spread", d)
	if d > 1e-4 {
		t.Fatalf("split attention is off by %.1e", d)
	}
}
