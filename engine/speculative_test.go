package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The accept/reject rule must leave the target's distribution unchanged, whatever the draft proposes.
func TestSpeculativeSamplingPreservesDistribution(t *testing.T) {
	p := tokDist{ids: []int{0, 1, 2, 3}, ps: []float64{0.5, 0.25, 0.15, 0.10}}
	drafts := map[string]tokDist{
		"close":    {ids: []int{0, 1, 2, 3}, ps: []float64{0.4, 0.3, 0.2, 0.1}},
		"far":      {ids: []int{3, 2, 4}, ps: []float64{0.6, 0.3, 0.1}}, // 4 is impossible for the target
		"greedy":   {ids: []int{2}, ps: []float64{1}},
		"same":     p,
		"disjoint": {ids: []int{7, 8}, ps: []float64{0.5, 0.5}},
	}
	const n = 400000
	for name, q := range drafts {
		rng := rand.New(rand.NewPCG(1, 2))
		counts := map[int]int{}
		for i := 0; i < n; i++ {
			g := q.draw(rng)
			pg, qg := p.p(g), q.p(g)
			var tok int
			if pg > 0 && (pg >= qg || rng.Float64() < pg/qg) {
				tok = g
			} else {
				tok = residual(&p, &q, rng)
			}
			counts[tok]++
		}
		for i, id := range p.ids {
			got := float64(counts[id]) / n
			if math.Abs(got-p.ps[i]) > 0.005 {
				t.Errorf("%s: token %d drawn %.4f of the time, want %.4f", name, id, got, p.ps[i])
			}
		}
		for id, c := range counts {
			if p.p(id) == 0 {
				t.Errorf("%s: token %d (impossible for the target) drawn %d times", name, id, c)
			}
		}
	}
}

// dist must describe the distribution sample draws from.
func TestDistMatchesSample(t *testing.T) {
	m := &Model{}
	rng := rand.New(rand.NewPCG(3, 4))
	logits := make([]float32, 300)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 3)
	}
	for _, o := range []GenerateOptions{
		{Temperature: 0},
		{Temperature: 0.7, TopK: 1},
		{Temperature: 0.8, TopK: 40, TopP: 0.9},
		{Temperature: 1.2, TopK: 0, TopP: 0},
		{Temperature: 0.5, TopK: 5},
	} {
		d, ok := m.dist(logits, o, nil)
		if !ok {
			t.Fatal("no distribution")
		}
		var sum float64
		for _, p := range d.ps {
			sum += p
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("%+v: probabilities sum to %v", o, sum)
		}
		const n = 60000
		counts := map[int]int{}
		srng := rand.New(rand.NewPCG(5, 6))
		for i := 0; i < n; i++ {
			counts[m.sample(logits, o, srng)]++
		}
		for id, c := range counts {
			if d.p(id) == 0 {
				t.Fatalf("%+v: sample drew %d, which dist gives no chance", o, id)
			}
			if got, want := float64(c)/n, d.p(id); math.Abs(got-want) > 0.01 {
				t.Errorf("%+v: token %d drawn %.4f of the time, dist says %.4f", o, id, got, want)
			}
		}
	}
}

func loadPair(t *testing.T) (target, draft *Model) {
	tp, dp := os.Getenv("GGUF"), os.Getenv("DRAFT_GGUF")
	if tp == "" || dp == "" {
		t.Skip("GGUF (target) and DRAFT_GGUF not set")
	}
	target, err := Load(tp, LoadOptions{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { target.Close() })
	draft, err = Load(dp, LoadOptions{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { draft.Close() })
	return target, draft
}

var specPrompts = []string{
	"The old lighthouse keeper climbed the spiral stairs one last time, carrying a lantern and a notebook full of stories he had never told anyone. He began to",
	ChatPrompt([]Message{{Role: "user", Content: "Name three primary colors, comma separated."}}),
	ChatPrompt([]Message{{Role: "user", Content: "Write a Go function that reverses a string."}}),
	ChatPrompt([]Message{{Role: "user", Content: "Hi"}}),
}

// forwardRows' exact rows are bit for bit the logits one-token forwards give; its fast rows are close.
func TestForwardRows(t *testing.T) {
	m, _ := loadPair(t)
	ids := m.Tokenize(specPrompts[0])
	head := len(ids) - 6
	V := m.hp.vocab
	ref := m.newState(len(ids)+1, len(ids))
	m.forward(ref, ids[:head], 0)
	want := make([]float32, 6*V)
	for i := 0; i < 6; i++ {
		m.forward(ref, ids[head+i:head+i+1], head+i)
		copy(want[i*V:], ref.logits)
	}
	for _, exact := range []bool{true, false} {
		s := m.newState(len(ids)+1, len(ids))
		m.forward(s, ids[:head], 0)
		rows := make([]float32, 6*V)
		m.forwardRows(s, ids[head:], head, rows, exact)
		for i := 0; i < 6; i++ {
			var maxDiff, maxAbs float64
			for j := i * V; j < (i+1)*V; j++ {
				maxDiff = max(maxDiff, math.Abs(float64(want[j]-rows[j])))
				maxAbs = max(maxAbs, math.Abs(float64(want[j])))
			}
			t.Logf("exact=%v row %d: max difference %.4f (largest logit %.2f)", exact, i, maxDiff, maxAbs)
			if exact && maxDiff != 0 {
				t.Errorf("exact row %d differs from a one-token forward by %.6f", i, maxDiff)
			}
			if !exact && maxDiff > 0.1*maxAbs {
				t.Errorf("fast row %d: max difference %.4f (largest logit %.2f)", i, maxDiff, maxAbs)
			}
		}
	}
}

func TestCheckDraft(t *testing.T) {
	m, d := loadPair(t)
	if err := CheckDraft(m, d); err != nil {
		t.Fatal(err)
	}
	if err := CheckDraft(m, m); err == nil {
		t.Fatal("a model accepted as its own draft")
	}
}

// With temperature 0 the answer is the target's own. Batched and single-token forwards round differently, so a
// near-tie could in principle split them; on these prompts they agree.
func TestSpeculativeGreedyMatches(t *testing.T) {
	m, d := loadPair(t)
	ctx := context.Background()
	for i, pr := range specPrompts {
		want, err := m.Generate(ctx, pr, GenerateOptions{MaxTokens: 48}, nil)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range []int{1, 4, 8} {
			var streamed strings.Builder
			got, err := m.Generate(ctx, pr, GenerateOptions{MaxTokens: 48, Draft: d, DraftTokens: k},
				func(p string) bool { streamed.WriteString(p); return true })
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != want.Text || got.StopReason != want.StopReason || got.OutputTokens != want.OutputTokens {
				t.Errorf("prompt %d k=%d:\n got %q (%s, %d)\nwant %q (%s, %d)", i, k, got.Text, got.StopReason,
					got.OutputTokens, want.Text, want.StopReason, want.OutputTokens)
			}
			if streamed.String() != got.Text {
				t.Errorf("prompt %d k=%d: streamed %q, result %q", i, k, streamed.String(), got.Text)
			}
			t.Logf("prompt %d k=%d: %d/%d guesses accepted", i, k, got.Accepted, got.Drafted)
		}
	}
}

func TestSpeculativeStopAndLength(t *testing.T) {
	m, d := loadPair(t)
	ctx := context.Background()
	pr := specPrompts[0]
	for _, o := range []GenerateOptions{
		{MaxTokens: 7},
		{MaxTokens: 1},
		{MaxTokens: 64, Stop: []string{"."}},
		{MaxTokens: 64, Stop: []string{" the"}},
	} {
		want, err := m.Generate(ctx, pr, o, nil)
		if err != nil {
			t.Fatal(err)
		}
		o.Draft = d
		got, err := m.Generate(ctx, pr, o, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got.Text != want.Text || got.StopReason != want.StopReason {
			t.Errorf("%v %v: got %q (%s), want %q (%s)", o.MaxTokens, o.Stop, got.Text, got.StopReason, want.Text, want.StopReason)
		}
	}
	// a callback that stops early
	n := 0
	res, err := m.Generate(ctx, pr, GenerateOptions{MaxTokens: 64, Draft: d}, func(string) bool { n++; return n < 3 })
	if err != nil {
		t.Fatal(err)
	}
	if res.StopReason != "callback" || n != 3 {
		t.Errorf("callback: reason %s after %d calls", res.StopReason, n)
	}
}

func TestSpeculativeSampling(t *testing.T) {
	m, d := loadPair(t)
	ctx := context.Background()
	o := GenerateOptions{MaxTokens: 64, Temperature: 0.8, TopP: 0.9, TopK: 40, Seed: 7, Draft: d}
	a, err := m.Generate(ctx, specPrompts[2], o, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Generate(ctx, specPrompts[2], o, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text != b.Text {
		t.Errorf("same seed, different answers:\n%q\n%q", a.Text, b.Text)
	}
	if a.OutputTokens == 0 || a.Drafted == 0 {
		t.Fatalf("nothing generated or drafted: %+v", a)
	}
	t.Logf("%d/%d guesses accepted: %q", a.Accepted, a.Drafted, a.Text)
}

// A conversation kept in a thread goes the same with a draft as without one.
func TestSpeculativeThread(t *testing.T) {
	_, d := loadPair(t)
	ctx := context.Background()
	m, err := Load(os.Getenv("GGUF"), LoadOptions{Threads: 4, ThreadMemory: 64 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	chat := func(thread string, draft *Model) (answers []string, cached int) {
		msgs := []Message{
			{Role: "system", Content: "You are a helpful grocer. Answer in one short sentence, naming one item and saying why it is good to eat this season."},
			{Role: "user", Content: "Name a fruit."},
		}
		for turn := 0; turn < 2; turn++ {
			res, err := m.Generate(ctx, ChatPrompt(msgs), GenerateOptions{MaxTokens: 24, Thread: thread, Draft: draft}, nil)
			if err != nil {
				t.Fatal(err)
			}
			answers = append(answers, res.Text)
			cached = res.CachedTokens
			msgs = append(msgs, Message{Role: "assistant", Content: res.Text}, Message{Role: "user", Content: "Another?"})
		}
		return answers, cached
	}
	got, cached := chat("with", d)
	want, _ := chat("without", nil)
	if cached == 0 {
		t.Error("the second turn read nothing from the thread")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("turn %d: with the draft %q, without %q", i, got[i], want[i])
		}
	}
}

// GGUF=<target> DRAFT_GGUF=<draft> SPEC_BENCH=1 go test -run TestSpeculativeBench -v -timeout 90m
// Writing speed with and without the draft, and how many guesses the target keeps. Medians of 3 runs, 128 tokens,
// 4 threads (the README's benchmark setup). Logs the load average with each line: speeds on a busy machine mean little,
// acceptance rates don't depend on it.
func TestSpeculativeBench(t *testing.T) {
	if os.Getenv("SPEC_BENCH") == "" {
		t.Skip("SPEC_BENCH not set")
	}
	m, d := loadPair(t)
	ctx := context.Background()
	load := func() string {
		out, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		return strings.TrimSpace(string(out))
	}
	prompts := map[string]string{
		"story": specPrompts[0],
		"code":  specPrompts[2],
		"json":  ChatPrompt([]Message{{Role: "user", Content: "Return a JSON object describing a book: title, author, year, genres (array), in_print (bool). Make up a plausible book."}}),
		"facts": ChatPrompt([]Message{{Role: "user", Content: "List the planets of the solar system in order, one per line, with one fact about each."}}),
	}
	names := []string{"story", "code", "json", "facts"}
	type cfg struct {
		k    int
		minP float64 // -1: off, 0: the default
		fast bool
	}
	cfgs := []cfg{{0, 0, false}, {4, 0, false}, {4, 0, true}, {8, 0, false}} // no draft; the defaults; fast checks (the same as the defaults when sampling); more guesses
	if ks := os.Getenv("SPEC_K"); ks != "" {                                 // e.g. "4": only no draft and k=4
		k, _ := strconv.Atoi(ks)
		cfgs = []cfg{{0, 0, false}, {k, 0, false}}
	}
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return v[len(v)/2]
	}
	temps := []float64{0, 0.7}
	if os.Getenv("SPEC_GREEDY") != "" {
		temps = temps[:1]
	}
	for _, temp := range temps {
		for _, name := range names {
			pr := prompts[name]
			// runs alternate between the settings, so a change in the machine's load hits them alike
			speeds := make([][]float64, len(cfgs))
			rates := make([][]float64, len(cfgs))
			perCheck := make([][]float64, len(cfgs))
			for run := 0; run < 3; run++ {
				for ci, c := range cfgs {
					o := GenerateOptions{MaxTokens: 128, IgnoreEOS: true, Temperature: temp, TopP: 0.9, TopK: 40, Seed: uint64(run + 1)}
					if c.k > 0 {
						o.Draft, o.DraftTokens, o.DraftFast, o.DraftMinP = d, c.k, c.fast, c.minP
					}
					res, err := m.Generate(ctx, pr, o, nil)
					if err != nil {
						t.Fatal(err)
					}
					// the first token comes from the prompt pass; speed is the rest over the decode time
					speeds[ci] = append(speeds[ci], float64(res.OutputTokens-1)/res.Decode.Seconds())
					if res.Drafted > 0 {
						rates[ci] = append(rates[ci], float64(res.Accepted)/float64(res.Drafted))
					}
					if res.Checks > 0 {
						perCheck[ci] = append(perCheck[ci], float64(res.OutputTokens)/float64(res.Checks))
					}
				}
			}
			base := median(speeds[0])
			t.Logf("temp %.1f %-5s no draft   : %5.1f tok/s  load %s", temp, name, base, load())
			for ci, c := range cfgs[1:] {
				ci++
				label := fmt.Sprintf("k=%d", c.k)
				if c.minP < 0 {
					label += " all"
				}
				if c.fast {
					label += " fast"
				}
				sp := median(speeds[ci])
				t.Logf("temp %.1f %-5s %-11s: %5.1f tok/s (%.2fx), %3.0f%% of guesses kept, %.1f per check  load %s", temp, name, label, sp, sp/base, 100*median(rates[ci]), median(perCheck[ci]), load())
			}
		}
	}
}
