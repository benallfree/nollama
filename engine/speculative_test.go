package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
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

// specPair is a target and a draft model to test with, by file.
type specPair struct{ name, target, draft string }

// specPairs are the pairs every test runs on: two different tiny random-weight models (guesses often wrong), the same
// tiny model loaded twice (greedy guesses always right), a pair whose models can end their answers, and, when
// GGUF (target) and DRAFT_GGUF are set, those real models.
func specPairs(t *testing.T) []specPair {
	dir := t.TempDir()
	write := func(name string, c tinymodel.Config) string {
		p := filepath.Join(dir, name+".gguf")
		if err := tinymodel.Write(p, c); err != nil {
			t.Fatal(err)
		}
		return p
	}
	big := write("target", tinymodel.Config{Seed: 1, Dim: 128, Layers: 3})
	small := write("draft", tinymodel.Config{Seed: 2})
	same := write("same", tinymodel.Config{Seed: 1, Dim: 128, Layers: 3})
	endT := write("end-target", tinymodel.Config{Seed: 3, Dim: 96, EndTurns: true})
	endD := write("end-draft", tinymodel.Config{Seed: 4, EndTurns: true})
	ps := []specPair{{"tiny", big, small}, {"tiny-same", big, same}, {"tiny-ends", endT, endD}}
	if tp, dp := os.Getenv("GGUF"), os.Getenv("DRAFT_GGUF"); tp != "" && dp != "" {
		ps = append(ps, specPair{"real", tp, dp})
	}
	return ps
}

func (p specPair) load(t *testing.T, o LoadOptions) (target, draft *Model) {
	if o.Threads == 0 {
		o.Threads = 4
	}
	target, err := Load(p.target, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { target.Close() })
	draft, err = Load(p.draft, LoadOptions{Threads: o.Threads})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { draft.Close() })
	return target, draft
}

// eachPair runs f on every pair, as subtests.
func eachPair(t *testing.T, f func(t *testing.T, p specPair)) {
	for _, p := range specPairs(t) {
		t.Run(p.name, func(t *testing.T) { f(t, p) })
	}
}

// loadPair is the real pair, for the benchmark.
func loadPair(t *testing.T) (target, draft *Model) {
	tp, dp := os.Getenv("GGUF"), os.Getenv("DRAFT_GGUF")
	if tp == "" || dp == "" {
		t.Skip("GGUF (target) and DRAFT_GGUF not set")
	}
	return specPair{"real", tp, dp}.load(t, LoadOptions{})
}

var specPrompts = []string{
	"The old lighthouse keeper climbed the spiral stairs one last time, carrying a lantern and a notebook full of stories he had never told anyone. He began to",
	ChatPrompt([]Message{{Role: "user", Content: "Name three primary colors, comma separated."}}),
	ChatPrompt([]Message{{Role: "user", Content: "Write a Go function that reverses a string."}}),
	ChatPrompt([]Message{{Role: "user", Content: "Hi"}}),
}

// forwardRows' exact rows are bit for bit the logits one-token forwards give; its fast rows are close.
func TestForwardRows(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, _ := p.load(t, LoadOptions{})
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
				if exact && maxDiff != 0 {
					t.Errorf("exact row %d differs from a one-token forward by %.6f", i, maxDiff)
				}
				if !exact && maxDiff > 0.15*maxAbs { // the batch kernels round differently, by about quantization's noise
					t.Errorf("fast row %d: max difference %.4f (largest logit %.2f)", i, maxDiff, maxAbs)
				}
			}
		}
	})
}

func TestCheckDraft(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{})
		if err := CheckDraft(m, d); err != nil {
			t.Fatal(err)
		}
		if err := CheckDraft(m, m); err == nil {
			t.Fatal("a model accepted as its own draft")
		}
	})
}

// With temperature 0 the answer is the target's own, bit for bit, whatever the draft guesses.
func TestSpeculativeGreedyMatches(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{})
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
				if p.name == "tiny-same" && got.Accepted != got.Drafted {
					t.Errorf("prompt %d k=%d: the same model as draft kept %d of %d guesses", i, k, got.Accepted, got.Drafted)
				}
				t.Logf("prompt %d k=%d: %d/%d guesses accepted", i, k, got.Accepted, got.Drafted)
			}
		}
	})
}

// A JSON schema constraint: the same answer with a draft, and it fits the schema.
func TestSpeculativeConstrained(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{})
		ctx := context.Background()
		// {"answer": "yes" | "no"}, written as a hand-rolled constraint (the schema package imports this one)
		allowed := []string{`{"answer": "yes"}`, `{"answer": "no"}`}
		fit := func(text string) Fit {
			for _, a := range allowed {
				if text == a {
					return FitDone
				}
				if strings.HasPrefix(a, text) {
					return FitPrefix
				}
			}
			return FitNo
		}
		pr := ChatPrompt([]Message{{Role: "user", Content: "Is the sky blue? Answer as JSON."}})
		for _, temp := range []float64{0, 0.8} {
			o := GenerateOptions{MaxTokens: 32, Constraint: fit, Temperature: temp, Seed: 5}
			want, err := m.Generate(ctx, pr, o, nil)
			if err != nil {
				t.Fatal(err)
			}
			o.Draft = d
			got, err := m.Generate(ctx, pr, o, nil)
			if err != nil {
				t.Fatal(err)
			}
			if fit(got.Text) != FitDone || got.StopReason != "stop" {
				t.Errorf("temp %v: %q (%s) doesn't complete the constraint", temp, got.Text, got.StopReason)
			}
			if temp == 0 && got.Text != want.Text {
				t.Errorf("greedy: got %q, want %q", got.Text, want.Text)
			}
		}
	})
}

func TestSpeculativeStopAndLength(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{})
		ctx := context.Background()
		pr := specPrompts[0]
		for _, o := range []GenerateOptions{
			{MaxTokens: 7},
			{MaxTokens: 1},
			{MaxTokens: 2},
			{MaxTokens: 64, Stop: []string{"."}},
			{MaxTokens: 64, Stop: []string{" the"}},
			{MaxTokens: 64, Stop: []string{"e"}},
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
			if got.Text != want.Text || got.StopReason != want.StopReason || got.OutputTokens != want.OutputTokens {
				t.Errorf("%v %v: got %q (%s, %d), want %q (%s, %d)", o.MaxTokens, o.Stop, got.Text, got.StopReason,
					got.OutputTokens, want.Text, want.StopReason, want.OutputTokens)
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
		// canceled
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := m.Generate(cctx, pr, GenerateOptions{MaxTokens: 16, Draft: d}, nil); err == nil {
			t.Error("a canceled context answered")
		}
		// too long for the context: the same error as without a draft
		_, err1 := m.Generate(ctx, pr, GenerateOptions{MaxTokens: 1 << 20}, nil)
		_, err2 := m.Generate(ctx, pr, GenerateOptions{MaxTokens: 1 << 20, Draft: d}, nil)
		if err1 == nil || err2 == nil || err1.Error() != err2.Error() {
			t.Errorf("context errors: %v / %v", err1, err2)
		}
	})
}

func TestSpeculativeSampling(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{})
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
		if a.StopReason != "eos" && (a.OutputTokens == 0 || (a.OutputTokens > 1 && a.Drafted == 0)) {
			t.Fatalf("nothing generated or drafted: %+v", a)
		}
		t.Logf("%d/%d guesses accepted", a.Accepted, a.Drafted)
	})
}

// Logprobs aren't computed by the check: asking for them answers without the draft, the same as with none.
func TestSpeculativeLogprobsFallBack(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{})
		ctx := context.Background()
		want, err := m.Generate(ctx, specPrompts[1], GenerateOptions{MaxTokens: 12, Logprobs: true}, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.Generate(ctx, specPrompts[1], GenerateOptions{MaxTokens: 12, Logprobs: true, Draft: d}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got.Text != want.Text || len(got.Logprobs) != len(want.Logprobs) || got.Drafted != 0 {
			t.Errorf("got %q (%d logprobs, %d drafted), want %q (%d)", got.Text, len(got.Logprobs), got.Drafted, want.Text, len(want.Logprobs))
		}
	})
}

// A conversation kept in a thread goes the same with a draft as without one.
func TestSpeculativeThread(t *testing.T) {
	eachPair(t, func(t *testing.T, p specPair) {
		m, d := p.load(t, LoadOptions{ThreadMemory: 64 << 20})
		ctx := context.Background()
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
	})
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
