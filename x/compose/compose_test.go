package compose

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/internal/tinymodel"
)

// modelPath is $GGUF, or a tiny random-weight Qwen3 written for the test (the plumbing runs in every go test).
func modelPath(t testing.TB, c tinymodel.Config) string {
	t.Helper()
	if p := os.Getenv("GGUF"); p != "" {
		return p
	}
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := tinymodel.Write(p, c); err != nil {
		t.Fatal(err)
	}
	return p
}

func load(t testing.TB, o engine.LoadOptions) *engine.Model {
	t.Helper()
	m, err := engine.Load(modelPath(t, tinymodel.Config{EndTurns: true}), o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// both runs f on a deterministic model and on a plain one (CPU; and the GPU when $GGUF names a model it runs).
func both(t *testing.T, f func(t *testing.T, m *engine.Model)) {
	for _, det := range []bool{true, false} {
		t.Run(fmt.Sprintf("det=%v", det), func(t *testing.T) {
			m := load(t, engine.LoadOptions{Deterministic: det, PromptCache: 64 << 20})
			t.Logf("GPU %q", m.Info().GPU)
			f(t, m)
		})
	}
}

const prompt = "<|im_start|>system\nYou answer questions about short stories, briefly and exactly.<|im_end|>\n<|im_start|>user\nthe quick brown fox jumps over the lazy dog. Which animal is lazy? Answer in one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

func sameFloats(t *testing.T, what string, a, b []float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: %d vs %d values", what, len(a), len(b))
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			t.Fatalf("%s[%d]: %v (engine) vs %v (compose)", what, i, a[i], b[i])
		}
	}
}

func TestScoreSame(t *testing.T) {
	both(t, func(t *testing.T, m *engine.Model) {
		ids := m.Tokenize(prompt + "The dog is lazy, and the fox is quick.")
		for _, b := range []int{0, 7} {
			want, err := m.Score(context.Background(), ids, engine.ScoreOptions{TopLogprobs: 3, Batch: b})
			if err != nil {
				t.Fatal(err)
			}
			got, err := Score(context.Background(), m, ids, engine.ScoreOptions{TopLogprobs: 3, Batch: b})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(want, got) {
				t.Fatalf("batch %d: Score differs", b)
			}
		}
	})
}

func labels() []string {
	ls := []string{"dog", "fox", "cat", "the lazy dog", "nobody"}
	for i := range 25 {
		ls = append(ls, fmt.Sprintf("dog %d fox", i*7))
	}
	return ls
}

func TestClassifySame(t *testing.T) {
	both(t, func(t *testing.T, m *engine.Model) {
		ids := m.Tokenize(prompt)
		for _, o := range []engine.ClassifyOptions{{}, {Complete: true, Normalize: true, CacheScope: "x"}, {CacheScope: "x"}} {
			// warm the prompt cache, so both calls below restore the prompt and read its last token alone (outside
			// deterministic mode a GPU's batch and one-token passes round differently)
			if _, err := m.ClassifyTokens(context.Background(), ids, labels(), o); err != nil {
				t.Fatal(err)
			}
			want, err := m.ClassifyTokens(context.Background(), ids, labels(), o)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Classify(context.Background(), m, ids, labels(), o)
			if err != nil {
				t.Fatal(err)
			}
			sameFloats(t, "Logprobs", want.Logprobs, got.Logprobs)
			sameFloats(t, "Probs", want.Probs, got.Probs)
			if !reflect.DeepEqual(want.Tokens, got.Tokens) || want.PromptTokens != got.PromptTokens {
				t.Fatalf("token counts differ")
			}
			// both read the prompt after the engine's call stored it, so both find it cached
			if o.CacheScope != "" && got.CachedTokens == 0 {
				t.Errorf("compose Classify didn't use the prompt cache")
			}
		}
	})
}

// Packed branches against one label at a time, rewinding: bit for bit in deterministic mode.
func TestClassifyRewind(t *testing.T) {
	m := load(t, engine.LoadOptions{Deterministic: true})
	ids := m.Tokenize(prompt)
	want, err := m.ClassifyTokens(context.Background(), ids, labels(), engine.ClassifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ClassifyRewind(context.Background(), m, ids, labels())
	if err != nil {
		t.Fatal(err)
	}
	sameFloats(t, "Logprobs", want.Logprobs, got)
}

func TestGenerateSame(t *testing.T) {
	digits := func(s string) engine.Fit { // a number of 1-6 digits, then done
		if len(s) > 6 || strings.Trim(s, "0123456789") != "" {
			return engine.FitNo
		}
		switch len(s) {
		case 0:
			return engine.FitPrefix
		case 6:
			return engine.FitDone
		}
		return engine.FitComplete
	}
	six := func(s string) engine.Fit { // exactly six digits: end of turn is never allowed, FitDone ends it
		if f := digits(s); f == engine.FitComplete {
			return engine.FitPrefix
		} else {
			return f
		}
	}
	afterDay := func(s string) engine.Fit { // free until "Day" (or ":"), then six digits
		i := strings.Index(s, "Day")
		if i < 0 {
			if i = strings.Index(s, ":"); i < 0 {
				return engine.FitFree
			}
			return six(s[i+1:])
		}
		return six(s[i+3:])
	}
	opts := []engine.GenerateOptions{
		{MaxTokens: 24},
		{MaxTokens: 24, Temperature: 0.9, TopK: 40, TopP: 0.9, Seed: 7, Logprobs: true, TopLogprobs: 2},
		{MaxTokens: 40, Temperature: 1.2, Seed: 3, Stop: []string{"e", "lazy"}},
		{MaxTokens: 24, Constraint: digits},
		{MaxTokens: 24, Temperature: 1, Seed: 11, Constraint: digits, Logprobs: true},
		{MaxTokens: 30, Temperature: 1, Seed: 5, Constraint: afterDay, Stop: []string{"zz"}},
		{MaxTokens: 30, Constraint: afterDay, Logprobs: true},
		{MaxTokens: 30, Temperature: 0.8, Seed: 9, Constraint: six, Stop: []string{"7"}},
		{MaxTokens: 30, Temperature: 2, TopK: 5, Seed: 2, Constraint: six, IgnoreEOS: true},
		{MaxTokens: 24, IgnoreEOS: true, Temperature: 0.7, Seed: 1},
	}
	both(t, func(t *testing.T, m *engine.Model) {
		ids := m.Tokenize(prompt)
		if _, err := m.GenerateTokens(context.Background(), ids, engine.GenerateOptions{MaxTokens: 1}, nil); err != nil {
			t.Fatal(err) // warm the prompt cache (see TestClassifySame)
		}
		for i, o := range opts {
			for _, stopAt := range []int{-1, 2} { // stream everything; stop the stream at the 3rd piece
				var wantP, gotP []string
				var wantL, gotL []engine.TokenLogprob
				o.OnLogprob = func(l engine.TokenLogprob) { wantL = append(wantL, l) }
				want, err := m.GenerateTokens(context.Background(), ids, o, func(p string) bool {
					wantP = append(wantP, p)
					return len(wantP) != stopAt+1
				})
				if err != nil {
					t.Fatal(err)
				}
				o.OnLogprob = func(l engine.TokenLogprob) { gotL = append(gotL, l) }
				got, err := Generate(context.Background(), m, ids, o, func(p string) bool {
					gotP = append(gotP, p)
					return len(gotP) != stopAt+1
				})
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("opts %d stop %d: %s %d tokens %q", i, stopAt, want.StopReason, len(want.Tokens), want.Text)
				if want.Text != got.Text || !reflect.DeepEqual(want.Tokens, got.Tokens) || want.StopReason != got.StopReason ||
					!reflect.DeepEqual(want.Logprobs, got.Logprobs) || !reflect.DeepEqual(wantP, gotP) || !reflect.DeepEqual(wantL, gotL) {
					t.Fatalf("options %d, stop %d:\nengine  %q %v %s %q\ncompose %q %v %s %q", i, stopAt,
						want.Text, want.Tokens, want.StopReason, wantP, got.Text, got.Tokens, got.StopReason, gotP)
				}
			}
		}
	})
}

// A fork, a rewind, and a saved and loaded state continue bit for bit like a state that read the same tokens straight.
func TestForkRewindSaveExact(t *testing.T) {
	m := load(t, engine.LoadOptions{Deterministic: true})
	ctx := context.Background()
	ids := m.Tokenize(prompt)
	more := m.Tokenize("The dog is lazy, and the fox is quick.")
	straight, err := m.NewState(engine.StateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer straight.Close()
	want, err := straight.Step(ctx, append(append([]int(nil), ids...), more...)...)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)

	base, _ := m.NewState(engine.StateOptions{})
	defer base.Close()
	if _, err := base.Step(ctx, ids...); err != nil {
		t.Fatal(err)
	}
	check := func(what string, st *engine.State) {
		t.Helper()
		got, err := st.Step(ctx, more...)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				t.Fatalf("%s: logit %d is %v, want %v", what, i, got[i], want[i])
			}
		}
	}
	f, err := base.Fork()
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var snap bytes.Buffer
	if err := base.Save(&snap, engine.SnapshotOptions{Full: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := base.Step(ctx, m.Tokenize(" something else entirely")...); err != nil { // base wanders off
		t.Fatal(err)
	}
	check("fork", f)
	if err := base.Rewind(base.Len() - len(ids)); err != nil {
		t.Fatal(err)
	}
	check("rewind", base)
	ld, err := m.LoadState(ctx, &snap, engine.StateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer ld.Close()
	check("save/load", ld)
}

// Forks step in parallel (go test -race).
func TestForksParallel(t *testing.T) {
	m := load(t, engine.LoadOptions{})
	ctx := context.Background()
	st, _ := m.NewState(engine.StateOptions{MaxContext: 512})
	defer st.Close()
	if _, err := st.Step(ctx, m.Tokenize(prompt)...); err != nil {
		t.Fatal(err)
	}
	const n = 4
	texts := make([]string, n)
	var wg sync.WaitGroup
	for i := range n {
		f, err := st.Fork()
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer f.Close()
			r, err := engine.Decoder{Middleware: []any{engine.MaxTokens(8)}}.Run(ctx, f)
			if err != nil {
				t.Error(err)
			}
			texts[i] = r.Text
		}()
	}
	wg.Wait()
	for i := range texts {
		if texts[i] != texts[0] {
			t.Errorf("greedy forks differ: %q vs %q", texts[i], texts[0])
		}
	}
}

// Beam search of width 1 is greedy decoding; a wider beam finds a continuation at least as likely.
func TestBeam(t *testing.T) {
	m := load(t, engine.LoadOptions{Deterministic: true})
	ctx := context.Background()
	ids := m.Tokenize(prompt)
	st, _ := m.NewState(engine.StateOptions{MaxContext: 512})
	defer st.Close()
	if _, err := st.Step(ctx, ids...); err != nil {
		t.Fatal(err)
	}
	g, err := m.GenerateTokens(ctx, ids, engine.GenerateOptions{MaxTokens: 12}, nil)
	if err != nil {
		t.Fatal(err)
	}
	toks, lp1, err := Beam(ctx, st, 1, 12)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(toks, g.Tokens) {
		t.Fatalf("width 1: %v, greedy %v", toks, g.Tokens)
	}
	if st.Len() != len(ids) {
		t.Fatalf("Beam changed its state")
	}
	toks4, lp4, err := Beam(ctx, st, 4, 12)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("width 1: %.3f %q; width 4: %.3f %q", lp1, m.Detokenize(toks), lp4, m.Detokenize(toks4))
	if len(toks4) == len(toks) && lp4 < lp1-1e-9 {
		t.Errorf("width 4 found %.4f, worse than greedy's %.4f", lp4, lp1)
	}
}

// Best-of-N with the model's own average log-probability as the scorer.
func TestBestOfN(t *testing.T) {
	m := load(t, engine.LoadOptions{Deterministic: true})
	ctx := context.Background()
	ids := m.Tokenize(prompt)
	st, _ := m.NewState(engine.StateOptions{MaxContext: 512})
	defer st.Close()
	if _, err := st.Step(ctx, ids...); err != nil {
		t.Fatal(err)
	}
	recs := map[int]*engine.RecordLogprobs{}
	newDec := func(i int) engine.Decoder {
		recs[i] = &engine.RecordLogprobs{}
		return engine.Decoder{Sampler: engine.Sampling{Temperature: 1}, Seed: uint64(i),
			Middleware: []any{engine.EOS{}, recs[i], engine.MaxTokens(10)}}
	}
	i := 0
	score := func(r engine.DecodeResult) float64 {
		defer func() { i++ }()
		var sum float64
		for _, l := range recs[i].Logprobs {
			sum += l.Logprob.Logprob
		}
		return sum / float64(max(1, len(recs[i].Logprobs)))
	}
	best, s, err := BestOfN(ctx, st, 5, newDec, score)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("best of 5: %.3f %q", s, best.Text)
	if math.IsInf(s, -1) || st.Len() != len(ids) {
		t.Fatalf("no answer, or the state changed")
	}
}
