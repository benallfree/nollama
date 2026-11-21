package engine

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
)

func TestSoftmax64(t *testing.T) {
	p := softmax64([]float64{-1, -2, -1000, 0})
	var sum float64
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 || p[3] < p[0] || p[0] < p[1] || p[2] > 1e-300 {
		t.Errorf("softmax: %v (sum %v)", p, sum)
	}
}

// GGUF=<file> go test -run TestClassify -v
func TestClassify(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m, err := Load(p, LoadOptions{Threads: 4, PromptCache: 256 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	labels := []string{"bug", "feature request", "billing question"}
	system := "You label customer support tickets for a software company. A bug is something that is broken or crashes. " +
		"A feature request asks for something new. A billing question is about invoices, charges or refunds. " +
		"Answer with exactly one label: bug, feature request or billing question."
	prompt := ChatPrompt([]Message{
		{Role: "system", Content: system},
		{Role: "user", Content: "Bug report: the app crashes and closes every time I press the Save button. It is broken."},
	})
	multi := false
	for _, l := range labels {
		multi = multi || len(m.Tokenize(l)) > 1
	}
	if !multi {
		t.Fatal("want a multi-token label in the test set")
	}

	for _, o := range []ClassifyOptions{{}, {Normalize: true}, {Complete: true}} {
		res, err := m.Classify(ctx, prompt, labels, o)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, v := range res.Probs {
			sum += v
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Errorf("%+v: probabilities sum to %v", o, sum)
		}
		if res.Best() != 0 {
			t.Errorf("%+v: %v, want bug to win", o, res.Probs)
		}
		t.Logf("%+v: %v (logprobs %v, tokens %v)", o, res.Probs, res.Logprobs, res.Tokens)

		// each label's log-probability must be what Score says for the prompt followed by the label
		ids := m.Tokenize(prompt)
		for i, l := range labels {
			lt := m.Tokenize(l)
			if o.Complete {
				lt = append(lt, m.Tokenize("<|im_end|>")...)
			}
			if res.Tokens[i] != len(lt) {
				t.Errorf("%q: %d tokens, want %d", l, res.Tokens[i], len(lt))
			}
			sc, err := m.Score(ctx, append(append([]int(nil), ids...), lt...), ScoreOptions{})
			if err != nil {
				t.Fatal(err)
			}
			var want float64
			for _, x := range sc[len(ids)-1:] {
				want += x.Logprob.Logprob
			}
			// compared as probabilities: batched and one-token forward passes round differently, by up to a few
			// tenths of a nat on unlikely tokens (TestScore sees it too)
			if d := math.Abs(math.Exp(res.Logprobs[i]) - math.Exp(want)); d > 0.01 {
				t.Errorf("%q: logprob %.4f, Score says %.4f", l, res.Logprobs[i], want)
			}
		}
	}

	// the other way round: a feature request must win too, and the shared prompt start comes from the cache
	prompt2 := ChatPrompt([]Message{
		{Role: "system", Content: system},
		{Role: "user", Content: "Please add a new feature: a dark mode for the dashboard."},
	})
	res, err := m.Classify(ctx, prompt2, labels, ClassifyOptions{Complete: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Best() != 1 {
		t.Errorf("dark mode: %v, want feature request to win", res.Probs)
	}
	if res.CachedTokens < prefixMin {
		t.Errorf("dark mode: %d prompt tokens from the cache, want the shared instructions", res.CachedTokens)
	}
	t.Logf("dark mode: %v, %d of %d prompt tokens cached", res.Probs, res.CachedTokens, res.PromptTokens)

	// many labels: several packed passes, and each label scores as it does alone
	var many []string
	for i := range 40 {
		many = append(many, fmt.Sprintf("category %d", i*7))
	}
	all, err := m.Classify(ctx, prompt, many, ClassifyOptions{Complete: true})
	if err != nil {
		t.Fatal(err)
	}
	packed := 0
	for _, n := range all.Tokens {
		packed += n - 1
	}
	if packed <= packMax {
		t.Fatalf("%d packed tokens, want more than one pass", packed)
	}
	for i, l := range many {
		one, err := m.Classify(ctx, prompt, []string{l}, ClassifyOptions{Complete: true})
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(math.Exp(one.Logprobs[0]) - math.Exp(all.Logprobs[i])); d > 0.01 || one.Tokens[0] != all.Tokens[i] {
			t.Errorf("%q: logprob %.4f packed, %.4f alone", l, all.Logprobs[i], one.Logprobs[0])
		}
	}

	if _, err := m.Classify(ctx, prompt, nil, ClassifyOptions{}); err == nil {
		t.Error("no labels: want an error")
	}
	if _, err := m.Classify(ctx, prompt, []string{"bug", ""}, ClassifyOptions{}); err == nil {
		t.Error("empty label: want an error")
	}
}
