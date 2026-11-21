package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

// Classify's plumbing on the tiny model, in every go test: probabilities sum to 1, packed passes score each label as
// it scores alone and as Score does, the label logits are released, and a state that comes back still generates.
func TestClassifyPlumbing(t *testing.T) {
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := tinymodel.Write(p, tinymodel.Config{}); err != nil {
		t.Fatal(err)
	}
	r := NewResources(ResourceOptions{})
	defer r.Close()
	m, err := Load(p, LoadOptions{Resources: r})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	prompt := "the quick brown fox jumps over the lazy dog, and the label is"
	var labels []string
	for i := range 30 {
		labels = append(labels, fmt.Sprintf("dog %d fox", i*7))
	}
	res, err := m.Classify(ctx, prompt, labels, ClassifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	packed := 0
	for i, pr := range res.Probs {
		sum += pr
		packed += res.Tokens[i] - 1
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("probabilities sum to %v", sum)
	}
	if packed <= packMax {
		t.Fatalf("%d packed tokens, want more than one pass", packed)
	}
	ids := m.Tokenize(prompt)
	for _, i := range []int{0, 13, 29} {
		one, err := m.Classify(ctx, prompt, labels[i:i+1], ClassifyOptions{})
		if err != nil {
			t.Fatal(err)
		}
		sc, err := m.Score(ctx, append(append([]int(nil), ids...), m.Tokenize(labels[i])...), ScoreOptions{Batch: 1})
		if err != nil {
			t.Fatal(err)
		}
		var want float64
		for _, x := range sc[len(ids)-1:] {
			want += x.Logprob.Logprob
		}
		if math.Abs(one.Logprobs[0]-res.Logprobs[i]) > 0.05 || math.Abs(want-res.Logprobs[i]) > 0.05 {
			t.Errorf("%q: %.4f packed, %.4f alone, %.4f by Score", labels[i], res.Logprobs[i], one.Logprobs[0], want)
		}
	}

	used := r.Usage().MemoryUsed
	if _, err := m.Classify(ctx, prompt, labels, ClassifyOptions{Complete: true, Normalize: true}); err != nil {
		t.Fatal(err)
	}
	if u := r.Usage().MemoryUsed; u != used {
		t.Errorf("memory used %d after a second Classify, %d after the first: the label logits weren't released", u, used)
	}
	if g, err := m.GenerateTokens(ctx, ids, GenerateOptions{MaxTokens: 4, Logprobs: true, IgnoreEOS: true}, nil); err != nil || len(g.Logprobs) != len(g.Tokens) {
		t.Errorf("generating after Classify: %v %v", g, err)
	}

	var ce *ContextError
	long := make([]int, m.Info().ContextLength)
	if _, err := m.ClassifyTokens(ctx, long, labels, ClassifyOptions{}); !errors.As(err, &ce) {
		t.Errorf("prompt as long as the context: %v, want a ContextError", err)
	}
}
