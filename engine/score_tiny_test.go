package engine

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

// Score's plumbing on the tiny model, in every go test: batch 1 is teacher forcing, generation logprobs line up
// with their tokens, and the all-position logits are reserved in the memory budget and released after.
func TestScorePlumbing(t *testing.T) {
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
	ids := m.Tokenize("the quick brown fox jumps over the lazy dog, and the dog sleeps on")
	if len(ids) < 10 {
		t.Fatalf("only %d tokens", len(ids))
	}
	got, err := m.Score(context.Background(), ids, ScoreOptions{Batch: 1, TopLogprobs: 2})
	if err != nil {
		t.Fatal(err)
	}
	_, want := m.teacherForce(ids[:1], ids[1:])
	if len(got) != len(ids)-1 {
		t.Fatalf("%d logprobs for %d tokens", len(got), len(ids))
	}
	for i := range got {
		if got[i].Token != ids[i+1] || math.Abs(got[i].Logprob.Logprob-want[i]) > 1e-4 || len(got[i].Top) != 2 {
			t.Fatalf("position %d: %+v, teacher forcing %v", i, got[i], want[i])
		}
	}

	if _, err := m.Score(context.Background(), ids, ScoreOptions{Batch: 64}); err != nil {
		t.Fatal(err)
	}
	used := r.Usage().MemoryUsed // weights and the states kept for reuse
	if _, err := m.Score(context.Background(), ids, ScoreOptions{Batch: 64}); err != nil {
		t.Fatal(err)
	}
	if u := r.Usage().MemoryUsed; u != used {
		t.Errorf("memory used %d after a second Score, %d after the first: the logits weren't released", u, used)
	}

	// a state that comes back from Score must not keep computing every position's logits
	res, err := m.GenerateTokens(context.Background(), ids, GenerateOptions{MaxTokens: 4, Logprobs: true, IgnoreEOS: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Logprobs) != len(res.Tokens) {
		t.Fatalf("%d logprobs for %d tokens", len(res.Logprobs), len(res.Tokens))
	}
	for i, lp := range res.Logprobs {
		if lp.Token != res.Tokens[i] || lp.Logprob.Logprob > 0 {
			t.Errorf("generated token %d: %+v", i, lp)
		}
	}

	// a budget with room for the weights but not the logits
	tight := NewResources(ResourceOptions{Memory: used + 1024})
	defer tight.Close()
	m2, err := Load(p, LoadOptions{Resources: tight})
	if err != nil {
		t.Skipf("tiny model doesn't load in %d bytes: %v", used+1024, err)
	}
	defer m2.Close()
	if _, err := m2.Score(context.Background(), ids, ScoreOptions{Batch: 64}); !errors.Is(err, ErrMemoryLimit) {
		t.Errorf("Score over the budget: %v, want ErrMemoryLimit", err)
	}
}
