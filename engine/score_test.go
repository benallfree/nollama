package engine

import (
	"context"
	"math"
	"os"
	"testing"
)

// GGUF=<file> go test -run TestScore -v
// Score (batched, logits at every position) must match teacher forcing one token at a time, and generation
// logprobs must match Score on the generated text.
func TestScore(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m, err := Load(p, LoadOptions{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	text := "func fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\treturn fib(n-1) + fib(n-2)\n}\n\n// The old lighthouse keeper climbed the spiral stairs one last time, carrying a lantern."
	ids := m.Tokenize(text)
	for _, batch := range []int{1, 7, 64} {
		got, err := m.Score(context.Background(), ids, ScoreOptions{TopLogprobs: 5, Batch: batch})
		if err != nil {
			t.Fatal(err)
		}
		_, want := m.teacherForce(ids[:1], ids[1:])
		if len(got) != len(want) {
			t.Fatalf("batch %d: %d logprobs, want %d", batch, len(got), len(want))
		}
		var worst float64
		for i := range got {
			if got[i].Token != ids[i+1] {
				t.Fatalf("batch %d position %d: token %d, want %d", batch, i, got[i].Token, ids[i+1])
			}
			worst = max(worst, math.Abs(got[i].Logprob.Logprob-want[i]))
			top := got[i].Top
			if len(top) != 5 || top[0].Logprob < top[4].Logprob || got[i].Logprob.Logprob > top[0].Logprob+1e-9 {
				t.Fatalf("batch %d position %d: bad top logprobs %v for %v", batch, i, top, got[i].Logprob)
			}
		}
		// batched matmuls quantize activations the same way per row, but prefill and decode kernels differ slightly
		if worst > 0.05 {
			t.Errorf("batch %d: logprobs differ from teacher forcing by up to %.4f", batch, worst)
		}
		t.Logf("batch %d: %d tokens, max difference from teacher forcing %.5f", batch, len(got), worst)
	}

	res, err := m.GenerateTokens(context.Background(), ids, GenerateOptions{MaxTokens: 8, Logprobs: true, TopLogprobs: 3, IgnoreEOS: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Logprobs) != len(res.Tokens) {
		t.Fatalf("%d logprobs for %d tokens", len(res.Logprobs), len(res.Tokens))
	}
	sc, _ := m.Score(context.Background(), append(append([]int(nil), ids...), res.Tokens...), ScoreOptions{})
	for i, lp := range res.Logprobs {
		if lp.Token != res.Tokens[i] || lp.Top[0].Token != lp.Token { // greedy: the chosen token is the most likely
			t.Fatalf("generated token %d: %+v, token %d", i, lp, res.Tokens[i])
		}
		if d := math.Abs(lp.Logprob.Logprob - sc[len(ids)-1+i].Logprob.Logprob); d > 0.05 {
			t.Errorf("generated token %d: logprob %.4f, Score says %.4f", i, lp.Logprob.Logprob, sc[len(ids)-1+i].Logprob.Logprob)
		}
	}
}
