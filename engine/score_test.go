package engine

import (
	"context"
	"math"
	"os"
	"testing"
)

// GGUF=<file> go test -run TestScore -v
// Score with Batch 1 must match teacher forcing one token at a time exactly (it is the same computation). Larger
// batches, and generation (whose prompt is read in batches), differ: batches of 4 or more tokens use the 4-column
// matmul kernels, which add a block's partial sums in a different order than the 1-column ones, and with 8-bit
// activations that rounding moves logprobs noticeably. Neither path is the more accurate one, so the test holds
// batched Score to the float reference (dequantized weights, float activations): it must be about as close to it
// as Batch 1 is. Measured (arm64, 2026-11-21), mean / worst nats over the test text:
//
//	Qwen3 0.6B Q8_0:    batch 64 vs batch 1 0.029 / 0.21; vs float: batch 1 0.047, batch 64 0.050
//	Qwen3 0.6B Q4_K_M:  batch 64 vs batch 1 0.074 / 0.60; vs float: batch 1 0.068 / 0.85, batch 64 0.054 / 0.25
//
// The loose bounds against Batch 1 and generation only catch a real error (a wrong position or mask is off by
// whole nats on many tokens).
const (
	scoreMeanTol = 0.15
	scoreMaxTol  = 1.0
)

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
	floatRef = true // the float reference: slow, but exact up to float rounding
	fr, err := m.Score(context.Background(), ids, ScoreOptions{Batch: 1})
	floatRef = false
	if err != nil {
		t.Fatal(err)
	}
	floatErr := func(lps []TokenLogprob) (sum float64) {
		for i := range lps {
			sum += math.Abs(lps[i].Logprob.Logprob - fr[i].Logprob.Logprob)
		}
		return sum / float64(len(lps))
	}
	var ref []TokenLogprob // batch 1
	for _, batch := range []int{1, 7, 64} {
		got, err := m.Score(context.Background(), ids, ScoreOptions{TopLogprobs: 5, Batch: batch})
		if err != nil {
			t.Fatal(err)
		}
		_, want := m.teacherForce(ids[:1], ids[1:])
		if len(got) != len(want) {
			t.Fatalf("batch %d: %d logprobs, want %d", batch, len(got), len(want))
		}
		var worst, sum float64
		for i := range got {
			if got[i].Token != ids[i+1] {
				t.Fatalf("batch %d position %d: token %d, want %d", batch, i, got[i].Token, ids[i+1])
			}
			d := math.Abs(got[i].Logprob.Logprob - want[i])
			if batch > 1 {
				d = math.Abs(got[i].Logprob.Logprob - ref[i].Logprob.Logprob)
			}
			worst, sum = max(worst, d), sum+d
			top := got[i].Top
			if len(top) != 5 || top[0].Logprob < top[4].Logprob || got[i].Logprob.Logprob > top[0].Logprob+1e-9 {
				t.Fatalf("batch %d position %d: bad top logprobs %v for %v", batch, i, top, got[i].Logprob)
			}
		}
		mean := sum / float64(len(got))
		switch {
		case batch == 1 && worst > 1e-4:
			t.Errorf("batch 1: logprobs differ from teacher forcing by up to %.5f", worst)
		case batch > 1 && (mean > scoreMeanTol || worst > scoreMaxTol):
			t.Errorf("batch %d: logprobs differ from batch 1 by %.4f on average, up to %.4f", batch, mean, worst)
		case batch > 1 && floatErr(got) > 1.5*floatErr(ref)+0.01:
			t.Errorf("batch %d: %.4f from the float reference on average, batch 1 is %.4f", batch, floatErr(got), floatErr(ref))
		}
		t.Logf("batch %d: %d tokens, difference from %s: mean %.5f, max %.5f; from float: mean %.5f", batch, len(got),
			map[bool]string{true: "teacher forcing", false: "batch 1"}[batch == 1], mean, worst, floatErr(got))
		if batch == 1 {
			ref = got
		}
	}

	res, err := m.GenerateTokens(context.Background(), ids, GenerateOptions{MaxTokens: 8, Logprobs: true, TopLogprobs: 3, IgnoreEOS: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Logprobs) != len(res.Tokens) {
		t.Fatalf("%d logprobs for %d tokens", len(res.Logprobs), len(res.Tokens))
	}
	sc, _ := m.Score(context.Background(), append(append([]int(nil), ids...), res.Tokens...), ScoreOptions{})
	var gsum float64
	for i, lp := range res.Logprobs {
		if lp.Token != res.Tokens[i] || lp.Top[0].Token != lp.Token { // greedy: the chosen token is the most likely
			t.Fatalf("generated token %d: %+v, token %d", i, lp, res.Tokens[i])
		}
		d := math.Abs(lp.Logprob.Logprob - sc[len(ids)-1+i].Logprob.Logprob)
		gsum += d
		if d > scoreMaxTol {
			t.Errorf("generated token %d: logprob %.4f, Score says %.4f", i, lp.Logprob.Logprob, sc[len(ids)-1+i].Logprob.Logprob)
		}
	}
	if mean := gsum / float64(len(res.Logprobs)); mean > scoreMeanTol {
		t.Errorf("generation logprobs differ from Score by %.4f on average", mean)
	}
}
