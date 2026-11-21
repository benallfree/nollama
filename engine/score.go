package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

// Logprob is a token and its log-probability (natural log), as the model saw it before sampling: temperature,
// top-k, top-p and constraints don't change it.
type Logprob struct {
	Token   int
	Logprob float64
}

// TokenLogprob is one token's log-probability given everything before it, with the most likely alternatives there.
type TokenLogprob struct {
	Logprob
	Top []Logprob // the TopLogprobs most likely tokens at this position, most likely first (nil when not asked for)
}

// ScoreOptions configures Score.
type ScoreOptions struct {
	TopLogprobs int // also return this many most likely tokens at each position (0: none)
	Batch       int // tokens per forward pass (default 64); each needs a vocabulary of logits, 600 KB for Qwen3, reserved in the model's Resources
}

// Score returns, for each token of ids after the first, its log-probability given the tokens before it: the model
// reads ids once, in batches, and keeps the logits of every position (prompt processing normally keeps only the
// last). The sum of -Logprob/ln 2 is the size in bits an arithmetic coder driven by the model would give ids[1:].
// Use it to rank candidate texts, measure perplexity, or compress text.
func (m *Model) Score(ctx context.Context, ids []int, o ScoreOptions) ([]TokenLogprob, error) {
	if len(ids) < 2 {
		return nil, nil
	}
	if len(ids) > m.hp.ctx {
		return nil, errors.New("engine: text exceeds the model's context length")
	}
	b := o.Batch
	if b <= 0 {
		b = 64
	}
	b = min(b, len(ids)-1)
	s, err := m.getState(len(ids), b)
	if err != nil {
		return nil, err
	}
	v := m.hp.vocab
	lb := int64(b) * int64(v) * 4 // the logits of every position in a batch
	if err := m.res.reserve(lb, fmt.Sprintf("logits for batches of %d tokens", b)); err != nil {
		m.putState(s)
		return nil, err
	}
	s.allLogits = make([]float32, b*v)
	defer func() {
		s.allLogits = nil // the state goes back for generation, which wants the last token's logits only
		m.putState(s)
		m.res.release(lb)
	}()
	out := make([]TokenLogprob, 0, len(ids)-1)
	for pos := 0; pos < len(ids)-1; pos += b {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		end := min(pos+b, len(ids)-1) // the last token is only a target
		m.forward(s, ids[pos:end], pos)
		for t := 0; t < end-pos; t++ {
			out = append(out, logprobOf(s.allLogits[t*v:(t+1)*v], ids[pos+t+1], o.TopLogprobs, m.det))
		}
	}
	return out, nil
}

// logprobOf is tok's log-softmax in logits, plus the top n tokens. det uses the portable exp and log (detmath.go),
// the same bits on every architecture; math.Exp is about 1.5x faster, and this runs over the whole vocabulary.
func logprobOf(logits []float32, tok, n int, det bool) TokenLogprob {
	exp, log := math.Exp, math.Log
	if det {
		exp, log = exp64, log64
	}
	mx := float32(math.Inf(-1))
	for _, x := range logits {
		mx = max(mx, x)
	}
	var sum float64
	for _, x := range logits {
		sum += exp(float64(x - mx))
	}
	lse := float64(mx) + log(sum)
	r := TokenLogprob{Logprob: Logprob{tok, float64(logits[tok]) - lse}}
	if n <= 0 {
		return r
	}
	n = min(n, len(logits))
	top := make([]Logprob, 0, n+1) // kept sorted, most likely first
	for i, x := range logits {
		if len(top) == n && x <= float32(top[n-1].Logprob) {
			continue
		}
		j := sort.Search(len(top), func(k int) bool { return float32(top[k].Logprob) < x })
		top = append(top, Logprob{})
		copy(top[j+1:], top[j:])
		top[j] = Logprob{i, float64(x)}
		if len(top) > n {
			top = top[:n]
		}
	}
	for i := range top {
		top[i].Logprob -= lse
	}
	r.Top = top
	return r
}

// TokenBytes is the raw bytes of one token (which may be part of a UTF-8 character).
func (m *Model) TokenBytes(id int) []byte { return append([]byte(nil), m.tok.tokenBytes(id)...) }
