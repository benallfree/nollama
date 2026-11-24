// Package compose is a set of inference programs built only from the engine's public, experimental State and
// Decoder API: Classify, Score and Generate rebuilt on it (giving the same bits as the engine's own), and programs the
// engine doesn't have (best-of-N, beam search). It is the test of that API: each program should be short and read as
// what it does. Experimental: the API may change.
package compose

import (
	"context"
	"errors"

	"github.com/benallfree/nollama/engine"
)

// Score is engine.Model.Score: each token's log-probability given the ones before it.
func Score(ctx context.Context, m *engine.Model, ids []int, o engine.ScoreOptions) ([]engine.TokenLogprob, error) {
	if len(ids) < 2 {
		return nil, nil
	}
	if o.Batch <= 0 {
		o.Batch = 64
	}
	st, err := m.NewState(engine.StateOptions{MaxContext: len(ids), Batch: min(o.Batch, len(ids)-1)})
	if err != nil {
		return nil, err
	}
	defer st.Close()
	out := make([]engine.TokenLogprob, 0, len(ids)-1)
	err = st.StepAll(ctx, ids[:len(ids)-1], func(i int, logits []float32) {
		out = append(out, m.LogprobOf(logits, ids[i+1], o.TopLogprobs))
	})
	return out, err
}

// Classify is engine.Model.ClassifyTokens: read the prompt once, score each label as a branch of it.
func Classify(ctx context.Context, m *engine.Model, ids []int, labels []string, o engine.ClassifyOptions) (engine.ClassifyResult, error) {
	res := engine.ClassifyResult{PromptTokens: len(ids)}
	if len(ids) == 0 || len(labels) == 0 {
		return res, errors.New("compose: empty prompt or no labels")
	}
	var end []int
	if o.Complete {
		end = m.Tokenize("<|im_end|>")
	}
	toks := make([][]int, len(labels))
	branches := make([][]int, len(labels)) // each label's tokens but the last: the logits after them score the next
	longest, total := 0, 0
	for i, l := range labels {
		if l == "" {
			return res, errors.New("compose: empty label")
		}
		toks[i] = append(m.Tokenize(l), end...)
		branches[i] = toks[i][:len(toks[i])-1]
		longest, total = max(longest, len(toks[i])), total+len(branches[i])
	}
	limit := m.Info().ContextLength
	if len(ids)+longest > limit {
		return res, &engine.ContextError{PromptTokens: len(ids), MaxTokens: longest, Limit: limit}
	}
	st, err := m.NewState(engine.StateOptions{MaxContext: max(len(ids)+longest, min(len(ids)+total, limit))})
	if err != nil {
		return res, err
	}
	defer st.Close()
	if res.CachedTokens, err = st.Prefill(ctx, ids, engine.PrefillOptions{CacheScope: o.CacheScope}); err != nil {
		return res, err
	}
	res.Logprobs = make([]float64, len(labels))
	res.Tokens = make([]int, len(labels))
	for i, lt := range toks {
		res.Logprobs[i] = m.LogprobOf(st.Logits(), lt[0], 0).Logprob.Logprob
		res.Tokens[i] = len(lt)
	}
	err = st.StepBranches(ctx, branches, func(b, i int, logits []float32) {
		res.Logprobs[b] += m.LogprobOf(logits, toks[b][i+1], 0).Logprob.Logprob
	})
	if err != nil {
		return res, err
	}
	scores := make([]float64, len(labels))
	for i, lp := range res.Logprobs {
		scores[i] = lp
		if o.Normalize {
			scores[i] /= float64(res.Tokens[i])
		}
	}
	res.Probs = engine.Softmax(scores)
	return res, nil
}

// ClassifyRewind is Classify without packed branches: each label is read after the prompt and rewound. Same answer
// (bit for bit in deterministic mode), one forward pass per multi-token label instead of one for all.
func ClassifyRewind(ctx context.Context, m *engine.Model, ids []int, labels []string) ([]float64, error) {
	st, err := m.NewState(engine.StateOptions{MaxContext: min(len(ids)+256, m.Info().ContextLength)})
	if err != nil {
		return nil, err
	}
	defer st.Close()
	if _, err := st.Step(ctx, ids...); err != nil {
		return nil, err
	}
	first := append([]float32(nil), st.Logits()...)
	lps := make([]float64, len(labels))
	for i, l := range labels {
		lt := m.Tokenize(l)
		lps[i] = m.LogprobOf(first, lt[0], 0).Logprob.Logprob
		err := st.StepAll(ctx, lt[:len(lt)-1], func(j int, logits []float32) {
			lps[i] += m.LogprobOf(logits, lt[j+1], 0).Logprob.Logprob
		})
		if err != nil {
			return nil, err
		}
		if err := st.Rewind(len(lt) - 1); err != nil {
			return nil, err
		}
	}
	return lps, nil
}
