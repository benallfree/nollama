package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// ClassifyOptions configures Classify.
type ClassifyOptions struct {
	// Normalize divides each label's log-probability by its token count before the softmax, so a long label isn't
	// penalized for having more tokens to get right. Off, a label's score is the probability of its whole text.
	Normalize bool
	// Complete scores each label as the entire answer: the label must be followed by the end of the turn
	// (<|im_end|>). Use it when one label is the start of another ("bug" and "bug report"), with a prompt from
	// ChatPrompt. Off, a label is scored as the start of the answer.
	Complete bool
	// CacheScope is GenerateOptions.CacheScope: the prompt cache reuses only entries from the same scope.
	CacheScope string
}

// ClassifyResult is Classify's answer, one entry per label in the order given.
type ClassifyResult struct {
	Probs        []float64 // softmax over the labels' scores; sums to 1
	Logprobs     []float64 // each label's log-probability given the prompt (natural log, summed over its tokens)
	Tokens       []int     // each label's token count (with the end of turn, when Complete)
	PromptTokens int
	CachedTokens int // of PromptTokens, how many came from the prompt cache
}

// Best is the index of the most likely label.
func (r ClassifyResult) Best() int {
	b := 0
	for i, p := range r.Probs {
		if p > r.Probs[b] {
			b = i
		}
	}
	return b
}

// Classify reads prompt (raw text; the caller applies the chat template, as for Generate) once and scores each label
// as its continuation, returning a probability for each: the model's own odds of answering with that label,
// renormalized over the labels. Nothing is generated or parsed, so the answer is always one of the labels, and the
// probabilities say how sure the model is.
//
// The prompt should end where the answer begins (ChatPrompt does), and labels are tokenized on their own, so write
// them as the model would start its answer ("bug", not " bug", after ChatPrompt). The prompt's state is computed once
// and shared by every label: single-token labels cost nothing more, and the other labels' tokens are read together in
// one batched pass (more past 64 tokens), each label seeing only the prompt and itself. The prompt cache applies, so
// classifying many items behind the same instructions skips re-reading them.
func (m *Model) Classify(ctx context.Context, prompt string, labels []string, o ClassifyOptions) (ClassifyResult, error) {
	return m.ClassifyTokens(ctx, m.Tokenize(prompt), labels, o)
}

// ClassifyTokens is Classify on a pre-tokenized prompt.
func (m *Model) ClassifyTokens(ctx context.Context, ids []int, labels []string, o ClassifyOptions) (res ClassifyResult, err error) {
	res.PromptTokens = len(ids)
	if len(ids) == 0 {
		return res, errors.New("engine: empty prompt")
	}
	if len(labels) == 0 {
		return res, errors.New("engine: no labels")
	}
	var end []int
	if o.Complete {
		end = m.Tokenize("<|im_end|>")
	}
	toks := make([][]int, len(labels))
	longest, total := 0, 0 // the longest label, and the tokens the packed passes read
	for i, l := range labels {
		if l == "" {
			return res, errors.New("engine: empty label")
		}
		toks[i] = append(m.Tokenize(l), end...)
		longest = max(longest, len(toks[i]))
		total += len(toks[i]) - 1
	}
	if err := checkContext(len(ids), longest, m.hp.ctx); err != nil { // a *ContextError: never cut the prompt
		return res, err
	}

	// label tokens per packed pass: up to packMax, at least the longest label, at most what the context has room for
	// after the prompt (which the check above leaves for the longest label)
	rows := min(max(packMax, longest-1), total, m.hp.ctx-len(ids))
	s, err := m.getState(len(ids)+rows, max(min(len(ids), prefillBatch), rows))
	if err != nil {
		return res, err
	}
	lb := int64(rows) * int64(m.hp.vocab) * 4 // the logits of every label token in a packed pass
	if err := m.res.reserve(lb, fmt.Sprintf("logits for %d label tokens", rows)); err != nil {
		m.putState(s)
		return res, err
	}
	defer func() {
		// pooled states go back to generation, which wants only the last logits and plain positions
		s.allLogits, s.ropePos, s.packStart = nil, nil, nil
		m.putState(s)
		m.res.release(lb)
	}()

	from := 0
	if e, n := m.lookupPrefix(o.CacheScope, ids); e != nil {
		m.restorePrefix(s, e, n)
		from, res.CachedTokens = n, n
	}
	for pos := from; pos < len(ids); pos += prefillBatch {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		m.forward(s, ids[pos:min(pos+prefillBatch, len(ids))], pos)
	}
	m.storePrefix(o.CacheScope, s, ids, "")

	// Every label's first token is scored on the prompt's own last logits. The rest are read in packed passes: the
	// labels' tokens go one after another after the prompt, each label rotated as if it came right after the prompt
	// and hidden from the others, so one pass scores many labels.
	last := append([]float32(nil), s.logits...)
	v := m.hp.vocab
	res.Logprobs = make([]float64, len(labels))
	res.Tokens = make([]int, len(labels))
	for i, lt := range toks {
		res.Logprobs[i] = logprobOf(last, lt[0], 0, m.det).Logprob.Logprob
		res.Tokens[i] = len(lt)
	}
	if total > 0 {
		s.allLogits = make([]float32, rows*v)
		s.packBase = len(ids)
	}
	for i := 0; i < len(toks); {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		var batch, rope, startOf []int
		first := i
		for ; i < len(toks) && (len(batch) == 0 || len(batch)+len(toks[i])-1 <= rows); i++ {
			lt := toks[i]
			at := len(ids) + len(batch)
			for j := 0; j < len(lt)-1; j++ {
				batch = append(batch, lt[j])
				rope = append(rope, len(ids)+j)
				startOf = append(startOf, at)
			}
		}
		if len(batch) == 0 {
			continue // single-token labels only
		}
		s.ropePos, s.packStart = rope, startOf
		m.forward(s, batch, len(ids))
		r := 0
		for k := first; k < i; k++ {
			lt := toks[k]
			for j := 1; j < len(lt); j++ {
				res.Logprobs[k] += logprobOf(s.allLogits[r*v:(r+1)*v], lt[j], 0, m.det).Logprob.Logprob
				r++
			}
		}
	}
	scores := make([]float64, len(labels))
	for i, lp := range res.Logprobs {
		scores[i] = lp
		if o.Normalize {
			scores[i] /= float64(res.Tokens[i])
		}
	}
	res.Probs = softmax64(scores)
	return res, nil
}

// packMax is the most label tokens a packed pass reads; each keeps a vocabulary of logits (600 KB for Qwen3).
const packMax = 64

func softmax64(x []float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range x {
		mx = max(mx, v)
	}
	out := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		out[i] = exp64(v - mx) // portable: a few labels, so its cost doesn't matter
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
