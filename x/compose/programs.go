package compose

import (
	"context"
	"math"
	"sort"

	"github.com/benallfree/nollama/engine"
)

// Generate is engine.Model.GenerateTokens (without threads or a draft model) as a Decoder: every option is a piece of
// middleware. It gives the same tokens, text, stop reason, streamed pieces and logprobs.
func Generate(ctx context.Context, m *engine.Model, ids []int, o engine.GenerateOptions, onToken func(string) bool) (engine.Result, error) {
	res := engine.Result{PromptTokens: len(ids)}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 256
	}
	if limit := m.Info().ContextLength; len(ids)+o.MaxTokens > limit {
		return res, &engine.ContextError{PromptTokens: len(ids), MaxTokens: o.MaxTokens, Limit: limit}
	}
	st, err := m.NewState(engine.StateOptions{MaxContext: len(ids) + o.MaxTokens})
	if err != nil {
		return res, err
	}
	defer st.Close()
	if res.CachedTokens, err = st.Prefill(ctx, ids, engine.PrefillOptions{CacheScope: o.CacheScope}); err != nil {
		return res, err
	}
	var mw []any
	if !o.IgnoreEOS {
		mw = append(mw, engine.EOS{})
	}
	rec := &engine.RecordLogprobs{Top: o.TopLogprobs}
	if o.Logprobs {
		mw = append(mw, rec)
	}
	if o.OnLogprob != nil {
		mw = append(mw, engine.ObserverFunc(func(d *engine.Decode, tok int, logits []float32) {
			o.OnLogprob(m.LogprobOf(logits, tok, o.TopLogprobs))
		}))
	}
	mw = append(mw, engine.StopStrings(o.Stop))
	if onToken != nil {
		mw = append(mw, engine.NewStreamText(onToken))
	}
	if o.Constraint != nil { // a filter, and a stop condition checked after streaming, as Generate does
		mw = append(mw, engine.NewConstraint(o.Constraint))
	}
	mw = append(mw, engine.MaxTokens(o.MaxTokens))
	dec := engine.Decoder{Sampler: engine.Sampling{Temperature: o.Temperature, TopK: o.TopK, TopP: o.TopP}, Middleware: mw, Seed: o.Seed}
	r, err := dec.Run(ctx, st)
	res.Text, res.Tokens, res.StopReason, res.OutputTokens = r.Text, r.Tokens, r.StopReason, len(r.Tokens)
	if len(rec.Logprobs) > 0 { // the tokens that put at least one byte into Text
		off, keep := 0, 0
		for _, t := range res.Tokens {
			if off >= len(res.Text) {
				break
			}
			off += len(m.TokenBytes(t))
			keep++
		}
		res.Logprobs = rec.Logprobs[:min(keep, len(rec.Logprobs))]
	}
	return res, err
}

// BestOfN decodes n answers, each from a fork of st, and returns the one score rates highest. newDecoder makes each
// run's decoder (middleware keeps per-run state, so runs can't share one); give them different seeds.
func BestOfN(ctx context.Context, st *engine.State, n int, newDecoder func(i int) engine.Decoder,
	score func(engine.DecodeResult) float64) (best engine.DecodeResult, bestScore float64, err error) {
	bestScore = math.Inf(-1)
	for i := range n {
		f, err := st.Fork()
		if err != nil {
			return best, bestScore, err
		}
		r, err := newDecoder(i).Run(ctx, f)
		f.Close()
		if err != nil {
			return best, bestScore, err
		}
		if s := score(r); s > bestScore {
			best, bestScore = r, s
		}
	}
	return best, bestScore, nil
}

// Beam is beam search: it keeps the width most likely continuations of st (by summed log-probability), extending
// each by its width most likely tokens per step, for up to n tokens or until every beam has ended its turn. It
// returns the best beam's tokens (without the end of turn) and log-probability. st is unchanged.
func Beam(ctx context.Context, st *engine.State, width, n int) ([]int, float64, error) {
	m := st.Model()
	type beam struct {
		st   *engine.State // nil once the beam has ended its turn
		toks []int
		lp   float64
	}
	beams := []beam{{st: st}}
	defer func() {
		for _, b := range beams {
			if b.st != nil && b.st != st {
				b.st.Close()
			}
		}
	}()
	for step := 0; step < n; step++ {
		var next []beam
		for _, b := range beams {
			if b.st == nil {
				next = append(next, b) // ended: competes as it is
				continue
			}
			for _, t := range m.LogprobOf(b.st.Logits(), 0, width).Top {
				next = append(next, beam{st: b.st, toks: append(b.toks[:len(b.toks):len(b.toks)], t.Token), lp: b.lp + t.Logprob})
			}
		}
		sort.SliceStable(next, func(i, j int) bool { return next[i].lp > next[j].lp })
		next = next[:min(width, len(next))]
		live := 0
		for i := range next {
			b := &next[i]
			if b.st == nil {
				continue
			}
			tok := b.toks[len(b.toks)-1]
			if m.IsEOS(tok) {
				b.st, b.toks = nil, b.toks[:len(b.toks)-1]
				continue
			}
			f, err := b.st.Fork() // every child gets its own copy of its parent
			if err != nil {
				return nil, 0, err
			}
			b.st = f
			if _, err := f.Step(ctx, tok); err != nil {
				return nil, 0, err
			}
			live++
		}
		for _, b := range beams { // the parents
			if b.st != nil && b.st != st {
				b.st.Close()
			}
		}
		beams = next
		if live == 0 {
			break
		}
	}
	return beams[0].toks, beams[0].lp, nil
}
