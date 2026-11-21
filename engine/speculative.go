package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

// Speculative decoding: a small draft model that shares the target's tokenizer guesses the next few tokens, and the
// target checks them all in one forward pass. Where checking k tokens costs about as much as writing one (reading the
// weights is the work), every guess the target agrees with is a token for free. On this engine's CPU kernels a short
// batch still costs several times one token, so today it is slower; it speeds up with the batch kernels. The guesses are accepted with standard
// speculative sampling (Leviathan et al. 2023; Chen et al. 2023), so the output follows the target's own distribution:
// with temperature 0 it is the target's own answer, and with sampling it is distributed as the target's sampler would
// draw it.

// DefaultDraftTokens is how many tokens the draft guesses at a time when GenerateOptions.DraftTokens is 0.
const DefaultDraftTokens = 4

// DefaultDraftMinP is GenerateOptions.DraftMinP's default: the draft stops guessing after a token it gives less than
// this probability.
const DefaultDraftMinP = 0.5

// softmaxAt is the probability softmax(logits) gives id.
func softmaxAt(logits []float32, id int) float64 {
	mx := float32(math.Inf(-1))
	for _, v := range logits {
		mx = max(mx, v)
	}
	var sum float64
	for _, v := range logits {
		sum += exp64(float64(v - mx))
	}
	return exp64(float64(logits[id]-mx)) / sum
}

// CheckDraft reports whether draft can guess tokens for target: the two must share a tokenizer (the same tokens with
// the same ids, the same end-of-turn tokens) and a vocabulary size.
func CheckDraft(target, draft *Model) error {
	if target == nil || draft == nil {
		return errors.New("engine: draft: nil model")
	}
	if target == draft {
		return errors.New("engine: draft: the draft is the target itself")
	}
	if target.hp.vocab != draft.hp.vocab {
		return fmt.Errorf("engine: draft: vocabulary sizes differ (%d and %d)", target.hp.vocab, draft.hp.vocab)
	}
	a, b := target.tok, draft.tok
	if len(a.tokens) != len(b.tokens) {
		return fmt.Errorf("engine: draft: tokenizers differ (%d and %d tokens)", len(a.tokens), len(b.tokens))
	}
	for i := range a.tokens {
		if a.tokens[i] != b.tokens[i] {
			return fmt.Errorf("engine: draft: tokenizers differ at token %d (%q and %q)", i, a.tokens[i], b.tokens[i])
		}
	}
	if len(a.ranks) != len(b.ranks) {
		return errors.New("engine: draft: tokenizers have different merges")
	}
	for k, r := range a.ranks {
		if b.ranks[k] != r {
			return fmt.Errorf("engine: draft: tokenizers differ at merge %q", k)
		}
	}
	if len(a.eos) != len(b.eos) {
		return errors.New("engine: draft: end-of-turn tokens differ")
	}
	for id := range a.eos {
		if !b.eos[id] {
			return errors.New("engine: draft: end-of-turn tokens differ")
		}
	}
	return nil
}

// forwardRows is forward, but writes the logits of every token (len(tokens) rows of the vocabulary, row t at
// out[t*vocab:]) instead of only the last. It relies on forward leaving every token's final hidden state in s.x.
// Verifying a draft is a short batched forward, so it gets faster with whatever makes reading prompts faster.
//
// exact runs the batch row by row through the kernels (still one pass over the weights, which is what costs), so each
// token's logits are bit for bit the ones a one-token forward gives, and a greedy answer is exactly the plain one.
func (m *Model) forwardRows(s *state, tokens []int, pos int, out []float32, exact bool) {
	s.an.rowByRow, s.aa.rowByRow, s.af.rowByRow = exact, exact, exact
	defer func() { s.an.rowByRow, s.aa.rowByRow, s.af.rowByRow = false, false, false }()
	m.forward(s, tokens, pos)
	n, V, dim := len(tokens), m.hp.vocab, m.hp.dim
	copy(out[(n-1)*V:n*V], s.logits)
	if n == 1 {
		return
	}
	a := s.an
	a.n = n - 1
	for t := 0; t < n-1; t++ {
		rmsnorm(a.f[t*dim:(t+1)*dim], s.x[t*dim:(t+1)*dim], m.norm, m.hp.eps)
	}
	a.quantize(m.pool, m.out)
	m.matmul(m.pool, a, 0, mmSeg{m.out, out, V})
}

// tokDist is the distribution a sampler draws the next token from: ids with probabilities summing to 1, best first.
// fit is the constraint's verdict on each (nil without a constraint).
type tokDist struct {
	ids []int
	ps  []float64
	fit []Fit
}

func (d *tokDist) p(id int) float64 {
	for i, x := range d.ids {
		if x == id {
			return d.ps[i]
		}
	}
	return 0
}

func (d *tokDist) fitOf(id int) Fit {
	if d.fit == nil {
		return FitPrefix
	}
	for i, x := range d.ids {
		if x == id {
			return d.fit[i]
		}
	}
	return FitNo
}

func (d *tokDist) draw(rng *rand.Rand) int {
	if len(d.ids) == 1 {
		return d.ids[0]
	}
	r := rng.Float64()
	for i, p := range d.ps {
		r -= p
		if r < 0 {
			return d.ids[i]
		}
	}
	return d.ids[len(d.ids)-1]
}

// dist is the distribution sample (no constraint) or sampleConstrained (with one, after the output out) draws from,
// written out. ok is false when the constraint allows no token.
func (m *Model) dist(logits []float32, o GenerateOptions, out []byte) (d tokDist, ok bool) {
	greedy := o.Temperature <= 0 || o.TopK == 1
	if o.Constraint == nil {
		if greedy {
			best, bv := 0, float32(math.Inf(-1))
			for i, v := range logits {
				if v > bv {
					best, bv = i, v
				}
			}
			return tokDist{ids: []int{best}, ps: []float64{1}}, true
		}
		mxl := float32(math.Inf(-1))
		for _, v := range logits {
			mxl = max(mxl, v)
		}
		cut := mxl - float32(30*o.Temperature)
		type cand struct {
			id int
			v  float32
		}
		cs := make([]cand, 0, 256)
		for i, v := range logits {
			if v >= cut {
				cs = append(cs, cand{i, v})
			}
		}
		sort.Slice(cs, func(a, b int) bool { return cs[a].v > cs[b].v })
		k := o.TopK
		if k <= 0 || k > len(cs) {
			k = len(cs)
		}
		cs = cs[:k]
		ps := make([]float64, len(cs))
		var sum float64
		for i, c := range cs {
			ps[i] = exp64(float64(c.v-cs[0].v) / o.Temperature)
			sum += ps[i]
		}
		for i := range ps {
			ps[i] /= sum
		}
		if o.TopP > 0 && o.TopP < 1 {
			var c float64
			for i := range ps {
				c += ps[i]
				if c >= o.TopP {
					cs, ps = cs[:i+1], ps[:i+1]
					break
				}
			}
			for i := range ps {
				ps[i] /= c
			}
		}
		d.ids = make([]int, len(cs))
		for i, c := range cs {
			d.ids[i] = c.id
		}
		d.ps = ps
		return d, true
	}

	type cand struct {
		id int
		v  float32
	}
	prefix := string(out)
	cur := o.Constraint(prefix)
	try := func(id int) (Fit, bool) {
		if m.tok.eos[id] {
			return FitDone, cur == FitComplete || cur == FitDone
		}
		if m.tok.isControl(id) {
			return FitNo, false // <|im_start|> and the like; user-defined ones such as <tool_call> are text the constraint judges
		}
		f := o.Constraint(prefix + string(m.tok.tokenBytes(id)))
		return f, f != FitNo
	}
	for _, width := range []float32{30, float32(math.Inf(1))} {
		mx := float32(math.Inf(-1))
		for _, v := range logits {
			mx = max(mx, v)
		}
		cut := mx - width
		cs := make([]cand, 0, 512)
		for i, v := range logits {
			if v >= cut {
				cs = append(cs, cand{i, v})
			}
		}
		sort.Slice(cs, func(a, b int) bool { return cs[a].v > cs[b].v })
		k := o.TopK
		if k <= 0 {
			k = 40
		}
		var fits []cand
		var fitOf []Fit
		for _, c := range cs {
			f, good := try(c.id)
			if !good {
				continue
			}
			if greedy {
				return tokDist{ids: []int{c.id}, ps: []float64{1}, fit: []Fit{f}}, true
			}
			fits, fitOf = append(fits, c), append(fitOf, f)
			if len(fits) >= k {
				break
			}
		}
		if len(fits) == 0 {
			continue
		}
		var sum float64
		ps := make([]float64, len(fits))
		for i, c := range fits {
			ps[i] = exp64(float64(c.v-fits[0].v) / o.Temperature)
			sum += ps[i]
		}
		d.ids = make([]int, len(fits))
		for i, c := range fits {
			d.ids[i] = c.id
			ps[i] /= sum
		}
		d.ps, d.fit = ps, fitOf
		return d, true
	}
	return tokDist{}, false
}

// residual draws from max(0, p-q), normalized: the token to emit when the target turns a guess down. When p and q
// agree everywhere (only possible by rounding) it draws from p.
func residual(p, q *tokDist, rng *rand.Rand) int {
	ws := make([]float64, len(p.ids))
	var sum float64
	for i, id := range p.ids {
		ws[i] = max(0, p.ps[i]-q.p(id))
		sum += ws[i]
	}
	if sum <= 0 {
		return p.draw(rng)
	}
	r := rng.Float64() * sum
	for i, w := range ws {
		r -= w
		if r < 0 && w > 0 {
			return p.ids[i]
		}
	}
	for i := len(ws) - 1; i >= 0; i-- {
		if ws[i] > 0 {
			return p.ids[i]
		}
	}
	return p.ids[0]
}

// emitter turns accepted tokens into text the way GenerateTokens does: complete UTF-8 only, stop strings cut and held
// back while they could still be forming, the callback, and the reasons to end.
type emitter struct {
	m       *Model
	o       GenerateOptions
	res     *Result
	onToken func(string) bool
	text    strings.Builder
	pending []byte
	raw     []byte // every generated byte, for the constraint
	emitted int
	maxStop int
	start   time.Time
	decode  time.Time
	ended   bool
}

func (e *emitter) finish(reason string, flush bool) {
	if flush && e.onToken != nil {
		if full := e.text.String(); e.emitted < len(full) {
			e.onToken(full[e.emitted:])
		}
	}
	e.res.StopReason = reason
	e.res.Text = e.text.String()
	if !e.decode.IsZero() {
		e.res.Decode = time.Since(e.decode)
	}
	e.ended = true
}

// push emits tok (done: the constraint says the output is finished after it) and reports whether the output ended.
func (e *emitter) push(tok int, done bool) bool {
	res := e.res
	if res.OutputTokens == 0 && e.decode.IsZero() {
		res.Prefill = time.Since(e.start)
		e.decode = time.Now()
	}
	if e.m.tok.eos[tok] && !e.o.IgnoreEOS {
		e.finish("eos", true)
		return true
	}
	res.Tokens = append(res.Tokens, tok)
	res.OutputTokens++
	tb := e.m.tok.tokenBytes(tok)
	e.raw = append(e.raw, tb...)
	e.pending = append(e.pending, tb...)
	k := 0
	for k < len(e.pending) {
		r, size := utf8.DecodeRune(e.pending[k:])
		if r == utf8.RuneError && size <= 1 && !utf8.FullRune(e.pending[k:]) {
			break
		}
		k += size
	}
	e.text.Write(e.pending[:k])
	e.pending = e.pending[k:]

	stopped := false
	full := e.text.String()
	if e.maxStop > 0 {
		for _, st := range e.o.Stop {
			if st == "" {
				continue
			}
			if i := strings.Index(full[max(0, e.emitted-len(st)):], st); i >= 0 {
				cut := max(0, e.emitted-len(st)) + i
				e.text.Reset()
				e.text.WriteString(full[:cut])
				full = full[:cut]
				stopped = true
				break
			}
		}
	}
	safe := len(full)
	if !stopped && e.maxStop > 0 {
		safe = max(e.emitted, len(full)-(e.maxStop-1))
	}
	if safe > e.emitted && e.onToken != nil {
		if !e.onToken(full[e.emitted:safe]) {
			e.emitted = safe
			e.finish("callback", false)
			return true
		}
	}
	e.emitted = max(e.emitted, safe)
	switch {
	case done && !stopped:
		e.finish("stop", true)
	case stopped:
		e.finish("stop", false)
	case res.OutputTokens >= e.o.MaxTokens:
		e.finish("length", true)
	}
	return e.ended
}

// generateSpeculative is GenerateTokens with o.Draft guessing tokens for m to check.
func (m *Model) generateSpeculative(ctx context.Context, ids []int, o GenerateOptions, onToken func(piece string) bool) (res Result, err error) {
	d := o.Draft
	res = Result{PromptTokens: len(ids)}
	if len(ids) == 0 {
		return res, errors.New("engine: empty prompt")
	}
	if err := CheckDraft(m, d); err != nil {
		return res, err
	}
	if m.respond != nil || d.respond != nil || o.Logprobs || o.OnLogprob != nil {
		// a scripted model has nothing to check, and the check doesn't compute logprobs: answer without the draft
		o.Draft = nil
		return m.generateTokens(ctx, ids, o, onToken)
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 256
	}
	k := o.DraftTokens
	if k <= 0 {
		k = DefaultDraftTokens
	}
	k = min(k, 16)
	// greedy: check row by row, so the answer is bit for bit the plain one. Sampling only promises a distribution,
	// which the batch kernels' rounding doesn't change in any way that matters, and they check faster.
	exact := (o.Temperature <= 0 || o.TopK == 1) && !o.DraftFast
	minP := o.DraftMinP
	if minP == 0 {
		minP = DefaultDraftMinP
	}
	maxCtx := len(ids) + o.MaxTokens
	if err := checkContext(len(ids), o.MaxTokens, min(m.hp.ctx, d.hp.ctx)); err != nil { // a *ContextError
		return res, err
	}
	s, err := m.getState(maxCtx, max(min(len(ids), prefillBatch), k+1))
	if err != nil {
		return res, err
	}
	defer m.putState(s)
	ds, err := d.getState(maxCtx, max(min(len(ids), prefillBatch), 2))
	if err != nil {
		return res, err
	}
	defer d.putState(ds)
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
	e := &emitter{m: m, o: o, res: &res, onToken: onToken, start: time.Now()}
	for _, st := range o.Stop {
		e.maxStop = max(e.maxStop, len(st))
	}

	// The target reads the prompt exactly as GenerateTokens does (same caches, same batches, so the same numbers) and
	// chooses the first token. From then on it is always one token behind: the token it last chose leads the next batch
	// it checks, with the draft's guesses after it.
	from := 0
	var te *prefixEntry
	tn := 0
	if o.Thread != "" {
		te, tn = m.lookupThread(o.CacheScope, o.Thread, o.ThreadDir, ids)
	}
	if pe, n := m.lookupPrefix(o.CacheScope, ids); pe != nil && n > tn {
		te, tn = pe, n
	}
	if te != nil {
		m.restorePrefix(s, te, tn)
		from, res.CachedTokens = tn, tn
	}
	for pos := from; pos < len(ids); pos += prefillBatch {
		if err := ctx.Err(); err != nil {
			res.StopReason = "canceled"
			return res, err
		}
		m.forward(s, ids[pos:min(pos+prefillBatch, len(ids))], pos)
	}
	m.storePrefix(o.CacheScope, s, ids, "")

	seq := append(make([]int, 0, len(ids)+o.MaxTokens), ids...)
	tValid := len(ids) // tokens of seq whose keys and values are in s
	if o.Thread != "" {
		defer func() {
			// the span GenerateTokens keeps: every answer token it read, which is all but the last unless it ended at
			// end-of-turn, so a thread continues the same way with a draft or without
			n := len(res.Tokens)
			if res.StopReason != "eos" {
				n--
			}
			n = max(0, min(n, tValid-len(ids)))
			m.storeThread(o, s, append(append([]int(nil), ids...), res.Tokens[:n]...))
		}()
	}
	{
		p, ok := m.dist(s.logits, o, nil)
		if !ok {
			e.finish("stop", true) // nothing allowed can follow
			return res, nil
		}
		tok := p.draw(rng)
		if e.push(tok, p.fit != nil && p.fitOf(tok) == FitDone && !m.tok.eos[tok]) {
			return res, nil
		}
		seq = append(seq, tok)
	}

	// the draft reads the whole prompt (it guesses the token after it)
	dFed := 0
	if pe, n := d.lookupPrefix(o.CacheScope, ids); pe != nil {
		d.restorePrefix(ds, pe, n)
		dFed = n
	}
	for pos := dFed; pos < len(ids); pos += prefillBatch {
		if err := ctx.Err(); err != nil {
			res.StopReason = "canceled"
			return res, err
		}
		d.forward(ds, ids[pos:min(pos+prefillBatch, len(ids))], pos)
	}
	dFed = len(ids)
	d.storePrefix(o.CacheScope, ds, ids, "")

	V := m.hp.vocab
	rows := make([]float32, (k+1)*V)
	drafts := make([]int, 0, k)
	qs := make([]tokDist, 0, k)
	batch := make([]int, 0, k+1)
	for {
		if err := ctx.Err(); err != nil {
			e.finish("canceled", false)
			return res, err
		}
		// the draft catches up on tokens it hasn't read (the target's last choice, or a guess it didn't read)
		if dFed < len(seq) {
			d.forward(ds, seq[dFed:], dFed)
			dFed = len(seq)
		}
		// guess up to kk tokens, never more than the answer has room for
		kk := min(k, o.MaxTokens-res.OutputTokens-1)
		drafts, qs = drafts[:0], qs[:0]
		draw := e.raw
		for i := 0; i < kk; i++ {
			q, ok := d.dist(ds.logits, o, draw)
			if !ok {
				break
			}
			tok := q.draw(rng)
			drafts, qs = append(drafts, tok), append(qs, q)
			if (m.tok.eos[tok] && !o.IgnoreEOS) || (q.fit != nil && q.fitOf(tok) == FitDone) {
				break // nothing follows
			}
			if i == kk-1 {
				break // the last guess needn't be read
			}
			if minP > 0 && softmaxAt(ds.logits, tok) < minP {
				break // the draft isn't sure of this one; guessing on from it is likely wasted work
			}
			if o.Constraint != nil {
				if i == 0 {
					draw = append([]byte(nil), e.raw...)
				}
				draw = append(draw, m.tok.tokenBytes(tok)...)
			}
			d.forward(ds, []int{tok}, dFed)
			dFed++
		}
		dRead := dFed // the draft has read seq plus the guesses before dRead

		// the target checks: its last choice and the guesses, in one pass
		batch = append(append(batch[:0], seq[len(seq)-1]), drafts...)
		base := len(seq) - 1
		m.forwardRows(s, batch, base, rows, exact)
		res.Drafted += len(drafts)
		res.Checks++

		accepted := 0
		for i := 0; i <= len(drafts); i++ {
			p, ok := m.dist(rows[i*V:(i+1)*V], o, e.raw)
			if !ok {
				tValid = base + 1 + accepted
				e.finish("stop", true) // nothing allowed can follow
				return res, nil
			}
			var tok int
			guessed := false
			if i < len(drafts) {
				g := drafts[i]
				pg, qg := p.p(g), qs[i].p(g)
				if pg > 0 && (pg >= qg || rng.Float64() < pg/qg) {
					tok, guessed = g, true
				} else {
					tok = residual(&p, &qs[i], rng)
				}
			} else {
				tok = p.draw(rng)
			}
			done := p.fit != nil && p.fitOf(tok) == FitDone && !m.tok.eos[tok]
			if guessed {
				accepted++
				res.Accepted++
			}
			ended := e.push(tok, done)
			if ended || !guessed {
				// keys and values in s: the batch's lead token, then the accepted guesses
				tValid = base + 1 + accepted
				if !e.ended {
					seq = append(seq, tok)
				}
				break
			}
			seq = append(seq, tok)
		}
		if e.ended {
			return res, nil
		}
		// all guesses taken, plus the target's own next token: seq grew by accepted+1
		tValid = base + 1 + accepted
		dFed = min(dRead, base+1+accepted)
	}
}
