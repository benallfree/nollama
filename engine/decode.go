package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"unicode/utf8"
)

// The decode loop, with middleware. Experimental: the API may change.
//
// Decoder.Run is Generate's loop on a State the caller prepared: each step it copies the logits, lets the logit
// processors change them, asks the sampler for a token (limited by the active token filters), checks the stop
// conditions, tells the observers, and reads the token. Everything Generate does per token is a piece of middleware
// here (Sampling, Constraint, EOS, MaxTokens, StopStrings, StreamText, RecordLogprobs), so a program changes one piece
// without copying the loop. A piece may implement several hooks (Constraint is a filter and a stop condition); each
// hook runs in the order the pieces are listed. Nothing here runs inside the forward pass: the boundary is the logits.

// Decode is one Run's progress, passed to every hook.
type Decode struct {
	Model  *Model
	State  *State
	Rand   *rand.Rand // seeded from Decoder.Seed, as Generate seeds from GenerateOptions.Seed
	Tokens []int      // generated so far (the one being judged included, for stop conditions and Finish)
	Raw    []byte     // their bytes, as generated (may end inside a UTF-8 character)
	Text   string     // the text so far: complete UTF-8 characters only, after any cut a stop condition made
	// Hold is how many bytes at the end of Text a stop condition may still cut (a stop string's possible start);
	// streamers hold them back. Stop conditions set it each step (StopStrings does); it starts at 0 each step.
	Hold int
	// Added is how many bytes Text gained this step.
	Added int

	pending []byte // bytes not yet forming complete UTF-8
}

// Cut shortens Text to its first n bytes (a stop string and what follows go). Tokens and Raw are unchanged.
func (d *Decode) Cut(n int) { d.Text = d.Text[:n] }

// LogitProcessor changes the logits before sampling: penalties, biases, bans. It sees a copy (State.Logits stays the
// model's own). Use it when deciding about every token is cheap; use a TokenFilter when it isn't.
type LogitProcessor interface {
	ProcessLogits(d *Decode, logits []float32)
}

// TokenFilter limits which tokens may come next, for checks too costly to run over the whole vocabulary (a grammar,
// a JSON schema): the sampler calls Allow lazily, best candidates first, and stops when it has what it needs.
type TokenFilter interface {
	// Active says whether the filter limits this step at all; when no filter is active the sampler samples freely.
	Active(d *Decode) bool
	Allow(d *Decode, tok int) bool
}

// Sampler picks the next token. allow is nil when no filter is active; otherwise only tokens it accepts may be picked,
// and ok false means none is.
type Sampler interface {
	Sample(d *Decode, logits []float32, allow func(tok int) bool) (tok int, ok bool)
}

// Rejecter stops before a sampled token is kept: EOS. A non-empty reason ends the run without the token.
type Rejecter interface {
	Reject(d *Decode, tok int) (reason string)
}

// Observer sees each token kept, with the model's own logits it was sampled from, before it reaches Text.
type Observer interface {
	Observe(d *Decode, tok int, logits []float32)
}

// StopCondition runs after a token is added to Tokens, Raw and Text, and may end the run (a non-empty reason, as
// Result.StopReason), cut Text, or set Hold.
type StopCondition interface {
	Stop(d *Decode) (reason string)
}

// Finisher is told how the run ended (streamers flush what they held back).
type Finisher interface {
	Finish(d *Decode, reason string)
}

// Decoder is a decode loop: a sampler and middleware.
type Decoder struct {
	Sampler    Sampler // nil: greedy
	Middleware []any   // each implements at least one of LogitProcessor, TokenFilter, Rejecter, Observer, StopCondition, Finisher
	Seed       uint64
}

// DecodeResult is what Run produced.
type DecodeResult struct {
	Text       string
	Tokens     []int
	StopReason string // from the hook that stopped, or "stop" when no token was allowed, "canceled"
}

// Run decodes from st, which must hold at least one token (its Logits are the first step's). On return st holds
// every generated token but the last, which ended the run (Step it to continue from there).
func (dc Decoder) Run(ctx context.Context, st *State) (DecodeResult, error) {
	if st.Len() == 0 || st.Logits() == nil {
		return DecodeResult{}, fmt.Errorf("engine: Decoder.Run needs a state with logits (Step it first)")
	}
	var procs []LogitProcessor
	var filters []TokenFilter
	var rejecters []Rejecter
	var observers []Observer
	var stops []StopCondition
	var finishers []Finisher
	for i, mw := range dc.Middleware {
		n := 0
		if h, ok := mw.(LogitProcessor); ok {
			procs, n = append(procs, h), n+1
		}
		if h, ok := mw.(TokenFilter); ok {
			filters, n = append(filters, h), n+1
		}
		if h, ok := mw.(Rejecter); ok {
			rejecters, n = append(rejecters, h), n+1
		}
		if h, ok := mw.(Observer); ok {
			observers, n = append(observers, h), n+1
		}
		if h, ok := mw.(StopCondition); ok {
			stops, n = append(stops, h), n+1
		}
		if h, ok := mw.(Finisher); ok {
			finishers, n = append(finishers, h), n+1
		}
		if n == 0 {
			return DecodeResult{}, fmt.Errorf("engine: middleware %d (%T) implements no decode hook", i, mw)
		}
	}
	sampler := dc.Sampler
	if sampler == nil {
		sampler = Sampling{}
	}
	d := &Decode{Model: st.m, State: st, Rand: rand.New(rand.NewPCG(dc.Seed, dc.Seed^0x9e3779b97f4a7c15))}
	finish := func(reason string) DecodeResult {
		for _, f := range finishers {
			f.Finish(d, reason)
		}
		return DecodeResult{Text: d.Text, Tokens: d.Tokens, StopReason: reason}
	}
	var work []float32
	var active []TokenFilter
	allow := func(tok int) bool {
		for _, f := range active {
			if !f.Allow(d, tok) {
				return false
			}
		}
		return true
	}
	for {
		logits := st.Logits()
		if len(procs) > 0 {
			work = append(work[:0], logits...)
			for _, p := range procs {
				p.ProcessLogits(d, work)
			}
			logits = work
		}
		active = active[:0]
		for _, f := range filters {
			if f.Active(d) {
				active = append(active, f)
			}
		}
		var tok int
		var ok bool
		if len(active) == 0 {
			tok, ok = sampler.Sample(d, logits, nil)
		} else {
			tok, ok = sampler.Sample(d, logits, allow)
		}
		if !ok {
			return finish("stop"), nil // nothing allowed can follow
		}
		for _, r := range rejecters {
			if reason := r.Reject(d, tok); reason != "" {
				return finish(reason), nil
			}
		}
		for _, o := range observers {
			o.Observe(d, tok, st.Logits())
		}
		tb := st.m.tok.tokenBytes(tok)
		d.Tokens = append(d.Tokens, tok)
		d.Raw = append(d.Raw, tb...)
		d.pending = append(d.pending, tb...)
		k := 0 // move the complete UTF-8 prefix into Text
		for k < len(d.pending) {
			r, size := utf8.DecodeRune(d.pending[k:])
			if r == utf8.RuneError && size <= 1 && !utf8.FullRune(d.pending[k:]) {
				break
			}
			k += size
		}
		d.Text += string(d.pending[:k])
		d.Added = k
		d.pending = d.pending[k:]
		d.Hold = 0
		for _, s := range stops {
			if reason := s.Stop(d); reason != "" {
				return finish(reason), nil
			}
		}
		if err := ctx.Err(); err != nil {
			return finish("canceled"), err
		}
		if _, err := st.Step(ctx, tok); err != nil {
			return finish("canceled"), err
		}
	}
}

// Sampling is Generate's sampler: greedy at temperature 0 (or TopK 1), else temperature, top-k and top-p, with the
// same draws for the same seed. With a filter active it samples among the TopK (default 40) best allowed tokens, as
// Generate does under a Constraint.
type Sampling struct {
	Temperature float64
	TopK        int
	TopP        float64
}

func (s Sampling) Sample(d *Decode, logits []float32, allow func(int) bool) (int, bool) {
	if allow != nil {
		return sampleFiltered(logits, s.Temperature, s.TopK, d.Rand, allow)
	}
	return d.Model.sample(logits, GenerateOptions{Temperature: s.Temperature, TopK: s.TopK, TopP: s.TopP}, d.Rand), true
}

// SamplerFunc adapts a function to a Sampler.
type SamplerFunc func(d *Decode, logits []float32, allow func(int) bool) (int, bool)

func (f SamplerFunc) Sample(d *Decode, logits []float32, allow func(int) bool) (int, bool) {
	return f(d, logits, allow)
}

// LogitProcessorFunc adapts a function to a LogitProcessor.
type LogitProcessorFunc func(d *Decode, logits []float32)

func (f LogitProcessorFunc) ProcessLogits(d *Decode, logits []float32) { f(d, logits) }

// ObserverFunc adapts a function to an Observer.
type ObserverFunc func(d *Decode, tok int, logits []float32)

func (f ObserverFunc) Observe(d *Decode, tok int, logits []float32) { f(d, tok, logits) }

// StopFunc adapts a function to a StopCondition.
type StopFunc func(d *Decode) string

func (f StopFunc) Stop(d *Decode) string { return f(d) }

// EOS ends the run at an end-of-turn token, which is not kept ("eos").
type EOS struct{}

func (EOS) Reject(d *Decode, tok int) string {
	if d.Model.tok.eos[tok] {
		return "eos"
	}
	return ""
}

// MaxTokens ends the run after n tokens ("length").
type MaxTokens int

func (n MaxTokens) Stop(d *Decode) string {
	if len(d.Tokens) >= int(n) {
		return "length"
	}
	return ""
}

// StopStrings ends the run when Text contains one of them, cutting it there ("stop"), and holds back text that could
// be the start of one.
type StopStrings []string

func (ss StopStrings) Stop(d *Decode) string {
	maxStop := 0
	for _, st := range ss {
		maxStop = max(maxStop, len(st))
	}
	if maxStop == 0 {
		return ""
	}
	// a new match ends in the text this step added (an earlier one would have stopped the run)
	from := max(0, len(d.Text)-d.Added-maxStop)
	for _, st := range ss {
		if st == "" {
			continue
		}
		if i := strings.Index(d.Text[from:], st); i >= 0 {
			d.Cut(from + i)
			return "stop"
		}
	}
	d.Hold = maxStop - 1
	return ""
}

// StreamText passes text to fn as it becomes final (holding back what a stop condition may still cut). fn returning
// false ends the run ("callback"). At the end it passes what is left, unless the run was canceled or fn stopped it.
type StreamText struct {
	fn      func(piece string) bool
	emitted int
}

// NewStreamText returns a StreamText calling fn.
func NewStreamText(fn func(piece string) bool) *StreamText { return &StreamText{fn: fn} }

func (s *StreamText) Stop(d *Decode) string {
	safe := max(s.emitted, len(d.Text)-d.Hold)
	if safe > s.emitted {
		piece := d.Text[s.emitted:safe]
		s.emitted = safe
		if !s.fn(piece) {
			return "callback"
		}
	}
	return ""
}

func (s *StreamText) Finish(d *Decode, reason string) {
	if reason == "canceled" || reason == "callback" || s.emitted >= len(d.Text) {
		return
	}
	s.fn(d.Text[s.emitted:])
	s.emitted = len(d.Text)
}

// RecordLogprobs keeps each kept token's log-probability (with the top N alternatives) in Logprobs.
type RecordLogprobs struct {
	Top      int
	Logprobs []TokenLogprob
}

func (r *RecordLogprobs) Observe(d *Decode, tok int, logits []float32) {
	r.Logprobs = append(r.Logprobs, d.Model.LogprobOf(logits, tok, r.Top))
}

// Constraint limits the output to text fn accepts, as GenerateOptions.Constraint does: a token filter while fn says
// the text isn't free, which allows end-of-turn only once the text is complete, and a stop condition ("stop") once fn
// says FitDone.
type Constraint struct {
	fn     func(text string) Fit
	cur    Fit
	curLen int // len(Raw) cur was computed for (-1: none)
}

// NewConstraint returns a Constraint calling fn.
func NewConstraint(fn func(text string) Fit) *Constraint { return &Constraint{fn: fn, curLen: -1} }

func (c *Constraint) fit(d *Decode) Fit {
	if c.curLen != len(d.Raw) {
		c.cur, c.curLen = c.fn(string(d.Raw)), len(d.Raw)
	}
	return c.cur
}

func (c *Constraint) Active(d *Decode) bool { return c.fit(d) != FitFree }

func (c *Constraint) Allow(d *Decode, tok int) bool {
	m := d.Model
	if m.tok.eos[tok] {
		return c.cur == FitComplete || c.cur == FitDone
	}
	if m.tok.isControl(tok) {
		return false
	}
	return c.fn(string(d.Raw)+string(m.tok.tokenBytes(tok))) != FitNo
}

// Stop ends the run after a token the filter judged FitDone. (A token sampled while the constraint was free isn't
// judged, as in Generate.)
func (c *Constraint) Stop(d *Decode) string {
	wasActive := c.cur != FitFree && c.curLen == len(d.Raw)-len(d.Model.tok.tokenBytes(d.Tokens[len(d.Tokens)-1]))
	if wasActive && c.fit(d) == FitDone {
		return "stop"
	}
	return ""
}

// BanTokens makes the given tokens impossible.
type BanTokens []int

func (b BanTokens) ProcessLogits(_ *Decode, logits []float32) {
	for _, t := range b {
		logits[t] = float32(math.Inf(-1))
	}
}

// RepetitionPenalty divides positive logits (multiplies negative ones) by Penalty for every token generated so far,
// as llama.cpp's repeat_penalty does over the output.
type RepetitionPenalty float32

func (p RepetitionPenalty) ProcessLogits(d *Decode, logits []float32) {
	seen := map[int]bool{}
	for _, t := range d.Tokens {
		if seen[t] {
			continue
		}
		seen[t] = true
		if logits[t] > 0 {
			logits[t] /= float32(p)
		} else {
			logits[t] *= float32(p)
		}
	}
}
