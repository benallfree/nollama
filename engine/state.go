package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
)

// State is one sequence's state: the keys and values of every token read so far (the KV cache), and the logits after
// the last one. It is the engine one level below Generate, Classify and Score, which are built from the same steps:
// read tokens, look at the logits, pick or score, read more. Programs that need something those calls don't do
// (best-of-N, tree search, custom samplers) drive a State directly.
//
// Experimental: the API may change.
//
// A State is not safe for concurrent use; separate States (forks included) may step in parallel, sharing the
// model's weights and worker pool. Close it to give its memory back.
type State struct {
	m      *Model
	s      *state
	toks   []int
	logits []float32 // after the last token, or nil when unknown (after Rewind)
	batch  int
}

// StateOptions sizes a State.
type StateOptions struct {
	// MaxContext is the most tokens the state holds (0: the model's context length, at most 4096). Its keys and
	// values take Model.KVCacheBytes per token, reserved in the model's Resources.
	MaxContext int
	// Batch is the most tokens one forward pass reads (0: 512). Step, StepAll and StepBranches split longer input
	// into passes of this size, and each StepAll and StepBranches pass keeps a vocabulary of logits per token.
	Batch int
}

// ErrStateClosed is returned by a State used after Close.
var ErrStateClosed = errors.New("engine: state is closed")

// NewState returns an empty State.
func (m *Model) NewState(o StateOptions) (*State, error) {
	if m.respond != nil {
		return nil, errors.New("engine: a Respond model has no state")
	}
	if o.MaxContext <= 0 {
		o.MaxContext = min(m.hp.ctx, 4096)
	}
	if o.MaxContext > m.hp.ctx {
		return nil, &ContextError{PromptTokens: o.MaxContext, Limit: m.hp.ctx}
	}
	if o.Batch <= 0 {
		o.Batch = prefillBatch
	}
	o.Batch = min(o.Batch, o.MaxContext)
	s, err := m.getState(o.MaxContext, o.Batch)
	if err != nil {
		return nil, err
	}
	return &State{m: m, s: s, batch: o.Batch}, nil
}

// Model is the model the state belongs to.
func (st *State) Model() *Model { return st.m }

// Len is how many tokens the state holds.
func (st *State) Len() int { return len(st.toks) }

// Cap is the most tokens the state can hold.
func (st *State) Cap() int {
	if st.s == nil {
		return 0
	}
	return min(st.s.maxCtx, st.m.hp.ctx)
}

// Tokens are the tokens read so far. The slice is the state's own; don't change it.
func (st *State) Tokens() []int { return st.toks }

// Logits are the model's logits after the last token read: what Step returned. nil when empty or after Rewind. The
// slice is the state's own buffer, overwritten by the next Step; copy it before changing it.
func (st *State) Logits() []float32 { return st.logits }

func (st *State) check(n int) error {
	if st.s == nil {
		return ErrStateClosed
	}
	if len(st.toks)+n > st.Cap() {
		return &ContextError{PromptTokens: len(st.toks), MaxTokens: n, Limit: st.Cap()}
	}
	return nil
}

// Step reads tokens after the ones the state holds, in passes of up to Batch tokens, and returns the logits after
// the last of them (the state's buffer, as Logits).
func (st *State) Step(ctx context.Context, tokens ...int) ([]float32, error) {
	if len(tokens) == 0 {
		return st.logits, nil
	}
	if err := st.check(len(tokens)); err != nil {
		return nil, err
	}
	var err error
	st.m.res.call(func() {
		for i := 0; i < len(tokens); i += st.batch {
			if err = ctx.Err(); err != nil {
				return
			}
			chunk := tokens[i:min(i+st.batch, len(tokens))]
			st.m.forward(st.s, chunk, len(st.toks))
			st.toks = append(st.toks, chunk...)
			st.logits = st.s.logits
		}
	})
	if err != nil {
		return nil, err
	}
	return st.logits, nil
}

// StepAll is Step that also calls fn with the logits after every token: logits after tokens[i] (the model's
// prediction for tokens[i+1]). The slice is only valid during the call. Each pass reserves Batch vocabularies of
// logits in the model's Resources.
func (st *State) StepAll(ctx context.Context, tokens []int, fn func(i int, logits []float32)) error {
	if len(tokens) == 0 {
		return nil
	}
	if err := st.check(len(tokens)); err != nil {
		return err
	}
	b := min(st.batch, len(tokens))
	release, err := st.allLogits(b)
	if err != nil {
		return err
	}
	defer release()
	v := st.m.hp.vocab
	st.m.res.call(func() {
		for i := 0; i < len(tokens); i += b {
			if err = ctx.Err(); err != nil {
				return
			}
			chunk := tokens[i:min(i+b, len(tokens))]
			st.m.forward(st.s, chunk, len(st.toks))
			st.toks = append(st.toks, chunk...)
			for t := range chunk {
				fn(i+t, st.s.allLogits[t*v:(t+1)*v])
			}
		}
		st.logits = st.s.logits
	})
	return err
}

// allLogits gives the state a buffer for n tokens' logits, reserved in the model's Resources, until release.
func (st *State) allLogits(n int) (release func(), err error) {
	lb := int64(n) * int64(st.m.hp.vocab) * 4
	if err := st.m.res.reserve(lb, fmt.Sprintf("logits for batches of %d tokens", n)); err != nil {
		return nil, err
	}
	st.s.allLogits = make([]float32, n*st.m.hp.vocab)
	return func() {
		st.s.allLogits = nil
		st.m.res.release(lb)
	}, nil
}

// StepBranches reads several continuations of the state at once, each seeing the state's tokens and its own earlier
// tokens but not the other branches, and calls fn with the logits after token i of branch b. The state is unchanged
// afterwards: branches are scratch. Branches are packed into passes of up to max(64, the longest branch) tokens
// (bounded by Batch and the room left in the state), so many short continuations cost about one forward pass. This
// is how Classify scores its labels.
func (st *State) StepBranches(ctx context.Context, branches [][]int, fn func(b, i int, logits []float32)) error {
	if st.s == nil {
		return ErrStateClosed
	}
	if len(st.toks) == 0 {
		return errors.New("engine: StepBranches needs a state with at least one token")
	}
	longest, total := 0, 0
	for _, br := range branches {
		longest = max(longest, len(br))
		total += len(br)
	}
	if total == 0 {
		return nil
	}
	if err := st.check(longest); err != nil {
		return err
	}
	if longest > st.batch {
		return fmt.Errorf("engine: a branch of %d tokens is longer than the state's Batch (%d)", longest, st.batch)
	}
	base := len(st.toks)
	rows := min(max(packMax, longest), total, st.Cap()-base, st.batch)
	release, err := st.allLogits(rows)
	if err != nil {
		return err
	}
	s := st.s
	defer func() {
		s.ropePos, s.packStart = nil, nil
		release()
	}()
	s.packBase = base
	var keep []float32 // the packed passes overwrite the state's logits
	if st.logits != nil {
		keep = append([]float32(nil), st.logits...)
	}
	v := st.m.hp.vocab
	st.m.res.call(func() {
		for i := 0; i < len(branches); {
			if err = ctx.Err(); err != nil {
				return
			}
			var batch, rope, startOf []int
			first := i
			for ; i < len(branches) && (len(batch) == 0 || len(batch)+len(branches[i]) <= rows); i++ {
				at := base + len(batch)
				for j, tok := range branches[i] {
					batch = append(batch, tok)
					rope = append(rope, base+j)
					startOf = append(startOf, at)
				}
			}
			if len(batch) == 0 {
				continue // empty branches only
			}
			s.ropePos, s.packStart = rope, startOf
			st.m.forward(s, batch, base)
			r := 0
			for k := first; k < i; k++ {
				for j := range branches[k] {
					fn(k, j, s.allLogits[r*v:(r+1)*v])
					r++
				}
			}
		}
	})
	// the packed passes wrote keys and values past base, which the next Step overwrites
	if keep != nil {
		copy(s.logits, keep)
	}
	return err
}

// PrefillOptions configures Prefill.
type PrefillOptions struct {
	CacheScope string // as GenerateOptions.CacheScope
}

// Prefill is Step on an empty state that also uses the model's prompt cache: it copies in the longest cached start of
// ids (from the same scope), reads the rest, and keeps ids' start in the cache for later calls, as Generate does with
// a prompt. cached is how many tokens came from the cache.
func (st *State) Prefill(ctx context.Context, ids []int, o PrefillOptions) (cached int, err error) {
	if len(st.toks) != 0 {
		return 0, errors.New("engine: Prefill needs an empty state")
	}
	if len(ids) == 0 {
		return 0, errors.New("engine: empty prompt")
	}
	if err := st.check(len(ids)); err != nil {
		return 0, err
	}
	if e, n := st.m.lookupPrefix(o.CacheScope, ids); e != nil {
		st.m.restorePrefix(st.s, e, n)
		st.toks = append(st.toks, ids[:n]...)
		cached = n
	}
	if _, err := st.stepPrompt(ctx, ids[cached:]); err != nil {
		return cached, err
	}
	st.m.storePrefix(o.CacheScope, st.s, ids, "")
	return cached, nil
}

// stepPrompt is Step in passes of prefillBatch tokens counted from the start of the prompt read so far, as Generate
// and Classify read prompts.
func (st *State) stepPrompt(ctx context.Context, tokens []int) ([]float32, error) {
	b := st.batch
	st.batch = min(b, prefillBatch)
	defer func() { st.batch = b }()
	return st.Step(ctx, tokens...)
}

// Fork returns a new State holding the same tokens and logits, independent from then on: what either reads doesn't
// change the other. The keys and values are copied (in memory, and on a GPU from the shared CPU copy), so a fork costs
// one copy of the cache so far; reading the same tokens after a fork gives the same bits as without it.
func (st *State) Fork() (*State, error) {
	if st.s == nil {
		return nil, ErrStateClosed
	}
	s, err := st.m.getState(st.s.maxCtx, st.s.maxN)
	if err != nil {
		return nil, err
	}
	st.m.copyKV(s, st.s, len(st.toks))
	f := &State{m: st.m, s: s, batch: st.batch, toks: append([]int(nil), st.toks...)}
	if st.logits != nil {
		copy(s.logits, st.logits)
		f.logits = s.logits
	}
	return f, nil
}

// Rewind drops the last n tokens, as if they were never read. Logits is nil until the next Step. On a compact KV
// cache (LoadOptions.KVCacheType) with block formats, rewinding into a finished block and reading again re-quantizes
// that block, so the rest may differ in rounding from never having read the dropped tokens.
func (st *State) Rewind(n int) error {
	if st.s == nil {
		return ErrStateClosed
	}
	if n < 0 || n > len(st.toks) {
		return fmt.Errorf("engine: can't rewind %d of %d tokens", n, len(st.toks))
	}
	if n == 0 {
		return nil
	}
	st.toks = st.toks[:len(st.toks)-n]
	st.logits = nil
	return nil
}

// Save writes the state's tokens and keys and values to w, in the thread snapshot format (SaveThread). With o.Full a
// loaded state continues bit for bit like this one.
func (st *State) Save(w io.Writer, o SnapshotOptions) error {
	if st.s == nil {
		return ErrStateClosed
	}
	n := len(st.toks)
	e := &prefixEntry{tokens: st.toks, k: make([][]float32, st.m.hp.layers), v: make([][]float32, st.m.hp.layers)}
	for li := range e.k {
		e.k[li], e.v[li] = st.m.saveKV(st.s, li, n)
	}
	return st.m.writeSnapshot(w, snapshotThread, []snapEntry{{label: "state", e: e}}, o)
}

// LoadState reads a state Save (or SaveThread) wrote. Its last token is read again, so Logits is ready.
func (m *Model) LoadState(ctx context.Context, r io.Reader, o StateOptions) (*State, error) {
	info, entries, err := m.readSnapshot(r, snapshotThread)
	if err != nil {
		return nil, err
	}
	if m.det && !info.Full {
		return nil, errHalfSnapshot
	}
	if len(entries) != 1 || len(entries[0].e.tokens) == 0 {
		return nil, ErrSnapshotCorrupt
	}
	e := entries[0].e
	n := len(e.tokens)
	if o.MaxContext <= 0 {
		o.MaxContext = min(m.hp.ctx, 4096)
	}
	o.MaxContext = max(o.MaxContext, n)
	st, err := m.NewState(o)
	if err != nil {
		return nil, err
	}
	for li := range e.k {
		m.loadKV(st.s, li, e.k[li], e.v[li], n, n-1)
	}
	m.kvChanged(st.s)
	st.toks = append(st.toks, e.tokens[:n-1]...)
	if _, err := st.Step(ctx, e.tokens[n-1]); err != nil {
		st.Close()
		return nil, err
	}
	return st, nil
}

// Close gives the state's memory back to the model. The state can't be used again.
func (st *State) Close() {
	if st.s == nil {
		return
	}
	st.m.putState(st.s)
	st.s, st.toks, st.logits = nil, nil, nil
}

// LogprobOf is tok's log-probability (natural log) in logits, with the top most likely tokens (0: none), computed as
// Score and Classify compute it (in deterministic mode, the same bits on every machine).
func (m *Model) LogprobOf(logits []float32, tok, top int) TokenLogprob {
	return logprobOf(logits, tok, top, m.det)
}

// IsEOS reports whether tok ends the model's turn.
func (m *Model) IsEOS(tok int) bool { return m.tok.eos[tok] }

// copyKV copies the first n positions of src's cache into dst, exactly (both states belong to m).
func (m *Model) copyKV(dst, src *state, n int) {
	hp := m.hp
	kvd := hp.kvHeads * hp.headDim
	if src.kv == nil {
		for li := range src.kc {
			for i := 0; i < kvd; i++ {
				copy(dst.kc[li][i*dst.ctxP:i*dst.ctxP+n], src.kc[li][i*src.ctxP:i*src.ctxP+n])
			}
			copy(dst.vc[li][:n*kvd], src.vc[li][:n*kvd])
		}
	} else {
		sc, dc := src.kv, dst.kv
		for li := range sc.k {
			kb := n * hp.kvHeads * sc.keyRow
			copy(dc.k[li][:kb], sc.k[li][:kb])
			vb := kvBytes(sc.vtyp, (n+31)&^31)
			if sc.vtyp == typeF16 {
				vb = kvBytes(sc.vtyp, n)
			}
			for ch := 0; ch < kvd; ch++ {
				copy(dc.v[li][ch*dc.vRow:ch*dc.vRow+vb], sc.v[li][ch*sc.vRow:ch*sc.vRow+vb])
			}
			if sc.stage != nil {
				copy(dc.stage[li], sc.stage[li])
				dc.stageBlk[li] = sc.stageBlk[li]
			}
		}
	}
	m.kvChanged(dst)
}

// kvChanged tells the GPU that s's cache was written outside a forward pass, so a discrete GPU uploads it before the
// next one.
func (m *Model) kvChanged(s *state) {
	if m.gpu == nil {
		return
	}
	m.gpu.mu.Lock()
	if gs := m.gpu.states[s]; gs != nil {
		gs.lastEnd = -1
	}
	m.gpu.mu.Unlock()
}

// Softmax is the softmax Classify uses over label scores (portable exp: the same bits on every machine).
func Softmax(x []float64) []float64 { return softmax64(x) }
