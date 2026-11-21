package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

// gpuModel runs a Model's forward pass on a GPU backend: the same steps as forward, recorded into one batch per
// call, with the CPU doing only the embedding lookup, the RoPE tables and reading back the logits.
//
// First cut: unified memory only (the KV cache is shared with the CPU, so the prompt cache and threads work
// unchanged), the f32 KV cache, weights the backend supports, and the plain logits path. Score (every
// position's logits) and Embed (hidden states) return errGPUUnsupported, and the caller uses forward.
type gpuModel struct {
	m        *Model
	b        gpuBackend
	normsBuf gpuBuf
	attnNorm []gpuRef // per layer
	ffnNorm  []gpuRef
	qNorm    []gpuRef
	kNorm    []gpuRef
	outNorm  gpuRef

	mu     sync.Mutex
	states map[*state]*gpuState
}

var errGPUUnsupported = errors.New("engine: not supported on the GPU yet")

// newGPUModel opens a GPU for m, or explains why m can't run on one.
func newGPUModel(m *Model) (*gpuModel, error) {
	if m.kvTyp != typeF32 {
		return nil, fmt.Errorf("engine: GPU needs the f32 KV cache, not %s", kvTypeName(m.kvTyp))
	}
	b, err := openGPU(m.g)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*gpuModel, error) {
		b.close()
		return nil, err
	}
	if !b.unified() {
		return fail(fmt.Errorf("engine: %s: discrete GPUs aren't supported yet", b.name()))
	}
	ws := []*weight{m.out}
	for _, l := range m.layers {
		ws = append(ws, l.wq, l.wk, l.wv, l.wo, l.wgate, l.wup, l.wdown)
	}
	for _, w := range ws {
		if !b.supports(w.typ) {
			return fail(fmt.Errorf("engine: %s doesn't support %s weights", b.name(), typeName(w.typ)))
		}
	}

	// All norm weights in one buffer.
	g := &gpuModel{m: m, b: b, states: map[*state]*gpuState{}}
	var all []float32
	var offs []int
	add := func(v []float32) {
		offs = append(offs, 4*len(all))
		all = append(all, v...)
	}
	for _, l := range m.layers {
		add(l.attnNorm)
		add(l.ffnNorm)
		add(l.qNorm)
		add(l.kNorm)
	}
	add(m.norm)
	if g.normsBuf, err = b.alloc(4 * len(all)); err != nil {
		return fail(err)
	}
	raw := make([]byte, 4*len(all))
	for i, v := range all {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	base := gpuRef{buf: g.normsBuf}
	b.write(base, raw)
	for li := range m.layers {
		g.attnNorm = append(g.attnNorm, base.at(offs[4*li]))
		g.ffnNorm = append(g.ffnNorm, base.at(offs[4*li+1]))
		g.qNorm = append(g.qNorm, base.at(offs[4*li+2]))
		g.kNorm = append(g.kNorm, base.at(offs[4*li+3]))
	}
	g.outNorm = base.at(offs[len(offs)-1])
	return g, nil
}

// gpuState is a state's buffers on the GPU. Its KV cache buffers back the state's kc/vc slices.
type gpuState struct {
	bufs                       []gpuBuf
	x, an, q, k, v, att, tmp   gpuRef
	gate, up, hf, cs, sn, mask gpuRef
	scratch, logits            gpuRef
	kc, vc                     []gpuRef
	maxN, ctxP, attnChunk      int
}

// attnScratchFloats bounds the attention scores' scratch; longer batches run attention in chunks of tokens.
const attnScratchFloats = 16 << 20

func (g *gpuModel) stateFor(s *state) (*gpuState, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if gs := g.states[s]; gs != nil {
		return gs, nil
	}
	hp := g.m.hp
	qd, kvd := hp.heads*hp.headDim, hp.kvHeads*hp.headDim
	gs := &gpuState{maxN: s.maxN, ctxP: s.ctxP}
	gs.attnChunk = max(1, min(s.maxN, attnScratchFloats/(hp.heads*s.ctxP)))
	alloc := func(floats int) (gpuRef, error) {
		b, err := g.b.alloc(4 * max(floats, 1))
		if err != nil {
			return gpuRef{}, err
		}
		gs.bufs = append(gs.bufs, b)
		return gpuRef{buf: b}, nil
	}
	n := s.maxN
	for _, a := range []struct {
		r      *gpuRef
		floats int
	}{
		{&gs.x, n * hp.dim}, {&gs.an, n * hp.dim}, {&gs.q, n * qd}, {&gs.k, n * kvd}, {&gs.v, n * kvd},
		{&gs.att, n * qd}, {&gs.tmp, n * hp.dim}, {&gs.gate, n * hp.ffn}, {&gs.up, n * hp.ffn}, {&gs.hf, n * hp.ffn},
		{&gs.cs, n * hp.headDim / 2}, {&gs.sn, n * hp.headDim / 2}, {&gs.mask, 2 * n},
		{&gs.scratch, gs.attnChunk * hp.heads * s.ctxP}, {&gs.logits, hp.vocab},
	} {
		r, err := alloc(a.floats)
		if err != nil {
			gs.release()
			return nil, err
		}
		*a.r = r
	}
	// The KV cache moves into GPU memory; the state's slices then point at it.
	for li := range g.m.layers {
		for _, c := range []struct {
			refs *[]gpuRef
			host [][]float32
		}{{&gs.kc, s.kc}, {&gs.vc, s.vc}} {
			r, err := alloc(len(c.host[li]))
			if err != nil {
				gs.release()
				return nil, err
			}
			mem := g.b.mapped(r.buf)
			f := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(mem))), len(c.host[li]))
			copy(f, c.host[li])
			c.host[li] = f
			*c.refs = append(*c.refs, r)
		}
	}
	g.states[s] = gs
	return gs, nil
}

func (gs *gpuState) release() {
	for _, b := range gs.bufs {
		b.release()
	}
	gs.bufs = nil
}

// close frees the GPU. States it ran get their KV caches back in ordinary memory, so they stay usable.
func (g *gpuModel) close() {
	g.mu.Lock()
	defer g.mu.Unlock()
	for s, gs := range g.states {
		for li := range s.kc {
			s.kc[li] = append([]float32(nil), s.kc[li]...)
			s.vc[li] = append([]float32(nil), s.vc[li]...)
		}
		gs.release()
	}
	g.states = nil
	g.normsBuf.release()
	g.b.close()
}

// forward is Model.forward on the GPU: it runs tokens at positions pos.. and fills s.logits for the last one.
func (g *gpuModel) forward(s *state, tokens []int, pos int) error {
	if s.allLogits != nil || s.hidden != nil || s.kv != nil {
		return errGPUUnsupported
	}
	gs, err := g.stateFor(s)
	if err != nil {
		return err
	}
	m, hp, b := g.m, g.m.hp, g.b
	n := len(tokens)
	dim, hd := hp.dim, hp.headDim
	qd, kvd, half := hp.heads*hd, hp.kvHeads*hd, hd/2

	// CPU: embeddings, RoPE tables, the packed-sequence mask.
	x := make([]float32, n*dim)
	cs, sn := make([]float32, n*half), make([]float32, n*half)
	for t, id := range tokens {
		dequantRow(m.embd.typ, m.embd.row(id), x[t*dim:(t+1)*dim])
		pp := pos + t
		if s.ropePos != nil {
			pp = s.ropePos[t]
		}
		for i := 0; i < half; i++ {
			si, co := sincos(float64(pp) * m.invFrq[i])
			cs[t*half+i], sn[t*half+i] = float32(co), float32(si)
		}
	}
	mask := make([]byte, 8*n)
	if s.packStart != nil {
		for t := 0; t < n; t++ {
			binary.LittleEndian.PutUint32(mask[8*t:], uint32(s.packBase))
			binary.LittleEndian.PutUint32(mask[8*t+4:], uint32(s.packStart[t]))
		}
	}
	b.write(gs.x, floatBytes(x))
	b.write(gs.cs, floatBytes(cs))
	b.write(gs.sn, floatBytes(sn))
	b.write(gs.mask, mask)

	bt := b.batch()
	for li := range m.layers {
		l := &m.layers[li]
		bt.rmsnorm(gs.x, g.attnNorm[li], gs.an, n, dim, hp.eps)
		bt.matmul(l.wq, gs.an, gs.q, n)
		bt.matmul(l.wk, gs.an, gs.k, n)
		bt.matmul(l.wv, gs.an, gs.v, n)
		bt.rmsnorm(gs.q, g.qNorm[li], gs.q, n*hp.heads, hd, hp.eps)
		bt.rmsnorm(gs.k, g.kNorm[li], gs.k, n*hp.kvHeads, hd, hp.eps)
		bt.rope(gs.q, gs.cs, gs.sn, n, hp.heads, hd)
		bt.rope(gs.k, gs.cs, gs.sn, n, hp.kvHeads, hd)
		bt.storeKV(gs.k, gs.v, gs.kc[li], gs.vc[li], n, pos, kvd, gs.ctxP)
		for t0 := 0; t0 < n; t0 += gs.attnChunk {
			c := min(gs.attnChunk, n-t0)
			bt.attention(gpuAttention{
				q: gs.q.at(4 * t0 * qd), out: gs.att.at(4 * t0 * qd), k: gs.kc[li], v: gs.vc[li],
				scratch: gs.scratch, mask: gs.mask.at(8 * t0), kvType: typeF32,
				n: c, pos: pos + t0, heads: hp.heads, kvHeads: hp.kvHeads, headDim: hd, ctxP: gs.ctxP,
			})
		}
		bt.matmul(l.wo, gs.att, gs.tmp, n)
		bt.add(gs.x, gs.tmp, n*dim)
		bt.rmsnorm(gs.x, g.ffnNorm[li], gs.an, n, dim, hp.eps)
		bt.matmul(l.wgate, gs.an, gs.gate, n)
		bt.matmul(l.wup, gs.an, gs.up, n)
		bt.swiglu(gs.gate, gs.up, gs.hf, n*hp.ffn)
		bt.matmul(l.wdown, gs.hf, gs.tmp, n)
		bt.add(gs.x, gs.tmp, n*dim)
	}
	last := gs.x.at(4 * (n - 1) * dim)
	bt.rmsnorm(last, g.outNorm, gs.an, 1, dim, hp.eps)
	bt.matmul(m.out, gs.an, gs.logits, 1)
	if err := bt.submit(); err != nil {
		return err
	}
	raw := make([]byte, 4*hp.vocab)
	b.read(gs.logits, raw)
	for i := range s.logits {
		s.logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return nil
}

func floatBytes(x []float32) []byte {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}
