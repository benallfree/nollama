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
// The KV cache stays in the state's kc/vc slices as the CPU's source of truth, so the prompt cache and threads work
// unchanged. On unified memory those slices point at the GPU's buffers. On a discrete GPU the cache lives on the
// device too: each call copies its new keys and values back into kc/vc (one small read per layer), and a call that
// doesn't continue where the state's last one ended (a restored prefix, a rewind) uploads kc/vc first.
//
// First cut: the f32 KV cache and weights the backend supports. Compact caches return errGPUUnsupported, and the
// caller uses forward.
type gpuModel struct {
	m        *Model
	b        gpuBackend
	normsBuf gpuBuf
	attnNorm []gpuRef // per layer
	ffnNorm  []gpuRef
	qNorm    []gpuRef
	kNorm    []gpuRef
	outNorm  gpuRef
	shared   bool // the KV cache is the GPU's memory (unified memory)

	mu     sync.Mutex
	states map[*state]*gpuState
}

var errGPUUnsupported = errors.New("engine: not supported on the GPU yet")

// newGPUModel opens a GPU for m, or explains why m can't run on one.
func newGPUModel(m *Model, det bool) (*gpuModel, error) {
	if m.kvTyp != typeF32 {
		return nil, fmt.Errorf("engine: GPU needs the f32 KV cache, not %s", kvTypeName(m.kvTyp))
	}
	b, err := openGPU(m.g, det)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*gpuModel, error) {
		b.close()
		return nil, err
	}
	ws := []*weight{m.out}
	for _, l := range m.layers {
		ws = append(ws, l.wq, l.wk, l.wv, l.wo, l.wgate, l.wup, l.wdown)
	}
	lo := uintptr(unsafe.Pointer(unsafe.SliceData(m.g.data)))
	hi := lo + uintptr(len(m.g.data))
	for _, w := range ws {
		if !b.supports(w.typ) {
			return fail(fmt.Errorf("engine: %s doesn't support %s weights", b.name(), typeName(w.typ)))
		}
		if p := uintptr(unsafe.Pointer(unsafe.SliceData(w.data))); p < lo || p+uintptr(len(w.data)) > hi {
			return fail(errors.New("engine: GPU needs the weights in the model file as it is"))
		}
	}

	// All norm weights in one buffer.
	g := &gpuModel{m: m, b: b, states: map[*state]*gpuState{}, shared: b.unified() && !gpuForceDiscrete}
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

// gpuForceDiscrete makes gpuModel treat a unified-memory GPU as discrete (tests run the discrete path on a Mac).
var gpuForceDiscrete = false

// gpuState is a state's buffers on the GPU. On unified memory its KV cache buffers back the state's kc/vc slices.
type gpuState struct {
	bufs                       []gpuBuf
	x, an, q, k, v, att, tmp   gpuRef
	gate, up, hf, cs, sn, mask gpuRef
	scratch, logits, act       gpuRef // act: deterministic backends' quantized activations
	allLogits                  gpuRef // maxN*vocab, allocated on the first Score
	kc, vc                     []gpuRef
	kStage, vStage             gpuRef // discrete GPUs: each layer's new keys and values, [layers][maxN][kvd]
	maxN, ctxP, attnChunk      int
	lastEnd                    int // discrete GPUs: where the last call ended (-1: the device cache is stale)
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
	gs := &gpuState{maxN: s.maxN, ctxP: s.ctxP, lastEnd: -1}
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
		{&gs.act, gpuActBytes(n, max(hp.dim, qd, hp.ffn)) / 4},
	} {
		r, err := alloc(a.floats)
		if err != nil {
			gs.release()
			return nil, err
		}
		*a.r = r
	}
	if !g.shared {
		for _, r := range []*gpuRef{&gs.kStage, &gs.vStage} {
			var err error
			if *r, err = alloc(len(g.m.layers) * n * kvd); err != nil {
				gs.release()
				return nil, err
			}
		}
	}
	// The KV cache in GPU memory; on unified memory the state's slices then point at it.
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
			if g.shared {
				mem := g.b.mapped(r.buf)
				f := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(mem))), len(c.host[li]))
				copy(f, c.host[li])
				c.host[li] = f
			}
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

// forget frees s's GPU buffers when the model drops s. s mustn't be used again.
func (g *gpuModel) forget(s *state) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if gs := g.states[s]; gs != nil {
		gs.release()
		delete(g.states, s)
	}
}

// close frees the GPU. States it ran get their KV caches back in ordinary memory, so they stay usable.
func (g *gpuModel) close() {
	g.mu.Lock()
	defer g.mu.Unlock()
	for s, gs := range g.states {
		for li := range s.kc {
			if !g.shared {
				break
			}
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
	if s.kv != nil {
		return errGPUUnsupported
	}
	gs, err := g.stateFor(s)
	if err != nil {
		return err
	}
	if s.allLogits != nil && gs.allLogits.buf == nil {
		buf, err := g.b.alloc(4 * gs.maxN * g.m.hp.vocab)
		if err != nil {
			return err
		}
		g.mu.Lock()
		gs.bufs = append(gs.bufs, buf)
		g.mu.Unlock()
		gs.allLogits = gpuRef{buf: buf}
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
	if !g.shared && pos != gs.lastEnd { // the device cache may not match kc/vc: upload them
		for li := range m.layers {
			b.write(gs.kc[li], floatBytes(s.kc[li]))
			b.write(gs.vc[li], floatBytes(s.vc[li]))
		}
	}

	bt := b.batch()
	for li := range m.layers {
		l := &m.layers[li]
		bt.rmsnorm(gs.x, g.attnNorm[li], gs.an, n, dim, hp.eps)
		bt.matmul(l.wq, gs.an, gs.q, n, gs.act)
		bt.matmul(l.wk, gs.an, gs.k, n, gs.act)
		bt.matmul(l.wv, gs.an, gs.v, n, gs.act)
		bt.rmsnorm(gs.q, g.qNorm[li], gs.q, n*hp.heads, hd, hp.eps)
		bt.rmsnorm(gs.k, g.kNorm[li], gs.k, n*hp.kvHeads, hd, hp.eps)
		bt.rope(gs.q, gs.cs, gs.sn, n, hp.heads, hd)
		bt.rope(gs.k, gs.cs, gs.sn, n, hp.kvHeads, hd)
		bt.storeKV(gs.k, gs.v, gs.kc[li], gs.vc[li], n, pos, kvd, gs.ctxP)
		if !g.shared {
			bt.copy(gs.kStage.at(4*li*gs.maxN*kvd), gs.k, n*kvd)
			bt.copy(gs.vStage.at(4*li*gs.maxN*kvd), gs.v, n*kvd)
		}
		for t0 := 0; t0 < n; t0 += gs.attnChunk {
			c := min(gs.attnChunk, n-t0)
			bt.attention(gpuAttention{
				q: gs.q.at(4 * t0 * qd), out: gs.att.at(4 * t0 * qd), k: gs.kc[li], v: gs.vc[li],
				scratch: gs.scratch, mask: gs.mask.at(8 * t0), kvType: typeF32,
				n: c, pos: pos + t0, heads: hp.heads, kvHeads: hp.kvHeads, headDim: hd, ctxP: gs.ctxP,
			})
		}
		bt.matmul(l.wo, gs.att, gs.tmp, n, gs.act)
		bt.add(gs.x, gs.tmp, n*dim)
		bt.rmsnorm(gs.x, g.ffnNorm[li], gs.an, n, dim, hp.eps)
		bt.matmul(l.wgate, gs.an, gs.gate, n, gs.act)
		bt.matmul(l.wup, gs.an, gs.up, n, gs.act)
		bt.swiglu(gs.gate, gs.up, gs.hf, n*hp.ffn)
		bt.matmul(l.wdown, gs.hf, gs.tmp, n, gs.act)
		bt.add(gs.x, gs.tmp, n*dim)
	}
	last := gs.x.at(4 * (n - 1) * dim)
	defer func() {
		if !g.shared { // bring kc/vc up to date with the keys and values this call wrote
			k, v := make([]float32, n*kvd), make([]float32, n*kvd)
			for li := range m.layers {
				g.readFloats(gs.kStage.at(4*li*gs.maxN*kvd), k)
				g.readFloats(gs.vStage.at(4*li*gs.maxN*kvd), v)
				for t := 0; t < n; t++ {
					for i := 0; i < kvd; i++ {
						s.kc[li][i*gs.ctxP+pos+t] = k[t*kvd+i]
					}
				}
				copy(s.vc[li][pos*kvd:(pos+n)*kvd], v)
			}
			gs.lastEnd = pos + n
		}
	}()
	switch {
	case s.hidden != nil: // Embed: the last token's normed hidden state, no LM head
		bt.rmsnorm(last, g.outNorm, gs.an, 1, dim, hp.eps)
		if err := bt.submit(); err != nil {
			return err
		}
		g.readFloats(gs.an, s.hidden[:dim])
	case s.allLogits != nil: // Score: every position's logits
		bt.rmsnorm(gs.x, g.outNorm, gs.an, n, dim, hp.eps)
		bt.matmul(m.out, gs.an, gs.allLogits, n, gs.act)
		if err := bt.submit(); err != nil {
			return err
		}
		g.readFloats(gs.allLogits, s.allLogits[:n*hp.vocab])
		copy(s.logits, s.allLogits[(n-1)*hp.vocab:n*hp.vocab])
	default:
		bt.rmsnorm(last, g.outNorm, gs.an, 1, dim, hp.eps)
		bt.matmul(m.out, gs.an, gs.logits, 1, gs.act)
		if err := bt.submit(); err != nil {
			return err
		}
		g.readFloats(gs.logits, s.logits)
	}
	return nil
}

// readFloats copies len(dst) floats from the device.
func (g *gpuModel) readFloats(src gpuRef, dst []float32) {
	b := g.b
	if g.shared { // straight from the shared memory
		mem := b.mapped(src.buf)[src.off:]
		copy(dst, unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(mem))), len(dst)))
		return
	}
	raw := make([]byte, 4*len(dst))
	b.read(src, raw)
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
}

func floatBytes(x []float32) []byte {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}
