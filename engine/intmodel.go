package engine

// Integer models: a Qwen2.5 forward pass in integer arithmetic only. Every step's result is independent of the order
// of operations, the instruction set and the thread count, so every machine computes the same next-token
// distribution, bit for bit, as exact integer counts. That is what an arithmetic coder needs to decode on one machine
// what another encoded (github.com/benallfree/compress13k's signal uses it), and it gives sampling that matches
// across machines. The model is a converted GGUF (architecture "qwen2-intcodec": int8 weights with an integer scale
// per 64 inputs, integer biases and norms, lookup tables, and the tokenizer); ConvertIntModel makes it from the
// official bf16 weights, deterministically. The arithmetic is defined in compress13k's SPEC.md, whose reference
// implementation produced the conformance data in testdata/intmodel.

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"path/filepath"
	"runtime"
	"unsafe"
)

const (
	intTotal = 1 << 28 // frequency total
	intNA    = 16 * 1024
	intNF    = 24 * 1024
	intNS    = 16 * 1024
	intLimKV = 1<<22 - 1
)

type intMat struct {
	q       []int8  // [out][in]
	s       []int16 // [out][in/64]
	e       int     // w = q * s * 2^e
	out, in int
}

type intLayer struct {
	qkv, o, gu, d intMat
	b, g1, g2     []int64
}

// IntModel is a converted model for the integer codec.
type IntModel struct {
	g                       *ggufFile
	tok                     *tokenizer
	path                    string
	D, I, L, nq, nkv, hd, V int
	bos                     int
	sc                      int64 // attention score scale round(2^16/sqrt(hd)), folded into q
	emb, head               intMat
	layers                  []intLayer
	gf                      []int64
	ropeC, ropeS            []int16
	expA, expF, sig         []int64
	pool                    *pool
}

func viewI8(b []byte) []int8 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), len(b))
}

func viewI16(b []byte) []int16 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int16)(unsafe.Pointer(&b[0])), len(b)/2)
}

func viewI32(b []byte) []int32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func viewI64(b []byte) []int64 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int64)(unsafe.Pointer(&b[0])), len(b)/8)
}

func i32to64(a []int32) []int64 {
	out := make([]int64, len(a))
	for i, v := range a {
		out[i] = int64(v)
	}
	return out
}

// LoadIntModel opens a converted model (GGUF, architecture "qwen2-intcodec"). threads <= 0 means all CPUs.
func LoadIntModel(path string, threads int) (*IntModel, error) {
	if !littleEndian() {
		return nil, errors.New("intmodel: needs a little-endian CPU")
	}
	g, err := openGGUF(path)
	if err != nil {
		return nil, err
	}
	m, err := newIntModel(g, path, threads)
	if err != nil {
		g.close()
		return nil, err
	}
	return m, nil
}

func littleEndian() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}

func newIntModel(g *ggufFile, path string, threads int) (*IntModel, error) {
	if a := g.getString("general.architecture"); a != "qwen2-intcodec" {
		return nil, fmt.Errorf("intmodel: %s is not a converted model (architecture %q)", filepath.Base(path), a)
	}
	if v, _ := g.getInt("intcodec.spec_version"); v != 1 {
		return nil, fmt.Errorf("intmodel: spec version %d, want 1", v)
	}
	m := &IntModel{g: g, path: path}
	geti := func(k string) int { v, _ := g.getInt("intcodec." + k); return v }
	m.D, m.I, m.L = geti("hidden_size"), geti("intermediate_size"), geti("num_hidden_layers")
	m.nq, m.nkv, m.V, m.bos = geti("num_attention_heads"), geti("num_key_value_heads"), geti("vocab_size"), geti("bos_token_id")
	if m.D == 0 || m.nq == 0 || m.nkv == 0 || m.L == 0 {
		return nil, errors.New("intmodel: incomplete config")
	}
	m.hd = m.D / m.nq
	m.sc = int64(math.Round(65536 / math.Sqrt(float64(m.hd))))
	tensor := func(name string, typ uint32) (*tensorInfo, error) {
		t := g.Tensors[name]
		if t == nil {
			return nil, fmt.Errorf("intmodel: missing tensor %s", name)
		}
		if t.Type != typ || t.Data == nil {
			return nil, fmt.Errorf("intmodel: tensor %s has type %s", name, typeName(t.Type))
		}
		return t, nil
	}
	mat := func(name string) (intMat, error) {
		q, err := tensor(name+".q", typeI8)
		if err != nil {
			return intMat{}, err
		}
		s, err := tensor(name+".s", typeI16)
		if err != nil {
			return intMat{}, err
		}
		e, ok := g.getInt("intcodec.exp." + name)
		if !ok {
			return intMat{}, fmt.Errorf("intmodel: missing exponent for %s", name)
		}
		in, out := int(q.Dims[0]), q.rows()
		if in%128 != 0 || len(s.Data)/2 != out*in/64 {
			return intMat{}, fmt.Errorf("intmodel: bad shape for %s", name)
		}
		return intMat{q: viewI8(q.Data), s: viewI16(s.Data), e: e, out: out, in: in}, nil
	}
	i64 := func(name string) ([]int64, error) {
		t, err := tensor(name, typeI64)
		if err != nil {
			return nil, err
		}
		return viewI64(t.Data), nil
	}
	var err error
	if m.emb, err = mat("emb"); err != nil {
		return nil, err
	}
	m.head = m.emb
	if _, ok := g.Tensors["head.q"]; ok {
		if m.head, err = mat("head"); err != nil {
			return nil, err
		}
	}
	if m.gf, err = i64("gf"); err != nil {
		return nil, err
	}
	m.layers = make([]intLayer, m.L)
	for l := range m.layers {
		ly := &m.layers[l]
		p := fmt.Sprint(l, ".")
		for _, x := range []struct {
			dst  *intMat
			name string
		}{{&ly.qkv, "qkv"}, {&ly.o, "o"}, {&ly.gu, "gu"}, {&ly.d, "d"}} {
			if *x.dst, err = mat(p + x.name); err != nil {
				return nil, err
			}
		}
		if ly.b, err = i64(p + "bqkv"); err != nil {
			return nil, err
		}
		if ly.g1, err = i64(p + "g1"); err != nil {
			return nil, err
		}
		if ly.g2, err = i64(p + "g2"); err != nil {
			return nil, err
		}
	}
	for _, x := range []struct {
		dst  *[]int16
		name string
	}{{&m.ropeC, "ROPE_C"}, {&m.ropeS, "ROPE_S"}} {
		t, err := tensor(x.name, typeI16)
		if err != nil {
			return nil, err
		}
		*x.dst = viewI16(t.Data)
	}
	for _, x := range []struct {
		dst  *[]int64
		name string
	}{{&m.expA, "EXPA"}, {&m.expF, "EXPF"}, {&m.sig, "SIG"}} {
		t, err := tensor(x.name, typeI32)
		if err != nil {
			return nil, err
		}
		*x.dst = i32to64(viewI32(t.Data))
	}
	if m.tok, err = newTokenizer(g); err != nil {
		return nil, err
	}
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	m.pool = newPool(threads, nil)
	return m, nil
}

// Close releases the model's file mapping.
func (m *IntModel) Close() error { return m.g.close() }

// ---- integer helpers (SPEC §2)

func irsh(x int64, n int) int64 {
	if n <= 0 {
		return x << -n
	}
	return (x + 1<<(n-1)) >> n
}

func bitlen(v int64) int { return bits.Len64(uint64(v)) }

func absMax(x []int64) int64 {
	var mx int64
	for _, v := range x {
		if v < 0 {
			v = -v
		}
		mx = max(mx, v)
	}
	return mx
}

// blockf: 16-bit mantissas with one shared exponent; returns the exponent shift.
func blockf(x []int64, m []int16) int {
	sh := max(0, bitlen(absMax(x))-15)
	for i, v := range x {
		m[i] = int16(min(max(irsh(v, sh), -32767), 32767))
	}
	return sh
}

func isqrt(v int64) int64 {
	r := int64(math.Sqrt(float64(v)))
	for r*r > v {
		r--
	}
	for (r+1)*(r+1) <= v {
		r++
	}
	return r
}

// matvec: acc[j] = sum_g s[j,g] * sum_{k in g} q[j,k] m[k]; then out[j] = rshift(acc[j], to - (W.e + e)).
func (m *IntModel) matvec(w *intMat, x []int16, e, to int, out []int64) {
	ng := w.in / 64
	n := to - (w.e + e)
	m.pool.parallelFor(w.out, 16, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			out[j] = irsh(intDotRow(w.q[j*w.in:(j+1)*w.in], x, w.s[j*ng:(j+1)*ng]), n)
		}
	})
}

// intDotRowGo is the reference for intDotRow: sum_g s[g] * sum_{k<64} q[64g+k]*x[64g+k].
func intDotRowGo(q []int8, x []int16, s []int16) int64 {
	var acc int64
	for gi := range s {
		qg := q[gi*64 : gi*64+64]
		xg := x[gi*64 : gi*64+64]
		var d int32
		for k := 0; k < 64; k++ {
			d += int32(qg[k]) * int32(xg[k])
		}
		acc += int64(d) * int64(s[gi])
	}
	return acc
}

func intDotQKGo(q, k []int32) int64 {
	var acc int64
	for i, v := range q {
		acc += int64(v) * int64(k[i])
	}
	return acc
}

func intAxpyGo(o []int64, w int64, v []int32) {
	for i, x := range v[:len(o)] {
		o[i] += w * int64(x)
	}
}

// rmsnorm returns the mantissas (in xm) and their exponent.
func (m *IntModel) rmsnorm(h, g []int64, xm []int16, y []int64) int {
	y = y[:len(h)] // y is scratch sized for the MLP
	sh := max(0, bitlen(absMax(h))-20)
	var ss int64
	for i, v := range h {
		x := irsh(v, sh)
		y[i] = x
		ss += x * x
	}
	ms := ss/int64(len(h)) + (int64(1)<<max(0, 48-2*sh))/1000000
	r := max(1, isqrt(ms))
	for i := range y {
		y[i] = y[i] * g[i] * (1 << 14) / r // Go's / truncates toward zero
	}
	return blockf(y, xm) - 30
}

// ---- per-sequence state

type intState struct {
	m       *IntModel
	maxlen  int
	pos     int
	K, Vc   []int32 // [L][nkv][maxlen][hd]
	h, y, x []int64
	xm, om  []int16
	pm      []int16
	gu      []int64
	qkv     []int64
	o       []int64
	lg, cum []int64
	scores  [][]int64 // per q head
	headO   [][]int64 // per q head [hd]
	headMQ  [][]int32
	headSh  []int
}

func (m *IntModel) newState(maxlen int) *intState {
	s := &intState{m: m, maxlen: maxlen}
	n := m.L * m.nkv * maxlen * m.hd
	s.K, s.Vc = make([]int32, n), make([]int32, n)
	s.h, s.y, s.x = make([]int64, m.D), make([]int64, max(m.D, m.I)), make([]int64, m.D)
	s.xm, s.om, s.pm = make([]int16, m.D), make([]int16, m.D), make([]int16, m.I)
	s.gu, s.qkv, s.o = make([]int64, 2*m.I), make([]int64, (m.nq+2*m.nkv)*m.hd), make([]int64, m.D)
	s.lg, s.cum = make([]int64, m.V), make([]int64, m.V)
	s.scores, s.headO, s.headMQ = make([][]int64, m.nq), make([][]int64, m.nq), make([][]int32, m.nq)
	s.headSh = make([]int, m.nq)
	for i := range s.scores {
		s.scores[i] = make([]int64, maxlen)
		s.headO[i] = make([]int64, m.hd)
		s.headMQ[i] = make([]int32, m.hd)
	}
	return s
}

func (s *intState) reset() { s.pos = 0 }

func (s *intState) rope(x []int64, nh int) {
	m := s.m
	half := m.hd / 2
	c := m.ropeC[s.pos*half : (s.pos+1)*half]
	sn := m.ropeS[s.pos*half : (s.pos+1)*half]
	for h := 0; h < nh; h++ {
		v := x[h*m.hd : (h+1)*m.hd]
		for j := 0; j < half; j++ {
			x1, x2 := v[j], v[j+half]
			cj, sj := int64(c[j]), int64(sn[j])
			v[j], v[j+half] = irsh(x1*cj-x2*sj, 14), irsh(x2*cj+x1*sj, 14)
		}
	}
}

// step feeds token tok at the current position and fills s.cum with the next token's cumulative frequencies.
func (s *intState) step(tok int) {
	m := s.m
	if s.pos >= s.maxlen {
		panic("intmodel: context full")
	}
	D, hd, nq, nkv := m.D, m.hd, m.nq, m.nkv
	grp := nq / nkv
	// embedding
	ng := D / 64
	q, sc := m.emb.q[tok*D:(tok+1)*D], m.emb.s[tok*ng:(tok+1)*ng]
	for k := 0; k < D; k++ {
		s.h[k] = irsh(int64(q[k])*int64(sc[k/64]), -24-m.emb.e)
	}
	T := s.pos + 1
	for l := range m.layers {
		ly := &m.layers[l]
		e := m.rmsnorm(s.h, ly.g1, s.xm, s.y)
		m.matvec(&ly.qkv, s.xm, e, -24, s.qkv)
		for i := range s.qkv {
			s.qkv[i] += ly.b[i]
		}
		qv := s.qkv[:nq*hd]
		kv := s.qkv[nq*hd : (nq+nkv)*hd]
		vv := s.qkv[(nq+nkv)*hd:]
		s.rope(qv, nq)
		s.rope(kv, nkv)
		for j := 0; j < nkv; j++ {
			base := ((l*nkv+j)*s.maxlen + s.pos) * hd
			for d := 0; d < hd; d++ {
				s.K[base+d] = int32(min(max(irsh(kv[j*hd+d], 12), -intLimKV), intLimKV))
				s.Vc[base+d] = int32(min(max(irsh(vv[j*hd+d], 12), -intLimKV), intLimKV))
			}
		}
		// attention, one q head per task
		m.pool.parallelFor(nq, 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				j := i / grp
				mq := s.headMQ[i]
				qs := qv[i*hd : (i+1)*hd]
				var mx int64
				for d := 0; d < hd; d++ {
					v := qs[d] * m.sc
					if v < 0 {
						v = -v
					}
					mx = max(mx, v)
				}
				sh := max(0, bitlen(mx)-15)
				for d := range mq {
					mq[d] = int32(min(max(irsh(qs[d]*m.sc, sh), -32767), 32767))
				}
				n := 42 - sh
				sc := s.scores[i][:T]
				kb := (l*nkv + j) * s.maxlen * hd
				M := int64(math.MinInt64)
				for t := 0; t < T; t++ {
					v := irsh(intDotQK(mq, s.K[kb+t*hd:kb+(t+1)*hd]), n)
					sc[t] = v
					M = max(M, v)
				}
				O := s.headO[i]
				clear(O)
				var Z int64
				for t := 0; t < T; t++ {
					w := m.expA[min(M-sc[t], intNA-1)]
					if w == 0 {
						continue
					}
					Z += w
					intAxpy(O, w, s.Vc[kb+t*hd:kb+(t+1)*hd])
				}
				for d := 0; d < hd; d++ {
					a := O[d]
					neg := a < 0
					if neg {
						a = -a
					}
					hi := a / Z
					o := hi<<12 + (a-hi*Z)<<12/Z
					if neg {
						o = -o
					}
					s.o[i*hd+d] = o
				}
			}
		})
		e = blockf(s.o, s.om) - 24
		m.matvec(&ly.o, s.om, e, -24, s.x)
		for i := range s.h {
			s.h[i] += s.x[i]
		}
		e = m.rmsnorm(s.h, ly.g2, s.xm, s.y)
		m.matvec(&ly.gu, s.xm, e, -16, s.gu)
		I := m.I
		p := s.y[:I]
		for i := 0; i < I; i++ {
			g, u := s.gu[i], s.gu[I+i]
			a := irsh(g*m.sig[min(max(irsh(g, 6), -intNS), intNS)+intNS], 16)
			p[i] = irsh(a*u, 16)
		}
		e = blockf(p, s.pm) - 16
		m.matvec(&ly.d, s.pm, e, -24, s.x)
		for i := range s.h {
			s.h[i] += s.x[i]
		}
	}
	e := m.rmsnorm(s.h, m.gf, s.xm, s.y)
	m.matvec(&m.head, s.xm, e, -10, s.lg)
	M := int64(math.MinInt64)
	for _, v := range s.lg {
		M = max(M, v)
	}
	var Z int64
	for i, v := range s.lg {
		f := m.expF[min(M-v, intNF-1)]
		s.lg[i] = f
		Z += f
	}
	var c int64
	for i, f := range s.lg {
		c += f*(intTotal-int64(m.V))/Z + 1
		s.cum[i] = c
	}
	s.pos++
}
