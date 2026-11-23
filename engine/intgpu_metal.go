package engine

// Integer models on Metal (gpu_int.metal): a session's KV cache and activations live in shared buffers, the model
// file is wrapped zero-copy, and one token is one prepared list of dispatches. Integer arithmetic throughout, so the
// distributions are the CPU's, bit for bit.

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
)

//go:embed gpu_int.metal
var metalIntSource string

var metalIntKernels = []string{"int_embed", "int_norm", "int_matvec", "int_qkv", "int_blockf", "int_mlp", "int_table1",
	"int_table2", "int_table3", "int_table4", "int_attn1", "int_attn2", "int_attn3", "int_gemm", "int_attnb64", "int_attnb128",
	"int_counts", "int_bscores64", "int_bscores128", "int_bpv64", "int_bpv128", "int_bcomb"}

const intTG = 1024 // threads per threadgroup int_qkv and int_blockf assume

// intMetal is a model's Metal state: the device, its pipelines and the wrapped model file.
type intMetal struct {
	dev  *metal.Device
	ps   map[string]*metal.Pipeline
	w    *metal.Buffer
	base uintptr
}

func openIntMetal(m *IntModel) (*intMetal, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if !dev.Unified {
		dev.Close()
		return nil, errors.New("metal: GPUs without unified memory aren't supported")
	}
	ps, err := dev.Compile(metalIntSource, metalIntKernels...)
	if err != nil {
		dev.Close()
		return nil, err
	}
	for _, k := range []string{"int_qkv", "int_blockf", "int_counts"} {
		if ps[k].MaxThreads < intTG {
			dev.Close()
			return nil, fmt.Errorf("metal: %s runs at most %d threads per group, needs %d", k, ps[k].MaxThreads, intTG)
		}
	}
	w, err := dev.Wrap(m.g.data)
	if err != nil {
		dev.Close()
		return nil, err
	}
	return &intMetal{dev: dev, ps: ps, w: w, base: uintptr(unsafe.Pointer(unsafe.SliceData(m.g.data)))}, nil
}

func (g *intMetal) close() {
	g.w.Release()
	g.dev.Close()
}

func (g *intMetal) name() string { return "metal (" + g.dev.Name + ")" }

// file is a kernel argument pointing into the wrapped model file.
func (g *intMetal) file(p unsafe.Pointer) metal.Arg {
	return metal.Arg{Buf: g.w, Offset: int(uintptr(p) - g.base)}
}

func i32b(vs ...int) []byte {
	b := make([]byte, 4*len(vs))
	for i, v := range vs {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(v)))
	}
	return b
}

// intMetalState is one session on the GPU.
type intMetalState struct {
	g      *intMetal
	maxlen int
	params *metal.Buffer // tok, pos
	cum    []int64       // the table, as the CPU sees it
	ds     []metal.Dispatch
	bufs   []*metal.Buffer
	// kept for batches (intgpu_batch.go)
	m       *IntModel
	kc, vc  *metal.Buffer
	kvLayer int
	b       *intBatch
}

const (
	intACH = 256  // positions per attention chunk (ACH in gpu_int.metal)
	intTCH = 2048 // table values per threadgroup (TCH)
)

func (g *intMetal) newState(m *IntModel, maxlen int) (*intMetalState, error) {
	s := &intMetalState{g: g, maxlen: maxlen, m: m}
	nb := s.newBuffer
	D, I, V, hd, nq, nkv, L := m.D, m.I, m.V, m.hd, m.nq, m.nkv, m.L
	qkvN := (nq + 2*nkv) * hd
	accN := max(qkvN, D, 2*I, V)
	var err error
	maxch := (maxlen + intACH - 1) / intACH
	tch := (V + intTCH - 1) / intTCH
	var h, mb, ex, acc, mq, shq, kc, vc, sb, o, p, lg, cum, cmax, zpart, opart, tpart *metal.Buffer
	for _, x := range []struct {
		dst **metal.Buffer
		n   int
	}{{&s.params, 8}, {&h, 8 * D}, {&mb, 4 * max(D, I, nq*hd)}, {&ex, 16}, {&acc, 8 * accN}, {&mq, 4 * nq * hd},
		{&shq, 4 * nq}, {&kc, 4 * (L*nkv*maxlen + intACH) * hd}, {&vc, 4 * (L*nkv*maxlen + intACH) * hd}, {&sb, 8 * nq * maxlen},
		{&o, 8 * nq * hd}, {&p, 8 * I}, {&lg, 8 * V}, {&cum, 8 * V}, {&cmax, 8 * nq * maxch}, {&zpart, 8 * nq * maxch},
		{&opart, 8 * nq * maxch * hd}, {&tpart, 8 * 3 * tch}} {
		if *x.dst, err = nb(x.n); err != nil {
			s.release()
			return nil, err
		}
	}
	s.cum = unsafe.Slice((*int64)(unsafe.Pointer(unsafe.SliceData(cum.Bytes()))), V)
	B := func(b *metal.Buffer) metal.Arg { return metal.Arg{Buf: b} }
	BO := func(b *metal.Buffer, off int) metal.Arg { return metal.Arg{Buf: b, Offset: off} }
	C, F := cArg, g.farg
	one := func(k string, args ...metal.Arg) metal.Dispatch {
		return metal.Dispatch{P: g.ps[k], Args: args, Groups: metal.Size{W: 1, H: 1, D: 1}, Threads: metal.Size{W: intTG, H: 1, D: 1}}
	}
	grid := func(k string, n int, args ...metal.Arg) metal.Dispatch {
		return metal.Dispatch{P: g.ps[k], Args: args, Groups: metal.Size{W: uint((n + 255) / 256), H: 1, D: 1},
			Threads: metal.Size{W: 256, H: 1, D: 1}}
	}
	tbl := func(k string, args ...metal.Arg) metal.Dispatch {
		return metal.Dispatch{P: g.ps[k], Args: args, Groups: metal.Size{W: uint(tch), H: 1, D: 1},
			Threads: metal.Size{W: 256, H: 1, D: 1}}
	}
	matvec := func(w *intMat, x *metal.Buffer, einSlot, to int, out *metal.Buffer) metal.Dispatch {
		return metal.Dispatch{P: g.ps["int_matvec"], Args: []metal.Arg{F(w.q), F(w.s), B(x), B(out), B(ex),
			C(w.out), C(w.in), C(w.e), C(einSlot), C(to)},
			Groups: metal.Size{W: uint((w.out + 7) / 8), H: 1, D: 1}, Threads: metal.Size{W: 256, H: 1, D: 1}}
	}
	norm := func(add int, E, einSlot int, gw []int64) metal.Dispatch {
		d := one("int_norm", B(h), B(acc), F(gw), B(mb), B(ex), C(D), C(E), C(einSlot), C(0), C(add), C(0, 0))
		d.Threads.W = 256
		return d
	}
	qkvArgs := make([]byte, 24)
	binary.LittleEndian.PutUint32(qkvArgs[0:], uint32(nq))
	binary.LittleEndian.PutUint32(qkvArgs[4:], uint32(nkv))
	binary.LittleEndian.PutUint32(qkvArgs[8:], uint32(hd))
	binary.LittleEndian.PutUint32(qkvArgs[12:], uint32(maxlen))
	binary.LittleEndian.PutUint64(qkvArgs[16:], uint64(m.sc))
	attnArgs := i32b(nq, nkv, hd, maxlen, intNA, maxch)
	kvLayer := 4 * nkv * maxlen * hd
	s.kc, s.vc, s.kvLayer = kc, vc, kvLayer
	rawA, rawF, rawS := m.g.Tensors["EXPA"].Data, m.g.Tensors["EXPF"].Data, m.g.Tensors["SIG"].Data

	ds := []metal.Dispatch{
		grid("int_embed", D, F(m.emb.q), F(m.emb.s), B(s.params), B(h), C(D), C(m.emb.e)),
		norm(0, 0, 0, m.layers[0].g1),
	}
	for l := range m.layers {
		ly := &m.layers[l]
		next := m.gf
		if l+1 < L {
			next = m.layers[l+1].g1
		}
		ds = append(ds,
			matvec(&ly.qkv, mb, 0, -24, acc),
			one("int_qkv", B(acc), F(ly.b), F(m.ropeC), F(m.ropeS), B(s.params), BO(kc, l*kvLayer), BO(vc, l*kvLayer),
				B(mq), B(shq), B(ex), metal.Arg{Bytes: qkvArgs}, C(ly.qkv.e), C(0), C(0)),
		)
		ds = append(ds,
			metal.Dispatch{P: g.ps["int_attn1"], Args: []metal.Arg{B(mq), B(shq), BO(kc, l*kvLayer), B(sb), B(cmax),
				B(s.params), metal.Arg{Bytes: attnArgs}},
				Groups: metal.Size{W: uint(nkv), H: 1, D: 1}, Threads: metal.Size{W: intACH, H: 1, D: 1}},
			metal.Dispatch{P: g.ps["int_attn2"], Args: []metal.Arg{B(sb), B(cmax), BO(vc, l*kvLayer), F(rawA), B(zpart),
				B(opart), B(s.params), metal.Arg{Bytes: attnArgs}},
				Groups: metal.Size{W: uint(nkv), H: 1, D: 1}, Threads: metal.Size{W: intACH, H: 1, D: 1}},
			metal.Dispatch{P: g.ps["int_attn3"], Args: []metal.Arg{B(zpart), B(opart), B(o), B(s.params),
				metal.Arg{Bytes: attnArgs}},
				Groups: metal.Size{W: uint(nq), H: 1, D: 1}, Threads: metal.Size{W: uint(hd), H: 1, D: 1}},
			one("int_blockf", B(o), B(mb), B(ex), C(nq*hd), C(-24), C(1), C(0, 0)),
			matvec(&ly.o, mb, 1, -24, acc),
			norm(1, ly.o.e, 1, ly.g2),
			matvec(&ly.gu, mb, 0, -16, acc),
			grid("int_mlp", I, B(acc), F(rawS), B(p), B(ex), C(I), C(ly.gu.e), C(0), C(0, 0)),
			one("int_blockf", B(p), B(mb), B(ex), C(I), C(-16), C(2), C(0, 0)),
			matvec(&ly.d, mb, 2, -24, acc),
			norm(1, ly.d.e, 2, next),
		)
	}
	ds = append(ds,
		matvec(&m.head, mb, 0, -10, lg),
		tbl("int_table1", B(lg), BO(tpart, 0), C(V)),
		tbl("int_table2", B(lg), F(rawF), BO(tpart, 0), BO(tpart, 8*tch), B(cum), C(V), C(intNF)),
		tbl("int_table3", BO(tpart, 8*tch), BO(tpart, 16*tch), B(cum), C(V)),
		tbl("int_table4", BO(tpart, 16*tch), B(cum), C(V)),
	)
	if err := s.constify(ds); err != nil {
		s.release()
		return nil, err
	}
	s.ds = ds
	return s, nil
}

func (s *intMetalState) newBuffer(n int) (*metal.Buffer, error) {
	b, err := s.g.dev.NewBuffer(max(n, 16))
	if err == nil {
		s.bufs = append(s.bufs, b)
	}
	return b, err
}

func cArg(vs ...int) metal.Arg { return metal.Arg{Bytes: i32b(vs...)} }

// farg is a kernel argument pointing at a slice of the model file.
func (g *intMetal) farg(x any) metal.Arg {
	switch v := x.(type) {
	case []int8:
		return g.file(unsafe.Pointer(unsafe.SliceData(v)))
	case []int16:
		return g.file(unsafe.Pointer(unsafe.SliceData(v)))
	case []int64:
		return g.file(unsafe.Pointer(unsafe.SliceData(v)))
	case []byte:
		return g.file(unsafe.Pointer(unsafe.SliceData(v)))
	}
	panic("intmetal: unknown file slice")
}

// constify moves the dispatches' constant arguments into one buffer (256-byte aligned), so every argument is a
// buffer and Run binds a dispatch's arguments in one call.
func (s *intMetalState) constify(ds []metal.Dispatch) error {
	var consts []byte
	type slot struct{ d, a, off int }
	var slots []slot
	for di := range ds {
		for ai, a := range ds[di].Args {
			if a.Buf == nil {
				slots = append(slots, slot{di, ai, len(consts)})
				consts = append(consts, a.Bytes...)
				consts = append(consts, make([]byte, (256-len(consts)%256)%256)...)
			}
		}
	}
	cb, err := s.newBuffer(len(consts))
	if err != nil {
		return err
	}
	copy(cb.Bytes(), consts)
	for _, sl := range slots {
		args := append([]metal.Arg(nil), ds[sl.d].Args...)
		args[sl.a] = metal.Arg{Buf: cb, Offset: sl.off}
		ds[sl.d].Args = args
	}
	return nil
}

func (s *intMetalState) step(tok, pos int) error {
	pb := s.params.Bytes()
	binary.LittleEndian.PutUint32(pb[0:], uint32(tok))
	binary.LittleEndian.PutUint32(pb[4:], uint32(pos))
	nch := uint((pos + intACH) / intACH)
	s1, s2 := s.g.ps["int_attn1"], s.g.ps["int_attn2"]
	for i := range s.ds { // the split-attention grids follow the position
		if p := s.ds[i].P; p == s1 || p == s2 {
			s.ds[i].Groups.H = nch
		}
	}
	return s.g.dev.Run(s.ds...)
}

func (s *intMetalState) release() {
	for _, b := range s.bufs {
		b.Release()
	}
	s.bufs = nil
}
