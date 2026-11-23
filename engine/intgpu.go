package engine

// Integer models on a GPU (Metal: gpu_int.metal, CUDA: gpu_int.cu): a session's KV cache and activations live in GPU
// buffers, the model file is on the device (wrapped zero-copy on unified memory, else uploaded once), and one token
// is one prepared list of kernel launches. Integer arithmetic throughout, so the distributions are the CPU's, bit for
// bit, on every GPU. The launch lists are built here, once, for every backend (intDev).

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"time"
	"unsafe"
)

var errBadTarget = errors.New("intmodel: target count past the table's total")

const intTG = 1024 // threads per threadgroup int_qkv, int_blockf and int_counts assume

// isize, iarg and idisp describe kernel launches independently of the GPU API.
type isize struct{ W, H, D uint }

// iarg is a kernel argument: a buffer at an offset, or a value's bytes (an int, a few ints, a small struct).
type iarg struct {
	Buf    ibuf
	Offset int
	Bytes  []byte
}

// idisp is one kernel launch: Groups threadgroups (blocks) of Threads threads.
type idisp struct {
	K               string
	Args            []iarg
	Groups, Threads isize
	pre             any // the backend's own form of Args, from prepare
}

type ibuf interface{ size() int }

// intDev is a GPU backend for integer models.
type intDev interface {
	name() string
	close()
	alloc(n int) (ibuf, error) // zeroed
	free(b ibuf)
	write(dst ibuf, off int, src []byte)
	read(src ibuf, off int, dst []byte)
	host(b ibuf) []byte // the buffer's memory as the CPU sees it (unified memory), or nil
	// prepare readies a launch list once (Metal moves constants into a buffer); alloc's buffers live with the list.
	prepare(ds []idisp, alloc func(int) (ibuf, error)) error
	run(ds []idisp) error // in order, each launch seeing the ones before; returns when they're done
	// replay runs a list the same way, possibly recorded once (CUDA graphs) under key: then the list, grids
	// included, must not change between calls. graphs says whether it records; forget drops an owner's recordings.
	replay(key replayKey, ds []idisp) error
	graphs() bool
	forget(owner any)
	// planes says whether attention reads a byte-plane copy of the KV cache (CUDA's tensor-core kernels): then
	// int_kvplanes keeps it, and the attention launches get it as extra arguments.
	planes() bool
	lastGPU() time.Duration // the last run's GPU time, when the backend knows it
}

// replayKey names a recorded launch list: its owner (forget drops an owner's recordings) and the list itself.
type replayKey struct {
	owner any
	first *idisp
	n     int
	v     int // a variant of the list (the attention chunk count it covers)
}

// intGPU is a model's GPU: the device and the model file on it.
type intGPU struct {
	normTG int // threads for a step's norms
	dev    intDev
	file   ibuf
	base   uintptr // the file's first byte in host memory
}

func openIntGPU(m *IntModel) (*intGPU, error) {
	dev, file, err := openIntMetal(m)
	if err != nil {
		var err2 error
		if dev, file, err2 = openIntCUDA(m); err2 != nil {
			return nil, errors.Join(err, err2)
		}
	}
	g := &intGPU{dev: dev, file: file, base: uintptr(unsafe.Pointer(unsafe.SliceData(m.g.data))), normTG: 256}
	return g, nil
}

func (g *intGPU) close() {
	g.dev.free(g.file)
	g.dev.close()
}

func (g *intGPU) name() string { return g.dev.name() }

// intGPUState is one session on the GPU.
type intGPUState struct {
	g      *intGPU
	maxlen int
	params ibuf    // tok, pos
	cum    []int64 // the table, as the CPU sees it
	cumBuf ibuf    // where it is on the GPU, when the CPU can't see GPU memory
	fresh  bool    // cum holds the last step's table (cumBuf's copy has been read)
	tgt    ibuf    // int_find's target and result
	res    ibuf
	find   []idisp
	maxch  int
	kp, vp ibuf // the byte-plane KV cache, when planes()
	kpTo   int  // positions the planes hold
	mp     int  // positions per plane
	kpl    int  // bytes per layer
	ds     []idisp
	bufs   []ibuf
	// kept for batches (intgpu_batch.go)
	m       *IntModel
	kc, vc  ibuf
	kvLayer int
	b       *intBatch
}

const (
	intACH = 256  // positions per attention chunk (ACH in gpu_int.metal)
	intTCH = 2048 // table values per threadgroup (TCH)
)

func (g *intGPU) newState(m *IntModel, maxlen int) (*intGPUState, error) {
	s := &intGPUState{g: g, maxlen: maxlen, m: m}
	nb := s.newBuffer
	D, I, V, hd, nq, nkv, L := m.D, m.I, m.V, m.hd, m.nq, m.nkv, m.L
	qkvN := (nq + 2*nkv) * hd
	accN := max(qkvN, D, 2*I, V)
	var err error
	maxch := (maxlen + intACH - 1) / intACH
	s.maxch = maxch
	tch := (V + intTCH - 1) / intTCH
	var h, mb, ex, acc, mq, shq, kc, vc, sb, o, p, lg, cum, cmax, zpart, opart, tpart ibuf
	for _, x := range []struct {
		dst *ibuf
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
	if hb := g.dev.host(cum); hb != nil {
		s.cum = unsafe.Slice((*int64)(unsafe.Pointer(unsafe.SliceData(hb))), V)
	} else {
		s.cum, s.cumBuf = make([]int64, V), cum // read back when asked for
	}
	if s.tgt, err = nb(8); err == nil {
		s.res, err = nb(32)
	}
	if err != nil {
		s.release()
		return nil, err
	}
	s.find = []idisp{{K: "int_find", Args: []iarg{{Buf: cum}, {Buf: s.tgt}, {Buf: s.res}, cArg(V)},
		Groups: isize{1, 1, 1}, Threads: isize{32, 1, 1}}}
	if err := g.dev.prepare(s.find, s.newBuffer); err != nil {
		s.release()
		return nil, err
	}
	B := func(b ibuf) iarg { return iarg{Buf: b} }
	BO := func(b ibuf, off int) iarg { return iarg{Buf: b, Offset: off} }
	C, F := cArg, g.farg
	one := func(k string, args ...iarg) idisp {
		return idisp{K: k, Args: args, Groups: isize{W: 1, H: 1, D: 1}, Threads: isize{W: intTG, H: 1, D: 1}}
	}
	grid := func(k string, n int, args ...iarg) idisp {
		return idisp{K: k, Args: args, Groups: isize{W: uint((n + 255) / 256), H: 1, D: 1},
			Threads: isize{W: 256, H: 1, D: 1}}
	}
	tbl := func(k string, args ...iarg) idisp {
		return idisp{K: k, Args: args, Groups: isize{W: uint(tch), H: 1, D: 1},
			Threads: isize{W: 256, H: 1, D: 1}}
	}
	matvec := func(w *intMat, x ibuf, einSlot, to int, out ibuf) idisp {
		d := idisp{K: "int_matvec", Args: []iarg{F(w.q), F(w.s), B(x), B(out), B(ex),
			C(w.out), C(w.in), C(w.e), C(einSlot), C(to)},
			Groups: isize{W: uint((w.out + 7) / 8), H: 1, D: 1}, Threads: isize{W: 256, H: 1, D: 1}}
		if s.kp != nil && w.in <= 1024 && w.in%128 == 0 { // CUDA: short rows get 8 lanes each (int_matvec8, gpu_int_mma.cu)
			d.K, d.Groups.W = "int_matvec8", uint((w.out+31)/32)
		}
		return d
	}
	norm := func(add int, E, einSlot int, gw []int64) idisp {
		d := one("int_norm", B(h), B(acc), F(gw), B(mb), B(ex), C(D), C(E), C(einSlot), C(0), C(add), C(0, 0))
		d.Threads.W = uint(g.normTG)
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
	if g.dev.planes() {
		// a step's attention reads whole chunks; batches read up to a block's rows plus a tile past the last position
		s.mp = (maxlen+intACH-1)/intACH*intACH + 128
		s.kpl = nkv * 3 * s.mp * hd
		if s.kp, err = nb(L * s.kpl); err == nil {
			s.vp, err = nb(L * s.kpl)
		}
		if err != nil {
			s.release()
			return nil, err
		}
	}
	rawA, rawF, rawS := m.g.Tensors["EXPA"].Data, m.g.Tensors["EXPF"].Data, m.g.Tensors["SIG"].Data
	ez := m.expaLen()

	ds := []idisp{
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
				B(mq), B(shq), B(ex), iarg{Bytes: qkvArgs}, C(ly.qkv.e), C(0), C(0)),
		)
		var planeArgs1, planeArgs2 []iarg // the tensor-core attention reads the planes (kept per step too)
		a1, a2 := "int_attn1", "int_attn2"
		if s.kp != nil {
			ds = append(ds, s.kvplanes(l, s.params, 1))
			a1, a2 = fmt.Sprintf("int_dattn1_%d", hd), fmt.Sprintf("int_dattn2_%d", hd)
			planeArgs1 = []iarg{BO(s.kp, l*s.kpl), C(s.mp)}
			planeArgs2 = []iarg{BO(s.vp, l*s.kpl), C(s.mp), C(ez)}
		}
		ds = append(ds,
			idisp{K: a1, Args: append([]iarg{B(mq), B(shq), BO(kc, l*kvLayer), B(sb), B(cmax),
				B(s.params), iarg{Bytes: attnArgs}}, planeArgs1...),
				Groups: isize{W: uint(nkv), H: uint(maxch), D: 1}, Threads: isize{W: intACH, H: 1, D: 1}},
			idisp{K: a2, Args: append([]iarg{B(sb), B(cmax), BO(vc, l*kvLayer), F(rawA), B(zpart),
				B(opart), B(s.params), iarg{Bytes: attnArgs}}, planeArgs2...),
				Groups: isize{W: uint(nkv), H: uint(maxch), D: 1}, Threads: isize{W: intACH, H: 1, D: 1}},
			idisp{K: "int_attn3", Args: []iarg{B(zpart), B(opart), B(o), B(s.params),
				iarg{Bytes: attnArgs}},
				Groups: isize{W: uint(nq), H: 1, D: 1}, Threads: isize{W: uint(hd), H: 1, D: 1}},
			one("int_blockf", B(o), B(mb), B(ex), C(nq*hd), C(-24), C(1), C(0, 0)),
			matvec(&ly.o, mb, 1, -24, acc),
			norm(1, ly.o.e, 1, ly.g2),
			matvec(&ly.gu, mb, 0, -16, acc),
			one("int_mlpb", B(acc), F(rawS), B(p), B(mb), B(ex), C(I)),
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
	if s.kp != nil { // CUDA with tensor cores: the reference's fast step (gpu_int_step.cu) instead
		cs, err := s.cudaSteps(stepBufs{h: h, mb: mb, ex: ex, mq: mq, shq: shq, kc: kc, vc: vc, sb: sb, cmax: cmax, o: o, p: p,
			lg: lg, cum: cum, tpart: tpart}, qkvArgs, attnArgs, ez, tch)
		if err != nil {
			s.release()
			return nil, err
		}
		ds = cs
	}
	if err := g.dev.prepare(ds, s.newBuffer); err != nil {
		s.release()
		return nil, err
	}
	s.ds = ds
	return s, nil
}

type stepBufs struct{ h, mb, ex, mq, shq, kc, vc, sb, cmax, o, p, lg, cum, tpart ibuf }

// cudaSteps is a step as the reference's fast path runs it (gpu_int_step.cu): matvecs split along K with atomic
// partial sums, applied and zeroed by their consumers; attention into per-head totals, merged once. Same integers.
func (s *intGPUState) cudaSteps(b stepBufs, qkvArgs, attnArgs []byte, ez, tch int) ([]idisp, error) {
	m, g := s.m, s.g
	D, I, V, hd, nq, nkv, L := m.D, m.I, m.V, m.hd, m.nq, m.nkv, m.L
	qkvN := (nq + 2*nkv) * hd
	var accQ, accO, accGU, accD, oacc, zacc, pmax ibuf
	var err error
	for _, x := range []struct {
		dst *ibuf
		n   int
	}{{&accQ, 8 * qkvN}, {&accO, 8 * D}, {&accGU, 16 * I}, {&accD, 8 * D}, {&oacc, 8 * nq * hd}, {&zacc, 8 * nq},
		{&pmax, 8}} {
		if *x.dst, err = s.newBuffer(x.n); err != nil {
			return nil, err
		}
	}
	B := func(x ibuf) iarg { return iarg{Buf: x} }
	BO := func(x ibuf, off int) iarg { return iarg{Buf: x, Offset: off} }
	C, F := cArg, g.farg
	null := iarg{}
	blk := func(k string, threads int, args ...iarg) idisp {
		return idisp{K: k, Args: args, Groups: isize{1, 1, 1}, Threads: isize{uint(threads), 1, 1}}
	}
	mv := func(w *intMat, x iarg, pro bool, out ibuf) idisp {
		pa, pm := null, null
		if pro {
			pa, pm = B(b.p), B(pmax)
		}
		return idisp{K: "int_mv2", Args: []iarg{F(w.q), F(w.s), x, pa, pm, B(out), C(w.out), C(w.in)},
			Groups: isize{uint((w.out + 63) / 64), uint(w.in / 128), 1}, Threads: isize{256, 1, 1}}
	}
	norm := func(acc iarg, gw []int64, E, einSlot int, pm iarg, add int) idisp {
		k, nt := "int_norm2", 1024
		if D <= 1024 { // small models: fewer threads, shorter reductions
			k, nt = "int_norm2s", 256
		}
		return blk(k, nt, B(b.h), acc, F(gw), B(b.mb), B(b.ex), C(D), C(E), C(einSlot), pm, C(add))
	}
	maxch := s.maxch
	rawA, rawF, rawS := m.g.Tensors["EXPA"].Data, m.g.Tensors["EXPF"].Data, m.g.Tensors["SIG"].Data
	ds := []idisp{
		{K: "int_embed", Args: []iarg{F(m.emb.q), F(m.emb.s), B(s.params), B(b.h), C(D), C(m.emb.e)},
			Groups: isize{uint((D + 255) / 256), 1, 1}, Threads: isize{256, 1, 1}},
		norm(null, m.layers[0].g1, 0, 0, null, 0),
	}
	for l := range m.layers {
		ly := &m.layers[l]
		next := m.gf
		if l+1 < L {
			next = m.layers[l+1].g1
		}
		kv, kpo := l*s.kvLayer, l*s.kpl
		ds = append(ds,
			mv(&ly.qkv, B(b.mb), false, accQ),
			blk("int_qkv2", 256, B(accQ), F(ly.b), F(m.ropeC), F(m.ropeS), B(s.params), BO(b.kc, kv), BO(b.vc, kv),
				BO(s.kp, kpo), BO(s.vp, kpo), C(s.mp), B(b.mq), B(b.shq), B(b.ex), iarg{Bytes: qkvArgs}, C(ly.qkv.e)),
			idisp{K: fmt.Sprintf("int_dattn1_%d", hd), Args: []iarg{B(b.mq), B(b.shq), BO(b.kc, kv), B(b.sb), B(b.cmax),
				B(s.params), iarg{Bytes: attnArgs}, BO(s.kp, kpo), C(s.mp)},
				Groups: isize{uint(nkv), uint(maxch), 1}, Threads: isize{256, 1, 1}},
			idisp{K: fmt.Sprintf("int_dattn2a_%d", hd), Args: []iarg{B(b.sb), B(b.cmax), BO(b.vc, kv), F(rawA), B(zacc),
				B(oacc), B(s.params), iarg{Bytes: attnArgs}, BO(s.vp, kpo), C(s.mp), C(ez)},
				Groups: isize{uint(nkv), uint(maxch), 1}, Threads: isize{256, 1, 1}},
			blk("int_mergea", 1024, B(oacc), B(zacc), B(b.o), B(b.mb), B(b.ex), C(nq), C(hd)),
			mv(&ly.o, B(b.mb), false, accO),
			norm(B(accO), ly.g2, ly.o.e, 1, null, 1),
			mv(&ly.gu, B(b.mb), false, accGU),
			idisp{K: "int_mlp2", Args: []iarg{B(accGU), F(rawS), B(b.p), B(pmax), B(b.ex), C(I), C(ly.gu.e)},
				Groups: isize{uint((I + 255) / 256), 1, 1}, Threads: isize{256, 1, 1}},
			mv(&ly.d, null, true, accD),
			norm(B(accD), next, ly.d.e, 0, B(pmax), 1),
		)
	}
	tbl := func(k string, args ...iarg) idisp {
		return idisp{K: k, Args: args, Groups: isize{uint(tch), 1, 1}, Threads: isize{256, 1, 1}}
	}
	ds = append(ds,
		idisp{K: "int_mvlog", Args: []iarg{F(m.head.q), F(m.head.s), B(b.mb), B(b.lg), B(b.ex), C(V), C(D), C(m.head.e)},
			Groups: isize{uint((V + 63) / 64), 1, 1}, Threads: isize{256, 1, 1}},
		tbl("int_table1", B(b.lg), BO(b.tpart, 0), C(V)),
		tbl("int_table2", B(b.lg), F(rawF), BO(b.tpart, 0), BO(b.tpart, 8*tch), B(b.cum), C(V), C(intNF)),
		tbl("int_table3", BO(b.tpart, 8*tch), BO(b.tpart, 16*tch), B(b.cum), C(V)),
		tbl("int_table4", BO(b.tpart, 16*tch), B(b.cum), C(V)),
	)
	return ds, nil
}

// kvplanes is the launch that copies rows' new keys and values (positions params.pos + row) into the planes.
func (s *intGPUState) kvplanes(l int, params ibuf, rows int) idisp {
	m := s.m
	return idisp{K: "int_kvplanes", Args: []iarg{{Buf: s.kc, Offset: l * s.kvLayer}, {Buf: s.vc, Offset: l * s.kvLayer},
		{Buf: s.kp, Offset: l * s.kpl}, {Buf: s.vp, Offset: l * s.kpl}, {Buf: params}, cArg(m.nkv), cArg(m.hd),
		cArg(s.maxlen), cArg(s.mp)}, Groups: isize{uint(rows), 1, 1}, Threads: isize{256, 1, 1}}
}

// expaLen is where EXPA (nonincreasing) becomes 0 for good: weights from there on are 0.
func (m *IntModel) expaLen() int {
	ez := intNA
	for ez > 0 && m.expA[ez-1] == 0 {
		ez--
	}
	return ez
}

// attnChunked says whether a step's kernel runs a threadgroup per attention chunk (its grid follows the position).
func attnChunked(k string) bool {
	return k == "int_attn1" || k == "int_attn2" || strings.HasPrefix(k, "int_dattn")
}

func (s *intGPUState) newBuffer(n int) (ibuf, error) {
	b, err := s.g.dev.alloc(max(n, 16))
	if err == nil {
		s.bufs = append(s.bufs, b)
	}
	return b, err
}

func cArg(vs ...int) iarg { return iarg{Bytes: i32b(vs...)} }

func i32b(vs ...int) []byte {
	b := make([]byte, 4*len(vs))
	for i, v := range vs {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(v)))
	}
	return b
}

// farg is a kernel argument pointing at a slice of the model file.
func (g *intGPU) farg(x any) iarg {
	var p unsafe.Pointer
	switch v := x.(type) {
	case []int8:
		p = unsafe.Pointer(unsafe.SliceData(v))
	case []int16:
		p = unsafe.Pointer(unsafe.SliceData(v))
	case []int64:
		p = unsafe.Pointer(unsafe.SliceData(v))
	case []byte:
		p = unsafe.Pointer(unsafe.SliceData(v))
	default:
		panic("intgpu: unknown file slice")
	}
	return iarg{Buf: g.file, Offset: int(uintptr(p) - g.base)}
}

func (s *intGPUState) step(tok, pos int) error {
	s.g.dev.write(s.params, 0, i32b(tok, pos))
	s.fresh = false
	if s.kp != nil && s.kpTo == pos {
		s.kpTo = pos + 1 // the step keeps the planes
	}
	nch := uint((pos + intACH) / intACH)
	if s.g.dev.graphs() {
		// a recording per power-of-two chunk count (chunks past the position return at once): few recordings, and
		// at most half the attention blocks idle
		b := uint(1)
		for b < nch {
			b *= 2
		}
		b = min(b, uint(s.maxch))
		for i := range s.ds {
			if attnChunked(s.ds[i].K) {
				s.ds[i].Groups.H = b
			}
		}
		return s.g.dev.replay(replayKey{s, &s.ds[0], len(s.ds), int(b)}, s.ds)
	}
	for i := range s.ds { // the split-attention grids follow the position
		if attnChunked(s.ds[i].K) {
			s.ds[i].Groups.H = nch
		}
	}
	return s.g.dev.run(s.ds)
}

// table returns the last step's table, copying it back if the CPU can't see it.
func (s *intGPUState) table() []int64 {
	if s.cumBuf != nil && !s.fresh {
		s.g.dev.read(s.cumBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s.cum))), 8*len(s.cum)))
	}
	s.fresh = true
	return s.cum
}

// find returns the first t with cum[t] > target(total) and t's counts, searching on the GPU when the table is there.
func (s *intGPUState) findTok(target func(total int64) int64) (int, IntCount, error) {
	if s.cumBuf == nil || s.fresh {
		return findIn(s.table(), target)
	}
	V := len(s.cum)
	b := make([]byte, 32)
	s.g.dev.read(s.cumBuf, 8*(V-1), b[:8])
	binary.LittleEndian.PutUint64(b, uint64(target(int64(binary.LittleEndian.Uint64(b)))))
	s.g.dev.write(s.tgt, 0, b[:8])
	if err := s.g.dev.run(s.find); err != nil {
		return 0, IntCount{}, err
	}
	s.g.dev.read(s.res, 0, b)
	r := func(i int) int64 { return int64(binary.LittleEndian.Uint64(b[8*i:])) }
	if r(0) >= int64(V) {
		return 0, IntCount{}, errBadTarget
	}
	return int(r(0)), IntCount{Total: r(3), High: r(1), Low: r(2)}, nil
}

func (s *intGPUState) release() {
	s.g.dev.forget(s)
	for _, b := range s.bufs {
		s.g.dev.free(b)
	}
	s.bufs = nil
}
