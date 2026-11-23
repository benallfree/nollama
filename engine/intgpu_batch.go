package engine

// Batches on a GPU: when every token is known in advance (compression), a session runs intBatchRows positions at a
// time as matrix-matrix work, and returns just each position's counts for the token that follows. Same kernels'
// integer arithmetic row by row, so the same counts as stepping one token at a time.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"
)

const (
	intBatchRows = 512  // positions per batch at most (a multiple of intHeadRows)
	intHeadRows  = 64   // rows per head + counts pass (bounds the logits buffer: rows x vocabulary)
	intGemmRows  = 32   // GBT in gpu_int.metal
	intKC        = 1024 // KC: keys per attention chunk
	intScoreMem  = 1 << 30
)

// grid dimensions that follow the batch's row count
const (
	rowsW    = iota // Groups.W = rows (a threadgroup per row)
	rowsH           // Groups.H = rows (a grid row per row)
	gemmH           // Groups.H = ceil(rows / intGemmRows)
	attnH           // Groups.H = ceil(rows / attention block) (the fallback attention)
	scoreG          // int_bscores*: H = ceil(rows / rows per block), D = chunks
	pvG             // int_bpv*: H = ceil(rows / BRV), D = chunks
	battnG          // int_battn*: W = ceil(rows / 8)
	gemm2H          // int_gemm_mma2: H = ceil(rows / 64)
	fixedDim        // no change
)

// intBatchSlow counts batches that needed the fallback attention (for tests and profiling).
var (
	intBatchSlow int
	intBatchGPU  time.Duration // GPU time of the batches run so far
)

var errUnsupportedHD = errors.New("intmodel: batches on the GPU need a head size of 64 or 128")

type intBatch struct {
	P                           int // rows per batch
	params, toks, tgt, cnt, ovf ibuf
	sc                          ibuf    // scores (without planes); with them, 64 bytes (int_bpv2_mma*_cnt's counters)
	ds, slow                    []idisp // slow: the same with int_attnb* (when a score overflows int32)
	kind, slowKind              []int
	head                        [][2]idisp // per head block: the head GEMM and the counts
	attnBR, scoreBR, pvBR       int
}

func (s *intGPUState) newBatch() (*intBatch, error) {
	g, m := s.g, s.m
	D, I, V, hd, nq, nkv := m.D, m.I, m.V, m.hd, m.nq, m.nkv
	// rows per batch: the stored scores (rows x heads x context, int32) stay within intScoreMem
	P := min(intBatchRows, max(intHeadRows, intScoreMem/(4*nq*s.maxlen))/intHeadRows*intHeadRows)
	scoreN := 4 * P * nq * s.maxlen
	nchmax := (s.maxlen + intKC - 1) / intKC
	partN := P * nq * nchmax
	oneKernel := s.kp != nil && os.Getenv("NOLLAMA_INT_BATTN") != "" // experiment: int_battn* (slower so far)
	if s.kp != nil {
		P, scoreN = intBatchRows, 64 // tensor-core attention recomputes scores instead of storing them
		if v, err := strconv.Atoi(os.Getenv("NOLLAMA_INT_BATCH")); err == nil && v >= intHeadRows {
			P = v / intHeadRows * intHeadRows
		}
		partN = P * nq * nchmax
		if oneKernel {
			partN = 2
		}
	}
	qkvN := (nq + 2*nkv) * hd
	accN := max(qkvN, D, 2*I)
	mN := max(D, I, nq*hd)
	b := &intBatch{P: P, attnBR: 32, scoreBR: 128 / (nq / nkv), pvBR: 64}
	attnK, scoreK, pvK := "int_attnb64", "int_bscores64", "int_bpv64"
	if hd == 128 {
		attnK, scoreK, pvK = "int_attnb128", "int_bscores128", "int_bpv128"
		b.attnBR, b.scoreBR, b.pvBR = 16, 64/(nq/nkv), 32
	} else if hd != 64 {
		return nil, errUnsupportedHD
	}
	var h, mb, ex, acc, mq, shq, o, p, lg, sc, cmax, zp, op ibuf
	var err error
	for _, x := range []struct {
		dst *ibuf
		n   int
	}{{&b.params, 8}, {&b.toks, 4 * P}, {&b.tgt, 4 * P}, {&b.cnt, 8 * 3 * P}, {&h, 8 * P * D}, {&mb, 2 * P * mN},
		{&ex, 16 * P}, {&acc, 8 * P * accN}, {&mq, 4 * P * nq * hd}, {&shq, 4 * P * nq}, {&o, 8 * P * nq * hd},
		{&p, 8 * P * I}, {&lg, 8 * intHeadRows * V}, {&b.ovf, 4}, {&sc, scoreN},
		{&cmax, 8 * partN}, {&zp, 8 * partN}, {&op, 8 * partN * hd}} {
		if *x.dst, err = s.newBuffer(x.n); err != nil {
			return nil, err
		}
	}
	b.sc = sc
	B := func(b ibuf) iarg { return iarg{Buf: b} }
	BO := func(b ibuf, off int) iarg { return iarg{Buf: b, Offset: off} }
	C, F := cArg, g.farg
	var ds, slowD []idisp
	var slowAt []int // where each layer's fast attention (three dispatches) starts in ds
	add := func(kind int, d idisp) {
		ds = append(ds, d)
		b.kind = append(b.kind, kind)
	}
	perRow := func(k string, threads int, args ...iarg) {
		add(rowsW, idisp{K: k, Args: args, Groups: isize{W: uint(P), H: 1, D: 1},
			Threads: isize{W: uint(threads), H: 1, D: 1}})
	}
	gemmD := func(w *intMat, x iarg, exA iarg, einSlot, to int, out iarg, outS, rows int) idisp {
		d := idisp{K: "int_gemm", Args: []iarg{F(w.q), F(w.s), x, out, exA, C(w.out), C(w.in),
			C(w.e), C(einSlot), C(to), C(0, mN, outS, 0)},
			Groups:  isize{W: uint((w.out + 63) / 64), H: uint((rows + intGemmRows - 1) / intGemmRows), D: 1},
			Threads: isize{W: 256, H: 1, D: 1}}
		if k := gemm128(s); k != "" { // 128 x 64 tiles
			d.K, d.Groups.W, d.Groups.H = k, uint((w.out+127)/128), uint((rows+63)/64)
		}
		return d
	}
	gemm := func(w *intMat, einSlot, to int, out ibuf, outS int) {
		k := gemmH
		if gemm128(s) != "" {
			k = gemm2H
		}
		add(k, gemmD(w, B(mb), B(ex), einSlot, to, B(out), outS, P))
	}
	norm := func(addv, E, einSlot int, gw []int64) {
		perRow("int_norm", 256, B(h), B(acc), F(gw), B(mb), B(ex), C(D), C(E), C(einSlot), C(0), C(addv), C(accN, mN))
	}
	qkvArgs := make([]byte, 24)
	binary.LittleEndian.PutUint32(qkvArgs[0:], uint32(nq))
	binary.LittleEndian.PutUint32(qkvArgs[4:], uint32(nkv))
	binary.LittleEndian.PutUint32(qkvArgs[8:], uint32(hd))
	binary.LittleEndian.PutUint32(qkvArgs[12:], uint32(s.maxlen))
	binary.LittleEndian.PutUint64(qkvArgs[16:], uint64(m.sc))
	attnArgs := i32b(nq, nkv, hd, s.maxlen, intNA, 0)
	ez := m.expaLen()
	if ez > 11392 {
		return nil, errors.New("intmodel: EXPA's nonzero part is longer than the batch kernels hold")
	}
	battnArgs := i32b(nq, nkv, hd, s.maxlen, intNA, s.maxlen, nchmax, ez)
	rawA, rawF, rawS := m.g.Tensors["EXPA"].Data, m.g.Tensors["EXPF"].Data, m.g.Tensors["SIG"].Data

	add(rowsH, idisp{K: "int_embed", Args: []iarg{F(m.emb.q), F(m.emb.s), B(b.toks), B(h), C(D),
		C(m.emb.e)}, Groups: isize{W: uint((D + 255) / 256), H: uint(P), D: 1}, Threads: isize{W: 256, H: 1, D: 1}})
	norm(0, 0, 0, m.layers[0].g1)
	for l := range m.layers {
		ly := &m.layers[l]
		next := m.gf
		if l+1 < m.L {
			next = m.layers[l+1].g1
		}
		kv := l * s.kvLayer
		gemm(&ly.qkv, 0, -24, acc, accN)
		perRow("int_qkv", intTG, B(acc), F(ly.b), F(m.ropeC), F(m.ropeS), B(b.params), BO(s.kc, kv), BO(s.vc, kv),
			B(mq), B(shq), B(ex), iarg{Bytes: qkvArgs}, C(ly.qkv.e), C(0), C(accN))
		if oneKernel { // the plane cache and the one-kernel tensor-core attention (gpu_int_mma.cu)
			add(rowsW, s.kvplanes(l, b.params, P))
			add(battnG, idisp{K: fmt.Sprintf("int_battn%d", hd), Args: []iarg{B(mq), B(shq), B(o), F(rawA), B(b.params),
				iarg{Bytes: battnArgs}, BO(s.kp, l*s.kpl), BO(s.vp, l*s.kpl), C(s.mp)},
				Groups: isize{W: uint((P + 7) / 8), H: uint(nkv), D: 1}, Threads: isize{W: 256, H: 1, D: 1}})
			perRow("int_blockf", intTG, B(o), B(mb), B(ex), C(nq*hd), C(-24), C(1), C(nq*hd, mN))
			gemm(&ly.o, 1, -24, acc, accN)
			norm(1, ly.o.e, 1, ly.g2)
			gemm(&ly.gu, 0, -16, acc, accN)
			add(rowsH, idisp{K: "int_mlp", Args: []iarg{B(acc), F(rawS), B(p), B(ex), C(I), C(ly.gu.e),
				C(0), C(accN, I)}, Groups: isize{W: uint((I + 255) / 256), H: uint(P), D: 1},
				Threads: isize{W: 256, H: 1, D: 1}})
			perRow("int_blockf", intTG, B(p), B(mb), B(ex), C(I), C(-16), C(2), C(I, mN))
			gemm(&ly.d, 2, -24, acc, accN)
			norm(1, ly.d.e, 2, next)
			continue
		}
		var planeArgs []iarg // the plane cache, for the tensor-core attention
		if s.kp != nil {
			add(rowsW, s.kvplanes(l, b.params, P))
			planeArgs = []iarg{BO(s.kp, l*s.kpl), BO(s.vp, l*s.kpl), C(s.mp)}
		}
		slowAt = append(slowAt, len(ds))
		slowD = append(slowD, idisp{K: attnK, Args: []iarg{B(mq), B(shq), BO(s.kc, kv), BO(s.vc, kv),
			F(rawA), B(o), B(b.params), iarg{Bytes: attnArgs}},
			Groups: isize{W: uint(nq), H: uint(P / b.attnBR), D: 1}, Threads: isize{W: 256, H: 1, D: 1}})
		add(scoreG, idisp{K: scoreK, Args: append([]iarg{B(mq), B(shq), BO(s.kc, kv), B(sc), B(cmax),
			B(b.ovf), B(b.params), iarg{Bytes: battnArgs}}, planeArgs...),
			Groups: isize{W: uint(nkv), H: 1, D: 1}, Threads: isize{W: 256, H: 1, D: 1}})
		pvArgs := []iarg{B(sc), B(cmax), BO(s.vc, kv), F(rawA), B(zp), B(op), B(b.params), iarg{Bytes: battnArgs}}
		if planeArgs != nil {
			pvArgs = append(append(pvArgs, planeArgs...), B(mq), B(shq))
		}
		add(pvG, idisp{K: pvK, Args: pvArgs,
			Groups: isize{W: uint(nq), H: 1, D: 1}, Threads: isize{W: 256, H: 1, D: 1}})
		add(rowsH, idisp{K: "int_bcomb", Args: []iarg{B(zp), B(op), B(o), B(b.params),
			iarg{Bytes: battnArgs}},
			Groups: isize{W: uint(nq), H: uint(P), D: 1}, Threads: isize{W: uint(hd), H: 1, D: 1}})
		perRow("int_blockf", intTG, B(o), B(mb), B(ex), C(nq*hd), C(-24), C(1), C(nq*hd, mN))
		gemm(&ly.o, 1, -24, acc, accN)
		norm(1, ly.o.e, 1, ly.g2)
		gemm(&ly.gu, 0, -16, acc, accN)
		add(rowsH, idisp{K: "int_mlp", Args: []iarg{B(acc), F(rawS), B(p), B(ex), C(I), C(ly.gu.e),
			C(0), C(accN, I)}, Groups: isize{W: uint((I + 255) / 256), H: uint(P), D: 1},
			Threads: isize{W: 256, H: 1, D: 1}})
		perRow("int_blockf", intTG, B(p), B(mb), B(ex), C(I), C(-16), C(2), C(I, mN))
		gemm(&ly.d, 2, -24, acc, accN)
		norm(1, ly.d.e, 2, next)
	}
	// head and counts, intHeadRows rows at a time
	var hds []idisp
	for r0 := 0; r0 < P; r0 += intHeadRows {
		hds = append(hds,
			gemmD(&m.head, BO(mb, 2*r0*mN), BO(ex, 16*r0), 0, -10, B(lg), V, intHeadRows),
			idisp{K: "int_counts", Args: []iarg{B(lg), F(rawF), BO(b.tgt, 4*r0), BO(b.cnt, 24*r0), C(V),
				C(intNF)}, Groups: isize{W: intHeadRows, H: 1, D: 1}, Threads: isize{W: intTG, H: 1, D: 1}})
	}
	all := append(append(ds, hds...), slowD...)
	if err := s.g.dev.prepare(all, s.newBuffer); err != nil {
		return nil, err
	}
	b.ds = all[:len(ds)]
	slowD = all[len(ds)+len(hds):]
	// the slow list: each layer's three attention dispatches replaced by one int_attnb*
	for i, l := 0, 0; i < len(b.ds); i++ {
		if l < len(slowAt) && i == slowAt[l] {
			b.slow, b.slowKind = append(b.slow, slowD[l]), append(b.slowKind, attnH)
			i += 2
			l++
			continue
		}
		b.slow, b.slowKind = append(b.slow, b.ds[i]), append(b.slowKind, b.kind[i])
	}
	for i := 0; i < len(hds); i += 2 {
		b.head = append(b.head, [2]idisp{all[len(ds)+i], all[len(ds)+i+1]})
	}
	return b, nil
}

// feedBatch runs toks at positions pos, pos+1, ... and stores, for each, the counts of next[i] in out.
func (s *intGPUState) feedBatch(toks, next []int, pos int, out []IntCount) error {
	if s.b == nil {
		b, err := s.newBatch()
		if err != nil {
			return err
		}
		s.b = b
	}
	b := s.b
	P := b.P
	dev := s.g.dev
	if s.kp != nil { // steps (Feed) don't keep the plane cache: copy the positions they added
		s.kpTo = min(s.kpTo, pos)
		for s.kpTo < pos {
			R := min(P, pos-s.kpTo)
			dev.write(b.params, 0, i32b(R, s.kpTo))
			cu := make([]idisp, s.m.L)
			for l := range cu {
				cu[l] = s.kvplanes(l, b.params, R)
			}
			if err := dev.run(cu); err != nil {
				return err
			}
			s.kpTo += R
		}
		s.kpTo = pos + len(toks)
	}
	tkb, tgb, cntb, ovf := make([]byte, 4*P), make([]byte, 4*P), make([]byte, 24*P), make([]byte, 4)
	ds := make([]idisp, 0, len(b.ds)+2*len(b.head))
	for r0 := 0; r0 < len(toks); r0 += P {
		R := min(P, len(toks)-r0)
		for i := 0; i < R; i++ {
			binary.LittleEndian.PutUint32(tkb[4*i:], uint32(toks[r0+i]))
			binary.LittleEndian.PutUint32(tgb[4*i:], uint32(next[r0+i]))
		}
		dev.write(b.toks, 0, tkb[:4*R])
		dev.write(b.tgt, 0, tgb[:4*R])
		dev.write(b.params, 0, i32b(R, pos+r0))
		nch := uint((pos+r0+R-1)/intKC + 1)
		if err := s.runBatch(b.ds, b.kind, R, nch, ds); err != nil {
			return err
		}
		intBatchGPU += dev.lastGPU()
		if dev.read(b.ovf, 0, ovf); binary.LittleEndian.Uint32(ovf) != 0 {
			// a score didn't fit int32: this batch again with the attention that keeps them in registers
			dev.write(b.ovf, 0, make([]byte, 4))
			intBatchSlow++
			if err := s.runBatch(b.slow, b.slowKind, R, nch, ds); err != nil {
				return err
			}
		}
		dev.read(b.cnt, 0, cntb[:24*R])
		for i := 0; i < R; i++ {
			c := func(j int) int64 { return int64(binary.LittleEndian.Uint64(cntb[8*(3*i+j):])) }
			out[r0+i] = IntCount{Total: c(0), High: c(1), Low: c(2)}
		}
	}
	return nil
}

func (s *intGPUState) runBatch(list []idisp, kind []int, R int, nch uint, ds []idisp) error {
	b := s.b
	ds = append(ds[:0], list...)
	for i := range ds {
		switch kind[i] {
		case rowsW:
			ds[i].Groups.W = uint(R)
		case rowsH:
			ds[i].Groups.H = uint(R)
		case gemmH:
			ds[i].Groups.H = uint((R + intGemmRows - 1) / intGemmRows)
		case attnH:
			ds[i].Groups.H = uint((R + b.attnBR - 1) / b.attnBR)
		case scoreG:
			ds[i].Groups.H, ds[i].Groups.D = uint((R+b.scoreBR-1)/b.scoreBR), nch
		case pvG:
			ds[i].Groups.H, ds[i].Groups.D = uint((R+b.pvBR-1)/b.pvBR), nch
		case battnG:
			ds[i].Groups.W = uint((R + 7) / 8)
		case gemm2H:
			ds[i].Groups.H = uint((R + 63) / 64)
		}
	}
	for hb := 0; hb*intHeadRows < R; hb++ {
		n := min(intHeadRows, R-hb*intHeadRows)
		gm, ct := b.head[hb][0], b.head[hb][1]
		if gm.K != "int_gemm" {
			gm.Groups.H = uint((n + 63) / 64)
		} else {
			gm.Groups.H = uint((n + intGemmRows - 1) / intGemmRows)
		}
		ct.Groups.W = uint(n)
		ds = append(ds, gm, ct)
	}
	return s.g.dev.run(ds)
}

// gemm128 names the 128-output x 64-row GEMM kernel batches use on tensor cores: int_gemm_mma3 (pipelined), or the
// unpipelined int_gemm_mma2 with NOLLAMA_INT_GEMM2; "" (int_gemm's 64 x 32 tiles) without planes or with
// NOLLAMA_INT_GEMM1.
func gemm128(s *intGPUState) string {
	switch {
	case s.kp == nil || os.Getenv("NOLLAMA_INT_GEMM1") != "":
		return ""
	case os.Getenv("NOLLAMA_INT_GEMM2") != "":
		return "int_gemm_mma2"
	}
	return "int_gemm_mma3"
}
