//go:build darwin

package engine

// The simdgroup-matrix attention kernels (gpu_int_sga.metal) against gpu_int.metal's: the same outputs bit for bit on
// random worst-case inputs at the 0.5B, 1.5B and 7B head shapes, full attention and a window, and each kernel's time.
// NOLLAMA_SGATTN=1.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/benallfree/nollama/engine/internal/metal"
)

type sgaShape struct {
	name               string
	nq, nkv, hd        int
	maxlen, base, rows int // R rows at positions base .. base+R-1
	R, win, sink       int
}

var sgaShapes = []sgaShape{
	{"0.5B 8k", 14, 2, 64, 8192, 8192 - 512, 512, 512, 0, 0},
	{"0.5B 8k R=500 win", 14, 2, 64, 8192, 8192 - 512, 512, 500, 2048, 16},
	{"1.5B 8k", 12, 2, 128, 8192, 8192 - 512, 512, 512, 0, 0},
	{"1.5B 8k R=333 win", 12, 2, 128, 8192, 8192 - 512, 512, 333, 4096, 16},
	{"7B 8k", 28, 4, 128, 8192, 8192 - 256, 256, 256, 0, 0},
}

func sgaCompile(t *testing.T, d *metal.Device, names ...string) map[string]*metal.Pipeline {
	ps, err := d.Compile("#define SGA_EXPERIMENTS\n"+metalIntSource+"\n"+metalIntSGASource, names...)
	if err != nil {
		t.Fatal(err)
	}
	return ps
}

// sgaInputs: queries (12-bit, a quarter at +-2047), their shifts, the key cache (16-bit mantissas, a quarter at
// +-32767, then shifts).
func sgaInputs(d *metal.Device, r *rand.Rand, sh sgaShape, sh0, shn int) (mq, shq, km *metal.Buffer) {
	mq, _ = d.NewBuffer(4 * sh.rows * sh.nq * sh.hd)
	shq, _ = d.NewBuffer(4 * sh.rows * sh.nq)
	km, _ = d.NewBuffer(intKLayer(sh.nkv, sh.maxlen, sh.hd))
	ext := func(lim int) int {
		if r.Intn(4) == 0 {
			return []int{-lim, lim}[r.Intn(2)]
		}
		return r.Intn(2*lim+1) - lim
	}
	for i := 0; i < sh.rows*sh.nq*sh.hd; i++ {
		binary.LittleEndian.PutUint32(mq.Bytes()[4*i:], uint32(int32(ext(2047))))
	}
	for i := 0; i < sh.rows*sh.nq; i++ {
		binary.LittleEndian.PutUint32(shq.Bytes()[4*i:], uint32(int32(sh0+r.Intn(shn))))
	}
	n := sh.nkv * sh.maxlen * sh.hd
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(km.Bytes()[2*i:], uint16(int16(ext(32767))))
	}
	for i := 0; i < sh.nkv*sh.maxlen; i++ {
		binary.LittleEndian.PutUint32(km.Bytes()[2*n+4*i:], uint32(int32(sh0+r.Intn(shn))))
	}
	return
}

func sgaArgs(sh sgaShape, ez int) []byte {
	nchmax := (sh.maxlen + intKC - 1) / intKC
	return i32b(sh.nq, sh.nkv, sh.hd, sh.maxlen, intNA, 0, nchmax, ez, sh.win, sh.sink, sh.rows)
}

func TestSGScores(t *testing.T) {
	if os.Getenv("NOLLAMA_SGATTN") == "" {
		t.Skip("set NOLLAMA_SGATTN=1")
	}
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	ps := sgaCompile(t, d, "int_bscores64", "int_bscores128", "int_bscores_sg64", "int_bscores_sg128")
	r := rand.New(rand.NewSource(1))
	for _, sh := range sgaShapes {
		mq, shq, km := sgaInputs(d, r, sh, 0, 9)
		nchmax := (sh.maxlen + intKC - 1) / intKC
		sN := 4 * sh.rows * sh.nq * ((sh.maxlen + 15) / 16 * 16)
		var outs [2][3]*metal.Buffer // S, cmax, ovf
		for k := range outs {
			outs[k][0], _ = d.NewBuffer(sN)
			outs[k][1], _ = d.NewBuffer(8 * sh.rows * sh.nq * nchmax)
			outs[k][2], _ = d.NewBuffer(4)
			for _, b := range outs[k] {
				for i := range b.Bytes() {
					b.Bytes()[i] = 0x5a
				}
			}
			binary.LittleEndian.PutUint32(outs[k][2].Bytes(), 0)
		}
		par, _ := d.NewBuffer(8)
		copy(par.Bytes(), i32b(sh.R, sh.base))
		grp := sh.nq / sh.nkv
		qr := 128
		if sh.hd == 128 {
			qr = 64
		}
		br := qr / grp
		nch := (sh.base+sh.R-1)/intKC + 1
		args := func(k int) []metal.Arg {
			return []metal.Arg{{Buf: mq}, {Buf: shq}, {Buf: km}, {Buf: outs[k][0]}, {Buf: outs[k][1]}, {Buf: outs[k][2]},
				{Buf: par}, {Bytes: sgaArgs(sh, 0)}}
		}
		g := metal.Size{W: uint(sh.nkv), H: uint((sh.R + br - 1) / br), D: uint(nch)}
		th := metal.Size{W: 256, H: 1, D: 1}
		suf := "64"
		if sh.hd == 128 {
			suf = "128"
		}
		ref := metal.Dispatch{P: ps["int_bscores"+suf], Args: args(0), Groups: g, Threads: th}
		sgd := metal.Dispatch{P: ps["int_bscores_sg"+suf], Args: args(1), Groups: g, Threads: th}
		if err := d.Run(ref); err != nil {
			t.Fatal(err)
		}
		if err := d.Run(sgd); err != nil {
			t.Fatal(err)
		}
		bad := 0
		names := []string{"S", "cmax", "ovf"}
		for i := 0; i < 3; i++ {
			if !bytes.Equal(outs[0][i].Bytes(), outs[1][i].Bytes()) {
				a, b := outs[0][i].Bytes(), outs[1][i].Bytes()
				n, first := 0, -1
				for x := 0; x < len(a); x += 4 {
					if binary.LittleEndian.Uint32(a[x:]) != binary.LittleEndian.Uint32(b[x:]) {
						if first < 0 {
							first = x / 4
						}
						n++
					}
				}
				t.Errorf("%s: %s differs in %d of %d words (first %d)", sh.name, names[i], n, len(a)/4, first)
				bad += n
			}
		}
		tr, ts := 1e9, 1e9
		for i := 0; i < 10; i++ { // alternate, best of 10 each (the GPU may be shared)
			d.Run(ref)
			tr = min(tr, d.LastGPU.Seconds())
			d.Run(sgd)
			ts = min(ts, d.LastGPU.Seconds())
		}
		t.Logf("%-20s scores: int_bscores %7.2f ms  sg %7.2f ms  %.2fx  ovf %d  mismatching words %d",
			sh.name, tr*1e3, ts*1e3, tr/ts, binary.LittleEndian.Uint32(outs[0][2].Bytes()), bad)
		for _, b := range []*metal.Buffer{mq, shq, km, par, outs[0][0], outs[0][1], outs[0][2], outs[1][0], outs[1][1], outs[1][2]} {
			b.Release()
		}
	}
}

// sgaExpa: an EXPA-like table (2^15 e^(-i/1024), rounded) and its nonzero length
func sgaExpa(d *metal.Device) (*metal.Buffer, int) {
	b, _ := d.NewBuffer(4 * intNA)
	ez := intNA
	for i := 0; i < intNA; i++ {
		v := int32(math.Round(32768 * math.Exp(-float64(i)/1024)))
		if v == 0 && ez == intNA {
			ez = i
		}
		binary.LittleEndian.PutUint32(b.Bytes()[4*i:], uint32(v))
	}
	return b, ez
}

// TestSGPV: int_bpv_sg* against int_bpv* on the scores int_bscores* makes from random inputs (score shifts chosen so
// the weights mix zeros, small and full ones), values v = vm << sv with |vm| up to 2047 and sv spread 0..11 per key
// (the three-digit path) or within 2 (two digits).
func TestSGPV(t *testing.T) {
	if os.Getenv("NOLLAMA_SGATTN") == "" {
		t.Skip("set NOLLAMA_SGATTN=1")
	}
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	variants := map[int][]string{64: {"int_bpv_sg64_nf", "int_bpv_sg64"}, 128: {"int_bpv_sg128_nf", "int_bpv_sg128"}}
	names := []string{"int_bscores64", "int_bscores128", "int_bpv64", "int_bpv128"}
	for _, v := range variants {
		names = append(names, v...)
	}
	ps := sgaCompile(t, d, names...)
	for _, k := range names[2:] {
		t.Logf("%s: max %d threads", k, ps[k].MaxThreads)
	}
	expa, ez := sgaExpa(d)
	r := rand.New(rand.NewSource(2))
	// value shifts: spread 0..11 (three digits), within 2 (two), or like real text (most tiles within 2, every
	// tenth tile within 4)
	for _, sv := range []struct{ s0, sn int }{{0, 12}, {4, 3}, {4, -1}} {
		for _, sh := range sgaShapes {
			mq, shq, km := sgaInputs(d, r, sh, 10, 5)
			vc, _ := d.NewBuffer(4 * sh.nkv * sh.maxlen * sh.hd)
			for k := 0; k < sh.nkv*sh.maxlen; k++ {
				sn := sv.sn
				if sn < 0 {
					sn = 3
					if (k%sh.maxlen/32)%10 == 0 {
						sn = 5
					}
				}
				s := sv.s0 + r.Intn(sn)
				for e := 0; e < sh.hd; e++ {
					v := r.Intn(4095) - 2047
					if r.Intn(4) == 0 {
						v = []int{-2047, 2047}[r.Intn(2)]
					}
					binary.LittleEndian.PutUint32(vc.Bytes()[4*(k*sh.hd+e):], uint32(int32(v<<s)))
				}
			}
			nchmax := (sh.maxlen + intKC - 1) / intKC
			sN := 4 * sh.rows * sh.nq * ((sh.maxlen + 15) / 16 * 16)
			sc, _ := d.NewBuffer(sN)
			cmax, _ := d.NewBuffer(8 * sh.rows * sh.nq * nchmax)
			ovf, _ := d.NewBuffer(4)
			par, _ := d.NewBuffer(8)
			copy(par.Bytes(), i32b(sh.R, sh.base))
			grp := sh.nq / sh.nkv
			qr, brv, suf := 128, 64, "64"
			if sh.hd == 128 {
				qr, brv, suf = 64, 32, "128"
			}
			nch := (sh.base+sh.R-1)/intKC + 1
			th := metal.Size{W: 256, H: 1, D: 1}
			ba := sgaArgs(sh, ez)
			if err := d.Run(metal.Dispatch{P: ps["int_bscores"+suf], Args: []metal.Arg{{Buf: mq}, {Buf: shq}, {Buf: km}, {Buf: sc},
				{Buf: cmax}, {Buf: ovf}, {Buf: par}, {Bytes: ba}},
				Groups: metal.Size{W: uint(sh.nkv), H: uint((sh.R + qr/grp - 1) / (qr / grp)), D: uint(nch)}, Threads: th}); err != nil {
				t.Fatal(err)
			}
			vs := variants[sh.hd]
			outs := make([][2]*metal.Buffer, 1+len(vs)) // zp, op: the reference, then each variant
			for k := range outs {
				outs[k][0], _ = d.NewBuffer(8 * sh.rows * sh.nq * nchmax)
				outs[k][1], _ = d.NewBuffer(8 * sh.rows * sh.nq * nchmax * sh.hd)
				for _, b := range outs[k] {
					for i := range b.Bytes() {
						b.Bytes()[i] = 0x5a
					}
				}
			}
			args := func(k int) []metal.Arg {
				return []metal.Arg{{Buf: sc}, {Buf: cmax}, {Buf: vc}, {Buf: expa}, {Buf: outs[k][0]}, {Buf: outs[k][1]},
					{Buf: par}, {Bytes: ba}}
			}
			ds := []metal.Dispatch{{P: ps["int_bpv"+suf], Args: args(0), Threads: th,
				Groups: metal.Size{W: uint(sh.nq), H: uint((sh.R + brv - 1) / brv), D: uint(nch)}}}
			for k, v := range vs {
				ds = append(ds, metal.Dispatch{P: ps[v], Args: args(k + 1), Threads: th,
					Groups: metal.Size{W: uint(sh.nq), H: uint((sh.R + sgaBR(v) - 1) / sgaBR(v)), D: uint(nch)}})
			}
			for _, x := range ds {
				if err := d.Run(x); err != nil {
					t.Fatal(err)
				}
			}
			best := make([]float64, len(ds))
			for i := range best {
				best[i] = 1e9
			}
			for rep := 0; rep < 10; rep++ {
				for i, x := range ds {
					d.Run(x)
					best[i] = min(best[i], d.LastGPU.Seconds())
				}
			}
			line := fmt.Sprintf("%-20s sv %d%+d pv: int_bpv %7.2f ms", sh.name, sv.s0, sv.sn, best[0]*1e3)
			for k, v := range vs {
				bad := 0
				for i, nm := range []string{"zp", "op"} {
					// rows < R only: rows past the batch's end are padding no later kernel reads (the 16-row variants
					// leave rows the reference's 32-row blocks zero unwritten)
					per := 8 * sh.nq * nchmax * []int{1, sh.hd}[i]
					a, b := outs[0][i].Bytes()[:sh.R*per], outs[k+1][i].Bytes()[:sh.R*per]
					n, first := 0, -1
					for x := 0; x < len(a); x += 8 {
						if binary.LittleEndian.Uint64(a[x:]) != binary.LittleEndian.Uint64(b[x:]) {
							if first < 0 {
								first = x / 8
							}
							n++
						}
					}
					if n > 0 {
						t.Errorf("%s %s sv %d%+d: %s differs in %d of %d (first %d)", v, sh.name, sv.s0, sv.sn, nm, n, len(a)/8, first)
					}
					bad += n
				}
				line += fmt.Sprintf(" | %s %7.2f ms %.2fx bad %d", v, best[k+1]*1e3, best[0]/best[k+1], bad)
			}
			t.Log(line)
			for _, o := range outs {
				o[0].Release()
				o[1].Release()
			}
			for _, b := range []*metal.Buffer{mq, shq, km, vc, sc, cmax, ovf, par} {
				b.Release()
			}
		}
	}
	expa.Release()
}

// TestSGPVAbl: where int_bpv_sg*'s time goes: each kernel with one part switched off (PVABL; outputs then wrong).
func TestSGPVAbl(t *testing.T) {
	if os.Getenv("NOLLAMA_SGATTN") == "" {
		t.Skip("set NOLLAMA_SGATTN=1")
	}
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	var ps [4]map[string]*metal.Pipeline
	for i := range ps {
		ps[i], err = d.Compile(fmt.Sprintf("#define PVABL %d\n", i)+metalIntSource+"\n"+metalIntSGASource, "int_bscores64",
			"int_bscores128", "int_bpv_sg64", "int_bpv_sg128")
		if err != nil {
			t.Fatal(err)
		}
	}
	expa, ez := sgaExpa(d)
	defer expa.Release()
	r := rand.New(rand.NewSource(3))
	for _, sh := range []sgaShape{sgaShapes[0], sgaShapes[2]} {
		mq, shq, km := sgaInputs(d, r, sh, 10, 5)
		vc, _ := d.NewBuffer(4 * sh.nkv * sh.maxlen * sh.hd)
		for k := 0; k < sh.nkv*sh.maxlen*sh.hd; k++ {
			binary.LittleEndian.PutUint32(vc.Bytes()[4*k:], uint32(int32((r.Intn(4095)-2047)<<(4+(k/sh.hd)%3))))
		}
		nchmax := (sh.maxlen + intKC - 1) / intKC
		sc, _ := d.NewBuffer(4 * sh.rows * sh.nq * ((sh.maxlen + 15) / 16 * 16))
		cmax, _ := d.NewBuffer(8 * sh.rows * sh.nq * nchmax)
		ovf, _ := d.NewBuffer(4)
		par, _ := d.NewBuffer(8)
		zp, _ := d.NewBuffer(8 * sh.rows * sh.nq * nchmax)
		op, _ := d.NewBuffer(8 * sh.rows * sh.nq * nchmax * sh.hd)
		copy(par.Bytes(), i32b(sh.R, sh.base))
		grp, qr, suf := sh.nq/sh.nkv, 128, "64"
		if sh.hd == 128 {
			qr, suf = 64, "128"
		}
		nch := (sh.base+sh.R-1)/intKC + 1
		th := metal.Size{W: 256, H: 1, D: 1}
		ba := sgaArgs(sh, ez)
		if err := d.Run(metal.Dispatch{P: ps[0]["int_bscores"+map[int]string{64: "64", 128: "128"}[sh.hd]], Args: []metal.Arg{{Buf: mq}, {Buf: shq}, {Buf: km}, {Buf: sc},
			{Buf: cmax}, {Buf: ovf}, {Buf: par}, {Bytes: ba}},
			Groups: metal.Size{W: uint(sh.nkv), H: uint((sh.R + qr/grp - 1) / (qr / grp)), D: uint(nch)}, Threads: th}); err != nil {
			t.Fatal(err)
		}
		best := [4]float64{1e9, 1e9, 1e9, 1e9}
		for rep := 0; rep < 10; rep++ {
			for i := range ps {
				d.Run(metal.Dispatch{P: ps[i]["int_bpv_sg"+suf], Args: []metal.Arg{{Buf: sc}, {Buf: cmax}, {Buf: vc}, {Buf: expa},
					{Buf: zp}, {Buf: op}, {Buf: par}, {Bytes: ba}}, Threads: th,
					Groups: metal.Size{W: uint(sh.nq), H: uint((sh.R + sgaBR("int_bpv_sg"+suf) - 1) / sgaBR("int_bpv_sg"+suf)), D: uint(nch)}})
				best[i] = min(best[i], d.LastGPU.Seconds()*1e3)
			}
		}
		t.Logf("%-10s int_bpv_sg%s: full %.2f ms, no MMA %.2f, no score loads %.2f, no value loads %.2f", sh.name, suf,
			best[0], best[1], best[2], best[3])
		for _, b := range []*metal.Buffer{mq, shq, km, vc, sc, cmax, ovf, par, zp, op} {
			b.Release()
		}
	}
}

// sgaBR: rows per block of an int_bpv_sg* variant
func sgaBR(k string) int {
	if k == "int_bpv_sg128" {
		return 16
	}
	return 32
}

// TestSGScoresAbl: where int_bscores_sg*'s time goes (SCABL; outputs then wrong).
func TestSGScoresAbl(t *testing.T) {
	if os.Getenv("NOLLAMA_SGATTN") == "" {
		t.Skip("set NOLLAMA_SGATTN=1")
	}
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	var ps [3]map[string]*metal.Pipeline
	for i := range ps {
		ps[i], err = d.Compile(fmt.Sprintf("#define SCABL %d\n", i)+metalIntSource+"\n"+metalIntSGASource, "int_bscores64",
			"int_bscores128", "int_bscores_sg64", "int_bscores_sg128")
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("max threads: sg64 %d, sg128 %d", ps[0]["int_bscores_sg64"].MaxThreads, ps[0]["int_bscores_sg128"].MaxThreads)
	r := rand.New(rand.NewSource(4))
	for _, sh := range []sgaShape{sgaShapes[0], sgaShapes[2]} {
		mq, shq, km := sgaInputs(d, r, sh, 0, 9)
		nchmax := (sh.maxlen + intKC - 1) / intKC
		sc, _ := d.NewBuffer(4 * sh.rows * sh.nq * ((sh.maxlen + 15) / 16 * 16))
		cmax, _ := d.NewBuffer(8 * sh.rows * sh.nq * nchmax)
		ovf, _ := d.NewBuffer(4)
		par, _ := d.NewBuffer(8)
		copy(par.Bytes(), i32b(sh.R, sh.base))
		grp, qr, suf := sh.nq/sh.nkv, 128, "64"
		if sh.hd == 128 {
			qr, suf = 64, "128"
		}
		nch := (sh.base+sh.R-1)/intKC + 1
		g := metal.Size{W: uint(sh.nkv), H: uint((sh.R + qr/grp - 1) / (qr / grp)), D: uint(nch)}
		args := []metal.Arg{{Buf: mq}, {Buf: shq}, {Buf: km}, {Buf: sc}, {Buf: cmax}, {Buf: ovf}, {Buf: par}, {Bytes: sgaArgs(sh, 0)}}
		th := metal.Size{W: 256, H: 1, D: 1}
		best := [4]float64{1e9, 1e9, 1e9, 1e9}
		for rep := 0; rep < 10; rep++ {
			d.Run(metal.Dispatch{P: ps[0]["int_bscores"+suf], Args: args, Groups: g, Threads: th})
			best[3] = min(best[3], d.LastGPU.Seconds()*1e3)
			for i := range ps {
				d.Run(metal.Dispatch{P: ps[i]["int_bscores_sg"+suf], Args: args, Groups: g, Threads: th})
				best[i] = min(best[i], d.LastGPU.Seconds()*1e3)
			}
		}
		t.Logf("%-10s old %.2f ms | sg full %.2f, no MMA %.2f, no score stores %.2f", sh.name, best[3], best[0], best[1], best[2])
		for _, b := range []*metal.Buffer{mq, shq, km, sc, cmax, ovf, par} {
			b.Release()
		}
	}
}
