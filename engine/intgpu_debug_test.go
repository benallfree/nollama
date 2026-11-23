package engine

import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
)

func i64s(b []byte, n int) []int64 { return unsafe.Slice((*int64)(unsafe.Pointer(&b[0])), n) }
func i32s(b []byte, n int) []int32 { return unsafe.Slice((*int32)(unsafe.Pointer(&b[0])), n) }

// TestIntGPUStages compares the GPU's buffers with the CPU's for token 0, kernel by kernel, through layer 0.
func TestIntGPUStages(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	run := func(n int) {
		if err := gs.step(m.BOS(), 0); err != nil {
			t.Fatal(err)
		}
		_ = n
	}
	all := gs.ds
	B := func(i int) []byte { return gs.g.dev.host(gs.bufs[i]) }
	D, I := m.D, m.I
	// CPU reference, stage by stage
	st := m.newState(8)
	q, sc := m.emb.q[m.bos*D:(m.bos+1)*D], m.emb.s[m.bos*(D/64):(m.bos+1)*(D/64)]
	h := make([]int64, D)
	for k := 0; k < D; k++ {
		h[k] = irsh(int64(q[k])*int64(sc[k/64]), -24-m.emb.e)
	}
	cmp64 := func(name string, got []int64, want []int64) {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: first difference at %d: gpu %d, cpu %d", name, i, got[i], want[i])
			}
		}
		t.Logf("%s: %d values match", name, len(want))
	}
	cmp16 := func(name string, got []int16, want []int16) {
		for i := range want {
			if int64(got[i]) != int64(want[i]) {
				t.Fatalf("%s: first difference at %d: gpu %d, cpu %d", name, i, got[i], want[i])
			}
		}
		t.Logf("%s: %d values match", name, len(want))
	}
	gs.ds = all[:1]
	run(1)
	cmp64("embed h", i64s(B(1), D), h)
	gs.ds = all[:2]
	run(2)
	copy(st.h, h)
	e := m.rmsnorm(st.h, m.layers[0].g1, st.xm, st.y)
	cmp16("norm m", unsafe.Slice((*int16)(unsafe.Pointer(&B(2)[0])), D), st.xm)
	if g := i32s(B(3), 4)[0]; int(g) != e {
		t.Fatalf("norm exponent: gpu %d, cpu %d", g, e)
	}
	gs.ds = all[:3]
	run(3)
	ly := &m.layers[0]
	acc := make([]int64, ly.qkv.out)
	m.matvec(&ly.qkv, st.xm, e, -24, acc)
	cmp64("qkv matvec", i64s(B(4), ly.qkv.out), acc)
	// qkv epilogue: K, V at position 0, q mantissas and shifts
	nq, nkv, hd := m.nq, m.nkv, m.hd
	for i := range acc {
		acc[i] += ly.b[i]
	}
	st.pos = 0
	qv := acc[:nq*hd]
	kv := acc[nq*hd : (nq+nkv)*hd]
	vv := acc[(nq+nkv)*hd:]
	st.rope(qv, nq)
	st.rope(kv, nkv)
	gs.ds = all[:4]
	run(4)
	gk, gv := i32s(B(7), nkv*8*hd), i32s(B(8), nkv*8*hd)
	for j := 0; j < nkv; j++ {
		for d := 0; d < hd; d++ {
			wk := int32(min(max(irsh(kv[j*hd+d], 12), -intLimKV), intLimKV))
			wv := int32(min(max(irsh(vv[j*hd+d], 12), -intLimKV), intLimKV))
			if gk[j*8*hd+d] != wk || gv[j*8*hd+d] != wv {
				t.Fatalf("kv head %d d %d: gpu k %d v %d, cpu k %d v %d", j, d, gk[j*8*hd+d], gv[j*8*hd+d], wk, wv)
			}
		}
	}
	t.Logf("k, v: match")
	gmq, gsh := i32s(B(5), nq*hd), i32s(B(6), nq)
	for i := 0; i < nq; i++ {
		var mx int64
		for d := 0; d < hd; d++ {
			mx = max(mx, abs64(qv[i*hd+d]*m.sc))
		}
		sh := max(0, bitlen(mx)-15)
		if int(gsh[i]) != sh {
			t.Fatalf("q head %d shift: gpu %d, cpu %d", i, gsh[i], sh)
		}
		for d := 0; d < hd; d++ {
			w := min(max(irsh(qv[i*hd+d]*m.sc, sh), -32767), 32767)
			if int64(gmq[i*hd+d]) != w {
				t.Fatalf("q head %d d %d: gpu %d, cpu %d", i, d, gmq[i*hd+d], w)
			}
		}
	}
	t.Logf("q mantissas: match")
	// attention at position 0: one key, weight EXPA[0]; o = V * 2^12 exactly
	gs.ds = all[:7] // three split-attention passes
	run(7)
	go_ := i64s(B(10), nq*hd)
	grp := nq / nkv
	for i := 0; i < nq; i++ {
		j := i / grp
		for d := 0; d < hd; d++ {
			want := int64(min(max(irsh(vv[j*hd+d], 12), -intLimKV), intLimKV)) << 12
			if go_[i*hd+d] != want {
				t.Fatalf("attn head %d d %d: gpu %d, cpu %d", i, d, go_[i*hd+d], want)
			}
		}
	}
	t.Logf("attention: match")
	_ = I
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

// TestIntGPUProfile times one token's dispatch list, whole and without each kernel type, to see where time goes.
func TestIntGPUProfile(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 32768)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	all := gs.ds
	at := 500
	if v, err := strconv.Atoi(os.Getenv("INTPOS")); err == nil {
		at = v
	}
	tm := func(ds []idisp) float64 {
		gs.ds = ds
		gs.step(m.BOS(), at)
		t0 := time.Now()
		for i := 0; i < 30; i++ {
			gs.step(m.BOS(), at)
		}
		return float64(time.Since(t0).Microseconds()) / 30 / 1000
	}
	full := tm(all)
	t.Logf("full step at position %d: %.2f ms (%d dispatches), GPU %.2f ms", at, full, len(all),
		float64(m.gpu.dev.lastGPU().Microseconds())/1000)
	for _, pos := range []int{100, 1000, 4000, 16000, 30000} {
		if pos < gs.maxlen {
			gs.ds = all
			gs.step(m.BOS(), pos)
			t0 := time.Now()
			gs.step(m.BOS(), pos)
			t.Logf("position %5d: %.2f ms, GPU %.2f ms", pos, float64(time.Since(t0).Microseconds())/1000,
				float64(m.gpu.dev.lastGPU().Microseconds())/1000)
		}
	}
	for _, k := range intKernels {
		var rest []idisp
		n := 0
		for _, d := range all {
			if d.K == k {
				n++
				continue
			}
			rest = append(rest, d)
		}
		if n == 0 {
			continue
		}
		t.Logf("without %-12s (%3d dispatches): %.2f ms  -> about %.2f ms", k, n, tm(rest), full-tm(rest))
	}
}

// TestIntGPUFloor: the cost of a dependent dispatch, and each matvec's bandwidth.
func TestIntGPUFloor(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	all := gs.ds
	tm := func(ds []idisp) float64 {
		gs.ds = ds
		gs.step(m.BOS(), 10)
		best := 1e9
		for i := 0; i < 10; i++ {
			gs.step(m.BOS(), 10)
			best = min(best, float64(m.gpu.dev.lastGPU().Microseconds()))
		}
		return best
	}
	byName := map[string][]idisp{}
	for _, d := range all {
		byName[d.K] = append(byName[d.K], d)
	}
	rep := func(d idisp, n int) []idisp {
		r := make([]idisp, n)
		for i := range r {
			r[i] = d
		}
		return r
	}
	for _, k := range []string{"int_blockf", "int_norm", "int_qkv", "int_mlp", "int_table1", "int_table2", "int_table3", "int_table4", "int_attn1", "int_attn2", "int_attn3"} {
		d := byName[k][0]
		t.Logf("%-12s x200: %.1f us each", k, tm(rep(d, 200))/200)
	}
	mv := byName["int_matvec"]
	ly := &m.layers[0]
	for i, w := range []*intMat{&ly.qkv, &ly.o, &ly.gu, &ly.d} {
		us := tm(rep(mv[i], 100)) / 100
		t.Logf("matvec %dx%d: %.1f us each, %.0f GB/s", w.out, w.in, us, float64(w.out*w.in)/us/1e3)
	}
	us := tm(rep(mv[len(mv)-1], 20)) / 20
	t.Logf("head %dx%d: %.1f us, %.0f GB/s", m.head.out, m.head.in, us, float64(m.head.out*m.head.in)/us/1e3)
}

// TestIntGPUEncode: how long encoding a token's dispatches takes, against the GPU's own time.
func TestIntGPUEncode(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	for _, ch := range []int{48, 100000} {
		old := metal.SetRunChunk(ch)
		gs.step(m.BOS(), 10)
		var wall, gpu float64
		for i := 0; i < 20; i++ {
			t0 := time.Now()
			gs.step(m.BOS(), 10)
			wall += float64(time.Since(t0).Microseconds()) / 20
			gpu += float64(m.gpu.dev.lastGPU().Microseconds()) / 20
		}
		metal.SetRunChunk(old)
		t.Logf("chunk %6d: wall %.0f us, gpu %.0f us (%d dispatches)", ch, wall, gpu, len(gs.ds))
	}
}

// TestIntGPUAttnLong times the attention passes at long positions.
func TestIntGPUAttnLong(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 32768)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	all := gs.ds
	for _, pos := range []int{1000, 8000, 16000, 30000} {
		for _, k := range []string{"int_attn1", "int_attn2", "int_attn3"} {
			var d idisp
			for _, x := range all {
				if x.K == k {
					d = x
					break
				}
			}
			r := make([]idisp, 50)
			for i := range r {
				r[i] = d
			}
			gs.ds = r
			gs.step(m.BOS(), pos)
			best := 1e9
			for i := 0; i < 5; i++ {
				gs.step(m.BOS(), pos)
				best = min(best, float64(m.gpu.dev.lastGPU().Microseconds())/50)
			}
			t.Logf("pos %5d %s: %.1f us", pos, k, best)
		}
	}
	gs.ds = all
}

// TestIntBatchProfile: GPU time of one full batch at several positions, and per kernel.
func TestIntBatchProfile(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 32768)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	P := 64
	toks, next := make([]int, P), make([]int, P)
	out := make([]IntCount, P)
	if err := gs.feedBatch(toks, next, 0, out); err != nil {
		t.Fatal(err)
	}
	b := gs.b
	P = b.P
	binary.LittleEndian.PutUint32(gs.g.dev.host(b.params)[0:], uint32(P))
	all := append([]idisp(nil), b.ds...)
	for _, h := range b.head {
		all = append(all, h[0], h[1])
	}
	run := func(ds []idisp) float64 {
		gs.g.dev.run(ds)
		best := 1e9
		for i := 0; i < 3; i++ {
			gs.g.dev.run(ds)
			best = min(best, float64(m.gpu.dev.lastGPU().Microseconds())/1000)
		}
		return best
	}
	for _, pos := range []int{16000} {
		binary.LittleEndian.PutUint32(gs.g.dev.host(b.params)[4:], uint32(pos))
		nch := uint((pos+P-1)/intKC + 1)
		for i := range all {
			if i < len(b.kind) && (b.kind[i] == scoreG || b.kind[i] == pvG) {
				all[i].Groups.D = nch
				br := map[int]int{scoreG: b.scoreBR, pvG: b.pvBR}[b.kind[i]]
				all[i].Groups.H = uint((P + br - 1) / br)
			}
		}
		full := run(all)
		line := ""
		for _, k := range []string{"int_gemm", "int_bscores64", "int_bpv64", "int_bcomb", "int_counts", "int_norm", "int_qkv", "int_blockf", "int_mlp", "int_embed"} {
			var sub []idisp
			for _, d := range all {
				if d.K == k {
					sub = append(sub, d)
				}
			}
			line += fmt.Sprintf(" %s %.1f", k, run(sub))
		}
		t.Logf("pos %5d: batch of %d %.1f ms (%.0f tok/s) |%s", pos, P, full, float64(P)/full*1000, line)
	}
}

// TestIntBatchBisect runs the first INTBISECT launches of a 0.5B batch (a GPU fault ends the process, so a script
// runs this once per prefix): which kernel faults.
func TestIntBatchBisect(t *testing.T) {
	n, err := strconv.Atoi(os.Getenv("INTBISECT"))
	if err != nil {
		t.Skip("set INTBISECT=<launches>")
	}
	m := loadIntDev(t, "0.5B", "auto")
	gs, err := m.gpu.newState(m, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	b, err := gs.newBatch()
	if err != nil {
		t.Fatal(err)
	}
	gs.b = b
	all := append([]idisp(nil), b.ds...)
	for _, h := range b.head {
		all = append(all, h[0], h[1])
	}
	gs.g.dev.write(b.params, 0, i32b(b.P, 0))
	for i := range all {
		switch b.kind[min(i, len(b.kind)-1)] {
		case scoreG, pvG:
			if i < len(b.kind) {
				all[i].Groups.D = 1
				br := map[int]int{scoreG: b.scoreBR, pvG: b.pvBR}[b.kind[i]]
				all[i].Groups.H = uint((b.P + br - 1) / br)
			}
		}
	}
	n = min(n, len(all))
	err = gs.g.dev.run(all[:n])
	t.Logf("%d launches of %d, last %s %+v x %+v: %v", n, len(all), all[n-1].K, all[n-1].Groups, all[n-1].Threads, err)
	if err != nil {
		t.Fail()
	}
}

// TestIntBatchSteps feeds the slice 512 tokens at a time, logging each, to find a batch that faults.
func TestIntBatchSteps(t *testing.T) {
	m := loadIntDev(t, "0.5B", "auto")
	ids := vecIDs(t, "slice.ids.json")
	toks := append([]int{m.BOS()}, ids[:len(ids)-1]...)
	a, err := m.NewSession(len(ids))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	step, _ := strconv.Atoi(os.Getenv("INTSTEP"))
	if step == 0 {
		step = 512
	}
	for k := 0; k < len(ids); k += step {
		e := min(k+step, len(ids))
		_, err := a.FeedBatch(toks[k:e], ids[k:e])
		t.Logf("positions %d..%d: %v", k, e, err)
		if err != nil {
			t.FailNow()
		}
	}
}
