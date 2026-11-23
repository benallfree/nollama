package engine

import (
	"testing"
	"unsafe"
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
	B := func(i int) []byte { return gs.bufs[i].Bytes() }
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
	cmp32 := func(name string, got []int32, want []int16) {
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
	cmp32("norm m", i32s(B(2), D), st.xm)
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
	gs.ds = all[:5]
	run(5)
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
