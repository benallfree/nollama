//go:build darwin

package engine

// int_gemm_sg (gpu_int_sg.metal) against gpu_int.metal's int_gemm: the same outputs bit for bit on random
// worst-case inputs at the 0.5B's (and one 7B) shapes, and each kernel's time. NOLLAMA_SGGEMM=1.

import (
	"encoding/binary"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/benallfree/nollama/engine/internal/metal"
)

func TestSGGemm(t *testing.T) {
	if os.Getenv("NOLLAMA_SGGEMM") == "" {
		t.Skip("set NOLLAMA_SGGEMM=1")
	}
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	// gpu_int.metal's helpers (rsh) plus the prototype
	ps, err := d.Compile(metalIntSource, "int_gemm")
	if err != nil {
		t.Fatal(err)
	}
	sp, err := d.Compile("#define AQ "+strconv.Itoa(intActBits)+"\n"+metalIntSGSource, "int_gemm_sg")
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(1))
	shapes := []struct {
		name       string
		O, K, rows int
	}{
		{"qkv", 1152, 896, 512}, {"o", 896, 896, 512}, {"lm-ish 7B gate+up", 37888, 3584, 128}, {"gate+up", 9728, 896, 512}, {"down", 896, 4864, 512},
		{"head", 151936, 896, 64},
	}
	for _, sh := range shapes {
		O, K, rows := sh.O, sh.K, sh.rows
		ng := K / 64
		qb, _ := d.NewBuffer(O * K)
		sb, _ := d.NewBuffer(2 * O * ng)
		xb, _ := d.NewBuffer(2 * rows * K)
		exb, _ := d.NewBuffer(16 * rows)
		o1, _ := d.NewBuffer(8 * rows * O)
		o2, _ := d.NewBuffer(8 * rows * O)
		for i := range qb.Bytes() {
			qb.Bytes()[i] = byte(int8(r.Intn(255) - 127))
		}
		for i := 0; i < O*ng; i++ {
			binary.LittleEndian.PutUint16(sb.Bytes()[2*i:], uint16(int16(r.Intn(65535)-32767)))
		}
		for i := 0; i < rows*K; i++ {
			v := r.Intn(4095) - 2047
			if r.Intn(4) == 0 {
				v = []int{-2047, 2047}[r.Intn(2)]
			}
			binary.LittleEndian.PutUint16(xb.Bytes()[2*i:], uint16(int16(v)))
		}
		for i := 0; i < 4*rows; i++ {
			binary.LittleEndian.PutUint32(exb.Bytes()[4*i:], uint32(int32(r.Intn(5))))
		}
		args := func(out *metal.Buffer) []metal.Arg {
			return []metal.Arg{{Buf: qb}, {Buf: sb}, {Buf: xb}, {Buf: out}, {Buf: exb}, {Bytes: i32b(O)}, {Bytes: i32b(K)},
				{Bytes: i32b(0)}, {Bytes: i32b(1)}, {Bytes: i32b(24)}, {Bytes: i32b(0, K, O, 0)}}
		}
		ref := metal.Dispatch{P: ps["int_gemm"], Args: args(o1), Groups: metal.Size{W: uint((O + 63) / 64), H: uint(rows / 64), D: 1},
			Threads: metal.Size{W: 256, H: 1, D: 1}}
		sgd := metal.Dispatch{P: sp["int_gemm_sg"], Args: args(o2), Groups: metal.Size{W: uint((O + 63) / 64), H: uint(rows / 64), D: 1},
			Threads: metal.Size{W: 256, H: 1, D: 1}}
		if err := d.Run(ref); err != nil {
			t.Fatal(err)
		}
		if err := d.Run(sgd); err != nil {
			t.Fatal(err)
		}
		tr, ts := 1e9, 1e9
		for i := 0; i < 20; i++ { // alternate, best of 20 each (the GPU may be shared)
			d.Run(ref)
			tr = min(tr, d.LastGPU.Seconds())
			d.Run(sgd)
			ts = min(ts, d.LastGPU.Seconds())
		}
		bad := 0
		b1, b2 := o1.Bytes(), o2.Bytes()
		for i := 0; i < len(b1); i += 8 {
			if binary.LittleEndian.Uint64(b1[i:]) != binary.LittleEndian.Uint64(b2[i:]) {
				bad++
			}
		}
		macs := float64(O) * float64(K) * float64(rows)
		t.Logf("%-8s O=%6d K=%5d rows=%3d  int_gemm %7.2f ms (%.2f T)  int_gemm_sg %7.2f ms (%.2f T)  %.2fx  mismatches %d/%d",
			sh.name, O, K, rows, tr*1e3, macs/tr/1e12, ts*1e3, macs/ts/1e12, tr/ts, bad, rows*O)
		if bad > 0 {
			t.Errorf("%s: %d mismatches", sh.name, bad)
		}
		for _, b := range []*metal.Buffer{qb, sb, xb, exb, o1, o2} {
			b.Release()
		}
	}
}
