//go:build darwin

package engine

// gpu_int.metal's int_matvec8 against its int_matvec: same outputs on random worst-case inputs, and time per
// call measured as 64 calls in one command buffer (in-stream, as in a decode step). NOLLAMA_DECMV=1.

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/benallfree/nollama/engine/internal/metal"
)

func TestDecMV(t *testing.T) {
	if os.Getenv("NOLLAMA_DECMV") == "" {
		t.Skip("set NOLLAMA_DECMV=1")
	}
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	ps, err := d.Compile(metalIntSource, "int_matvec", "int_matvec8")
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(1))
	for _, sh := range []struct {
		name string
		O, K int
	}{{"qkv", 1152, 896}, {"o", 896, 896}, {"gate+up", 9728, 896}, {"down", 896, 4864}, {"head", 151936, 896}} {
		O, K := sh.O, sh.K
		ng := K / 64
		qb, _ := d.NewBuffer(O * K)
		sb, _ := d.NewBuffer(2 * O * ng)
		xb, _ := d.NewBuffer(2 * K)
		exb, _ := d.NewBuffer(16)
		o1, _ := d.NewBuffer(8 * O)
		o2, _ := d.NewBuffer(8 * O)
		for i := range qb.Bytes() {
			qb.Bytes()[i] = byte(int8(r.Intn(255) - 127))
		}
		for i := 0; i < O*ng; i++ {
			binary.LittleEndian.PutUint16(sb.Bytes()[2*i:], uint16(int16(r.Intn(65535)-32767)))
		}
		for i := 0; i < K; i++ {
			binary.LittleEndian.PutUint16(xb.Bytes()[2*i:], uint16(int16(r.Intn(4095)-2047)))
		}
		args := func(out *metal.Buffer) []metal.Arg {
			return []metal.Arg{{Buf: qb}, {Buf: sb}, {Buf: xb}, {Buf: out}, {Buf: exb}, {Bytes: i32b(O)}, {Bytes: i32b(K)},
				{Bytes: i32b(0)}, {Bytes: i32b(1)}, {Bytes: i32b(24)}}
		}
		ref := metal.Dispatch{P: ps["int_matvec"], Args: args(o1), Groups: metal.Size{W: uint((O + 7) / 8), H: 1, D: 1},
			Threads: metal.Size{W: 256, H: 1, D: 1}}
		v8 := func(th int) metal.Dispatch {
			return metal.Dispatch{P: ps["int_matvec8"], Args: args(o2), Groups: metal.Size{W: uint((O + th/8 - 1) / (th / 8)), H: 1, D: 1},
				Threads: metal.Size{W: uint(th), H: 1, D: 1}}
		}
		rep := func(x metal.Dispatch) []metal.Dispatch {
			ds := make([]metal.Dispatch, 64)
			for i := range ds {
				ds[i] = x
			}
			return ds
		}
		tm := func(ds []metal.Dispatch) float64 {
			b := 1e9
			for i := 0; i < 8; i++ {
				if err := d.Run(ds...); err != nil {
					t.Fatal(err)
				}
				b = min(b, d.LastGPU.Seconds()/float64(len(ds)))
			}
			return b
		}
		tr := tm(rep(ref))
		line := ""
		for _, th := range []int{32, 64, 128, 256} {
			for i := range o2.Bytes() {
				o2.Bytes()[i] = 0
			}
			t8 := tm(rep(v8(th)))
			bad := 0
			for i := 0; i < 8*O; i += 8 {
				if binary.LittleEndian.Uint64(o1.Bytes()[i:]) != binary.LittleEndian.Uint64(o2.Bytes()[i:]) {
					bad++
				}
			}
			if bad > 0 {
				t.Errorf("%s tg %d: %d mismatches", sh.name, th, bad)
			}
			line += fmt.Sprintf("  tg%-3d %6.1f us %.2fx", th, t8*1e6, tr/t8)
		}
		gb := float64(O*K+2*O*ng) / 1e9
		t.Logf("%-8s %6d x %4d  int_matvec %6.1f us (%3.0f GB/s) |%s", sh.name, O, K, tr*1e6, gb/tr, line)
		for _, b := range []*metal.Buffer{qb, sb, xb, exb, o1, o2} {
			b.Release()
		}
	}
}
