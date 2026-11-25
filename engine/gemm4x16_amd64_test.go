//go:build amd64

package engine

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// gemmQ4K4x16Emu computes what gemmQ4K4x16 does, lane by lane, from the same packed layouts (checks the packing
// on any amd64 machine, AVX-512 or not).
func gemmQ4K4x16Emu(w, y []byte, gstride, nb int, out []float32, ostride int) {
	var acc [64]float32
	for b := 0; b < nb; b++ {
		wb := w[b*q4k4x16Bytes:]
		for g := 0; g < 4; g++ {
			yb := y[g*gstride+b*q8k16GBytes:]
			for r := 0; r < 4; r++ {
				for t := 0; t < 4; t++ {
					var sumi, mins int32
					for s := 0; s < 8; s++ {
						var s16 int32
						for o := 0; o < 32; o++ {
							q := wb[64*(2*(s/2)+o/16)+16*r+o%16]
							if s%2 == 1 {
								q >>= 4
							}
							kq := 8*s + o/4
							s16 += int32(q&15) * int32(int8(yb[80+16*kq+4*t+o%4]))
						}
						sumi += s16 * int32(int16(binary.LittleEndian.Uint16(wb[512+64*s+16*r+4*t:])))
					}
					for p := 0; p < 4; p++ {
						for i := 0; i < 2; i++ {
							m := int32(int16(binary.LittleEndian.Uint16(wb[1024+64*p+16*r+4*t+2*i:])))
							mins += m * int32(int16(binary.LittleEndian.Uint16(yb[16+16*p+4*t+2*i:])))
						}
					}
					yd := math.Float32frombits(binary.LittleEndian.Uint32(yb[4*t:]))
					d := math.Float32frombits(binary.LittleEndian.Uint32(wb[1280+16*r+4*t:])) * yd
					dmin := math.Float32frombits(binary.LittleEndian.Uint32(wb[1344+16*r+4*t:])) * yd
					i := 16*g + 4*r + t
					acc[i] = fma32(d, float32(sumi), acc[i])
					acc[i] = fma32(-dmin, float32(mins), acc[i])
				}
			}
		}
	}
	for g := 0; g < 4; g++ {
		for r := 0; r < 4; r++ {
			for t := 0; t < 4; t++ {
				out[(4*g+t)*ostride+r] = acc[16*g+4*r+t]
			}
		}
	}
}

func TestGemmQ4K4x16(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for _, nb := range []int{1, 2, 3, 8, 11} {
		w := randBlocks(r, 4*nb, q4kBytes, []int{0, 2})
		var rows [4][]byte
		for i := range rows {
			rows[i] = w[i*nb*q4kBytes : (i+1)*nb*q4kBytes]
		}
		pw := make([]byte, nb*q4k4x16Bytes)
		q4kPack4x16(&rows, pw, nb)
		var plain [16][]byte
		gstride := nb*q8k16GBytes + 48 // groups need not be adjacent
		y := make([]byte, 4*gstride)
		for g := 0; g < 4; g++ {
			p, _ := makeQ8K4(r, nb)
			copy(plain[4*g:], p[:])
			for i := range p {
				q8kPack16(p[i], y[g*gstride:], i, nb)
			}
		}
		type impl struct {
			name string
			run  func(*[64]float32)
		}
		impls := []impl{{"emu", func(o *[64]float32) { gemmQ4K4x16Emu(pw, y, gstride, nb, o[:], 4) }}}
		if hasAVX512BW() {
			impls = append(impls, impl{"avx512", func(o *[64]float32) { gemmQ4K4x16Asm(pw, y, gstride, nb, o[:], 4, false) }})
		}
		if hasAVX512VNNI() {
			impls = append(impls, impl{"vnni", func(o *[64]float32) { gemmQ4K4x16Asm(pw, y, gstride, nb, o[:], 4, true) }})
		}
		for _, im := range impls {
			var o [64]float32
			im.run(&o)
			for g := 0; g < 4; g++ {
				for ri := 0; ri < 4; ri++ {
					for tt := 0; tt < 4; tt++ {
						want := dotQ4KQ8KGo(rows[ri], plain[4*g+tt], nb)
						if got := o[(4*g+tt)*4+ri]; math.Float32bits(got) != math.Float32bits(want) {
							t.Errorf("%s nb=%d row %d token %d: %v, want %v", im.name, nb, ri, 4*g+tt, got, want)
						}
					}
				}
			}
		}
		t.Logf("nb=%d: checked %d implementations", nb, len(impls))
	}
}

// kab4x16: TestKernelAB entries for the 4x16 prototype (16 tokens per call).
type kabKern struct {
	name string
	run  func()
	toks int
}

func kab4x16(r *rand.Rand, rows, nb int) []kabKern {
	if !hasAVX512BW() {
		return nil
	}
	w := randBlocks(r, rows*nb, q4kBytes, []int{0, 2})
	pw := make([]byte, rows/4*nb*q4k4x16Bytes)
	for q := 0; q < rows/4; q++ {
		var rs [4][]byte
		for i := range rs {
			rs[i] = w[(4*q+i)*nb*q4kBytes:]
		}
		q4kPack4x16(&rs, pw[q*nb*q4k4x16Bytes:], nb)
	}
	gstride := nb * q8k16GBytes
	y := make([]byte, 4*gstride)
	for g := 0; g < 4; g++ {
		p, _ := makeQ8K4(r, nb)
		for i := range p {
			q8kPack16(p[i], y[g*gstride:], i, nb)
		}
	}
	var o [64]float32
	var ks []kabKern
	for _, v := range []bool{false, true} {
		if v && !hasAVX512VNNI() {
			continue
		}
		name := "q4k 4x16 (proto)"
		if v {
			name = "q4k 4x16 vnni"
		}
		ks = append(ks, kabKern{name, func() {
			for q := 0; q < rows/4; q++ {
				gemmQ4K4x16Asm(pw[q*nb*q4k4x16Bytes:], y, gstride, nb, o[:], 4, v)
			}
		}, 16})
	}
	return ks
}
