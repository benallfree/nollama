//go:build amd64

package engine

import (
	"bytes"
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
	packAll := func(size, pb int, f16 []int, pack func(*[4][]byte, []byte, int)) []byte {
		w := randBlocks(r, rows*nb, size, f16)
		pw := make([]byte, rows/4*nb*pb)
		for q := 0; q < rows/4; q++ {
			var rs [4][]byte
			for i := range rs {
				rs[i] = w[(4*q+i)*nb*size:]
			}
			pack(&rs, pw[q*nb*pb:], nb)
		}
		return pw
	}
	pw := packAll(q4kBytes, q4k4x16Bytes, []int{0, 2}, q4kPack4x16)
	pw6 := packAll(q6kBytes, q6k4x16Bytes, []int{208}, q6kPack4x16)
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
		}, 16}, kabKern{"q6k" + name[3:], func() {
			for q := 0; q < rows/4; q++ {
				gemmQ6K4x16Asm(pw6[q*nb*q6k4x16Bytes:], y, gstride, nb, o[:], 4, v)
			}
		}, 16})
	}
	return ks
}

// gemmQ6K4x16Emu computes what gemmQ6K4x16 does from the packed layouts.
func gemmQ6K4x16Emu(w, y []byte, gstride, nb int, out []float32, ostride int) {
	var acc [64]float32
	le := binary.LittleEndian
	for b := 0; b < nb; b++ {
		wb := w[b*q6k4x16Bytes:]
		for g := 0; g < 4; g++ {
			yb := y[g*gstride+b*q8k16GBytes:]
			for r := 0; r < 4; r++ {
				for t := 0; t < 4; t++ {
					var sumi int32
					for c := 0; c < 16; c++ {
						var s int32
						for o := 0; o < 16; o++ {
							s += int32(wb[64*c+16*r+o]) * int32(int8(yb[80+16*(4*c+o/4)+4*t+o%4]))
						}
						sumi += s * int32(int16(le.Uint16(wb[1024+64*c+16*r+4*t:])))
					}
					for p := 0; p < 8; p++ {
						for i := 0; i < 2; i++ {
							sumi += int32(int16(le.Uint16(wb[2048+64*p+16*r+4*t+2*i:]))) *
								int32(int16(le.Uint16(yb[1104+16*p+4*t+2*i:])))
						}
					}
					yd := math.Float32frombits(le.Uint32(yb[4*t:]))
					d := math.Float32frombits(le.Uint32(wb[2560+16*r+4*t:])) * yd
					i := 16*g + 4*r + t
					acc[i] = fma32(d, float32(sumi), acc[i])
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

func TestGemmQ6K4x16(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
	for _, nb := range []int{1, 2, 3, 8, 11} {
		w := randBlocks(r, 4*nb, q6kBytes, []int{208})
		var rows [4][]byte
		for i := range rows {
			rows[i] = w[i*nb*q6kBytes : (i+1)*nb*q6kBytes]
		}
		pw := make([]byte, nb*q6k4x16Bytes)
		q6kPack4x16(&rows, pw, nb)
		var plain [16][]byte
		gstride := nb*q8k16GBytes + 48
		y := make([]byte, 4*gstride)
		for g := 0; g < 4; g++ {
			p, _ := makeQ8K4(r, nb)
			copy(plain[4*g:], p[:])
			for i := range p {
				q8kPack16(p[i], y[g*gstride:], i, nb)
			}
		}
		impls := map[string]func(o []float32){"emu": func(o []float32) { gemmQ6K4x16Emu(pw, y, gstride, nb, o, 4) }}
		if hasAVX512BW() {
			impls["avx512"] = func(o []float32) { gemmQ6K4x16Asm(pw, y, gstride, nb, o, 4, false) }
		}
		if hasAVX512VNNI() {
			impls["vnni"] = func(o []float32) { gemmQ6K4x16Asm(pw, y, gstride, nb, o, 4, true) }
		}
		for name, run := range impls {
			var o [64]float32
			run(o[:])
			for c := 0; c < 16; c++ {
				for ri := 0; ri < 4; ri++ {
					want := dotQ6KQ8KGo(rows[ri], plain[c], nb)
					if got := o[4*c+ri]; math.Float32bits(got) != math.Float32bits(want) {
						t.Errorf("%s nb=%d row %d token %d: %v, want %v", name, nb, ri, c, got, want)
					}
				}
			}
		}
		t.Logf("nb=%d: checked %d implementations", nb, len(impls))
	}
}

// byte-at-a-time packers the fast ones must match
func q4kPack4x16Ref(rows *[4][]byte, dst []byte, nb int) {
	le := binary.LittleEndian
	for b := 0; b < nb; b++ {
		o := dst[b*q4k4x16Bytes : (b+1)*q4k4x16Bytes]
		for r := 0; r < 4; r++ {
			xb := rows[r][b*q4kBytes : (b+1)*q4kBytes]
			for c := 0; c < 8; c++ {
				copy(o[64*c+16*r:64*c+16*r+16], xb[16+16*c:32+16*c])
			}
			var mn [8]uint64
			for s := 0; s < 8; s++ {
				a, m := scaleMinK4(s, xb[4:16])
				v := uint64(a) * 0x0001000100010001
				le.PutUint64(o[512+64*s+16*r:], v)
				le.PutUint64(o[520+64*s+16*r:], v)
				mn[s] = uint64(m)
			}
			for p := 0; p < 4; p++ {
				v := (mn[2*p] | mn[2*p+1]<<16) * 0x0000000100000001
				le.PutUint64(o[1024+64*p+16*r:], v)
				le.PutUint64(o[1032+64*p+16*r:], v)
			}
			d := uint64(math.Float32bits(f16(xb))) * 0x0000000100000001
			m := uint64(math.Float32bits(f16(xb[2:]))) * 0x0000000100000001
			le.PutUint64(o[1280+16*r:], d)
			le.PutUint64(o[1288+16*r:], d)
			le.PutUint64(o[1344+16*r:], m)
			le.PutUint64(o[1352+16*r:], m)
		}
	}
}

func q6kPack4x16Ref(rows *[4][]byte, dst []byte, nb int) {
	le := binary.LittleEndian
	for b := 0; b < nb; b++ {
		o := dst[b*q6k4x16Bytes : (b+1)*q6k4x16Bytes]
		for r := 0; r < 4; r++ {
			xb := rows[r][b*q6kBytes : (b+1)*q6kBytes]
			for h := 0; h < 2; h++ {
				ql, qh := xb[64*h:64*h+64], xb[128+32*h:128+32*h+32]
				for l := 0; l < 32; l++ {
					v := 128*h + l
					for j, u := range [4]byte{ql[l]&0xF | (qh[l]&3)<<4, ql[l+32]&0xF | (qh[l]>>2&3)<<4,
						ql[l]>>4 | (qh[l]>>4&3)<<4, ql[l+32]>>4 | (qh[l]>>6&3)<<4} {
						vj := v + 32*j
						o[64*(vj/16)+16*r+vj%16] = u
					}
				}
			}
			for c := 0; c < 16; c++ {
				v := uint64(uint16(int16(int8(xb[192+c])))) * 0x0001000100010001
				le.PutUint64(o[1024+64*c+16*r:], v)
				le.PutUint64(o[1032+64*c+16*r:], v)
			}
			for p := 0; p < 8; p++ {
				a := uint64(uint16(-32 * int16(int8(xb[192+2*p]))))
				c := uint64(uint16(-32 * int16(int8(xb[193+2*p]))))
				v := (a | c<<16) * 0x0000000100000001
				le.PutUint64(o[2048+64*p+16*r:], v)
				le.PutUint64(o[2056+64*p+16*r:], v)
			}
			d := uint64(math.Float32bits(f16(xb[208:]))) * 0x0000000100000001
			le.PutUint64(o[2560+16*r:], d)
			le.PutUint64(o[2568+16*r:], d)
		}
	}
}

func TestPack4x16MatchesRef(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 18))
	const nb = 5
	for _, c := range []struct {
		size, pb  int
		f16       []int
		fast, ref func(*[4][]byte, []byte, int)
	}{{q4kBytes, q4k4x16Bytes, []int{0, 2}, q4kPack4x16, q4kPack4x16Ref}, {q6kBytes, q6k4x16Bytes, []int{208}, q6kPack4x16, q6kPack4x16Ref}} {
		w := randBlocks(r, 4*nb, c.size, c.f16)
		var rows [4][]byte
		for i := range rows {
			rows[i] = w[i*nb*c.size : (i+1)*nb*c.size]
		}
		a, b := make([]byte, nb*c.pb), make([]byte, nb*c.pb)
		c.fast(&rows, a, nb)
		c.ref(&rows, b, nb)
		if !bytes.Equal(a, b) {
			t.Fatalf("packer for %d-byte blocks differs from the reference", c.size)
		}
	}
}

func BenchmarkPack4x16(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 1))
	const nb = 24
	for _, c := range []struct {
		name     string
		size, pb int
		pack     func(*[4][]byte, []byte, int)
	}{{"q4k", q4kBytes, q4k4x16Bytes, q4kPack4x16}, {"q4kRef", q4kBytes, q4k4x16Bytes, q4kPack4x16Ref},
		{"q6k", q6kBytes, q6k4x16Bytes, q6kPack4x16}, {"q6kRef", q6kBytes, q6k4x16Bytes, q6kPack4x16Ref}} {
		w := randBlocks(r, 4*nb, c.size, nil)
		var rows [4][]byte
		for i := range rows {
			rows[i] = w[i*nb*c.size : (i+1)*nb*c.size]
		}
		dst := make([]byte, nb*c.pb)
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				c.pack(&rows, dst, nb)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*4*nb), "ns/row-block")
		})
	}
}

func TestGemmF16x4x16(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 18))
	if !hasAVX512BW() {
		t.Skip("no AVX-512")
	}
	for _, nb := range []int{1, 2, 5, 128} {
		k := 16 * nb
		w := make([]byte, 4*2*k)
		for i := 0; i < 4*k; i++ {
			binary.LittleEndian.PutUint16(w[2*i:], f32ToF16(float32(r.NormFloat64()*0.1)))
		}
		var rows [4][]byte
		for i := range rows {
			rows[i] = w[i*2*k : (i+1)*2*k]
		}
		pw := make([]byte, nb*f16x4Bytes)
		f16Pack4x16(&rows, pw, nb)
		ystride := k + 3
		y := make([]float32, 16*ystride)
		for i := range y {
			y[i] = float32(r.NormFloat64())
		}
		const ostride = 6
		o := make([]float32, 16*ostride)
		gemmF16x4x16Asm(pw, y, ystride, nb, o, ostride)
		for c := 0; c < 16; c++ {
			for ri := 0; ri < 4; ri++ {
				want := dotF16F32Go(rows[ri], y[c*ystride:c*ystride+k])
				if got := o[c*ostride+ri]; math.Float32bits(got) != math.Float32bits(want) {
					t.Errorf("nb=%d row %d token %d: %v, want %v", nb, ri, c, got, want)
				}
			}
		}
	}
}
