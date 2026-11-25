//go:build amd64

package engine

import (
	"encoding/binary"
	"math"
	"os"
	"unsafe"
)

//go:noescape
func gemmQ6K2x4PlainAVX2(w0, w1, s0, s1, y unsafe.Pointer, stride, nb int, out *float32)

//go:noescape
func gemmQ4K2x4PlainAVX512(w0, w1, s0, s1, y unsafe.Pointer, stride, nb int, out *float32)

//go:noescape
func gemmQ6K2x4PlainAVX512(w0, w1, s0, s1, y unsafe.Pointer, stride, nb int, out *float32)

//go:noescape
func gemmQ4K2x4PlainAVX2(w0, w1, s0, s1, y unsafe.Pointer, stride, nb int, out *float32)

// noGemm (AGENT_ENGINE_NOGEMM set) keeps the 4-column prompt kernels, for
// comparing against them.
var noGemm = os.Getenv("AGENT_ENGINE_NOGEMM") != ""

// useAVX512 picks the AVX-512BW prompt kernels, only when NOLLAMA_AVX512 is set: on Cascade Lake (2026-11-24) they
// were no faster than the AVX2 ones (512-bit integer multiplies run on one port there, so they don't double the
// rate), and their wider pre-decoded scales cost more to build.
var useAVX512 = hasAVX512BW() && os.Getenv("NOLLAMA_AVX512") != ""

// hasAVX512BW: AVX-512 F, BW and VL, with the OS saving the opmask and ZMM state.
func hasAVX512BW() bool {
	if !hasAVX2FMA() {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	const f, bw, vl = 1 << 16, 1 << 30, 1 << 31
	if ebx7&(f|bw|vl) != f|bw|vl {
		return false
	}
	xcr0, _ := xgetbv()
	return xcr0&0xe6 == 0xe6
}

// The amd64 Q4_K prompt kernels read a row's block scales pre-decoded, q4kScaleBytes per block:
//
//	sc[8]    int16: the 6-bit scale of each 32-value sub-block
//	mdup[16] int16: the 6-bit mins, each twice (m0 m0 m1 m1 ...), to multiply the Q8_K 16-value sums pairwise
//	d, dmin  float32 (the block's f16 values, converted exactly)
//
// (80 bytes per block are reserved; 56 are used.)
func q4kScalesX86(w, dst []byte, nb int) {
	for b := 0; b < nb; b++ {
		xb, d := w[b*q4kBytes:], dst[b*q4kScaleBytes:]
		for j := 0; j < 8; j++ {
			s, m := scaleMinK4(j, xb[4:16])
			binary.LittleEndian.PutUint16(d[2*j:], uint16(s))
			binary.LittleEndian.PutUint16(d[16+4*j:], uint16(m))
			binary.LittleEndian.PutUint16(d[16+4*j+2:], uint16(m))
		}
		binary.LittleEndian.PutUint32(d[48:], math.Float32bits(f16(xb)))
		binary.LittleEndian.PutUint32(d[52:], math.Float32bits(f16(xb[2:])))
	}
}

// q6kScaleBytes is the pre-decoded size of one Q6_K block's scales for the amd64 prompt kernel:
//
//	sv[2][4][16] int16: for half h and 32-value group m, sub-block 8h+2m's scale 8 times, then 8h+2m+1's
//	                    (one per 16-bit lane of a 256-bit product, low 128 bits then high)
//	neg[16]      int16: -32 * each scale, for the activations' 16-value sums (Q6_K values are stored +32)
//	d            float32
const q6kScaleBytes = 320

func q6kScalesX86(w, dst []byte, nb int) {
	for b := 0; b < nb; b++ {
		xb, d := w[b*q6kBytes:], dst[b*q6kScaleBytes:]
		sc := xb[192:208]
		for h := 0; h < 2; h++ {
			for m := 0; m < 4; m++ {
				v := d[32*(4*h+m):]
				for i := 0; i < 16; i++ {
					binary.LittleEndian.PutUint16(v[2*i:], uint16(int16(int8(sc[8*h+2*m+i/8]))))
				}
			}
		}
		for i := 0; i < 16; i++ {
			binary.LittleEndian.PutUint16(d[256+2*i:], uint16(-32*int16(int8(sc[i]))))
		}
		binary.LittleEndian.PutUint32(d[288:], math.Float32bits(f16(xb[208:])))
	}
}

// q4kScalesX86_512 is the AVX-512 kernel's layout, q4kScaleSize512 bytes per block:
//
//	sv[4][32] int16: for chunk c, sub-block 2c's scale 16 times, then 2c+1's (the two 256-bit halves)
//	mdup[16]  int16: the mins, each twice
//	d, dmin   float32
const q4kScaleSize512 = 320

func q4kScalesX86_512(w, dst []byte, nb int) {
	for b := 0; b < nb; b++ {
		xb, d := w[b*q4kBytes:], dst[b*q4kScaleSize512:]
		for j := 0; j < 8; j++ {
			s, m := scaleMinK4(j, xb[4:16])
			for i := 0; i < 16; i++ {
				binary.LittleEndian.PutUint16(d[32*j+2*i:], uint16(s))
			}
			binary.LittleEndian.PutUint16(d[256+4*j:], uint16(m))
			binary.LittleEndian.PutUint16(d[256+4*j+2:], uint16(m))
		}
		binary.LittleEndian.PutUint32(d[288:], math.Float32bits(f16(xb)))
		binary.LittleEndian.PutUint32(d[292:], math.Float32bits(f16(xb[2:])))
	}
}

// q6kScalesX86_512: like q6kScalesX86 (same size), but sv is [2][2][32]: for half h and 64-value group p,
// sub-blocks 8h+4p .. 8h+4p+3, each scale 8 times (one 128-bit quarter of a 512-bit product each).
func q6kScalesX86_512(w, dst []byte, nb int) {
	for b := 0; b < nb; b++ {
		xb, d := w[b*q6kBytes:], dst[b*q6kScaleBytes:]
		sc := xb[192:208]
		for i := 0; i < 128; i++ {
			binary.LittleEndian.PutUint16(d[2*i:], uint16(int16(int8(sc[i/8]))))
		}
		for i := 0; i < 16; i++ {
			binary.LittleEndian.PutUint16(d[256+2*i:], uint16(-32*int16(int8(sc[i]))))
		}
		binary.LittleEndian.PutUint32(d[288:], math.Float32bits(f16(xb[208:])))
	}
}

// archIntAsm turns on the 2-row x 4-column Q4_K and Q6_K prompt kernels.
func archIntAsm() {
	if noGemm {
		return
	}
	q4, q6 := gemmQ4K2x4PlainAVX2, gemmQ6K2x4PlainAVX2
	q4kScalesPlain, q4kScaleSize = q4kScalesX86, q4kScaleBytes
	q6kScales, q6kScaleSize = q6kScalesX86, q6kScaleBytes
	if useAVX512 {
		q4, q6 = gemmQ4K2x4PlainAVX512, gemmQ6K2x4PlainAVX512
		q4kScalesPlain, q4kScaleSize = q4kScalesX86_512, q4kScaleSize512
		q6kScales = q6kScalesX86_512
	}
	gemmQ4K2x4Plain = func(w0, w1, s0, s1, y []byte, stride, nb int, out *[8]float32) {
		_, _, _ = w0[nb*q4kBytes-1], w1[nb*q4kBytes-1], y[3*stride+nb*q8kBytes-1]
		_, _ = s0[nb*q4kScaleSize-1], s1[nb*q4kScaleSize-1]
		q4(bptr(w0), bptr(w1), bptr(s0), bptr(s1), bptr(y), stride, nb, &out[0])
	}
	gemmQ6K2x4Plain = func(w0, w1, s0, s1, y []byte, stride, nb int, out *[8]float32) {
		_, _, _ = w0[nb*q6kBytes-1], w1[nb*q6kBytes-1], y[3*stride+nb*q8kBytes-1]
		_, _ = s0[nb*q6kScaleBytes-1], s1[nb*q6kScaleBytes-1]
		q6(bptr(w0), bptr(w1), bptr(s0), bptr(s1), bptr(y), stride, nb, &out[0])
	}
	if hasAVX512BW() && !no4x16 {
		vnni := hasAVX512VNNI()
		gemmQ4K4x16 = func(w, y []byte, gstride, nb int, out []float32, ostride int) {
			gemmQ4K4x16Asm(w, y, gstride, nb, out, ostride, vnni)
		}
	}
}

func archFloatAsm() {}
