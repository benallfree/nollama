package engine

import "encoding/binary"

// The q8k4 layout holds 4 activation rows' Q8_K blocks interleaved, for the
// 2-row x 4-column prompt kernels. Per 256-value block:
//
//	d[4]          float32, one per row (column of the product)
//	qs[4][4][64]  int8: chunk c, row j, 64 values reordered by q8k4Perm
//	bs[8][4]      int16: sum of each 32-value sub-block k, row j
//
// The reordering matches a Q4_K chunk after its low- and high-nibble weights
// are zipped 32 bits at a time: byte 16*v + 4*g + b of a chunk comes from
// value q8k4Perm[16*v + 4*g] + b.
const q8k4Bytes = 16 + 4*qk + 8*4*2

var q8k4Perm = func() (p [64]int) {
	for v := 0; v < 4; v++ {
		half, kb := v/2, 2*(v%2)
		for g := 0; g < 4; g++ {
			p[16*v+4*g] = (g%2)*32 + half*16 + 4*(kb+g/2)
		}
	}
	return
}()

// Prompt kernels for Q4_K weights, set where the CPU has them:
// q4kScales unpacks a row's block scales into q4kScaleBytes per block (once
// per row and matmul, reused across activation groups); gemmQ4K2x4 computes
// out[r*4+c] = dot(row r, column c) for rows w0, w1 (scales s0, s1) against
// one q8k4 group of 4 columns.
var (
	q4kScales  func(w, dst []byte, nb int)
	gemmQ4K2x4 func(w0, w1, s0, s1, y []byte, nb int, out *[8]float32)
)

const q4kScaleBytes = 80

// packQ8K4 copies nb plain Q8_K blocks of one activation row into lane j of a
// q8k4 group.
func packQ8K4(src, dst []byte, j, nb int) {
	for b := 0; b < nb; b++ {
		s, d := src[b*q8kBytes:], dst[b*q8k4Bytes:]
		copy(d[4*j:4*j+4], s[:4])
		for c := 0; c < qk/64; c++ {
			sq, dq := s[q8kQsOff+64*c:], d[16+256*c+64*j:]
			for i := 0; i < 64; i += 4 {
				o := q8k4Perm[i]
				copy(dq[i:i+4], sq[o:o+4])
			}
		}
		for k := 0; k < 8; k++ {
			s0 := int16(binary.LittleEndian.Uint16(s[q8kBsumOff+4*k:]))
			s1 := int16(binary.LittleEndian.Uint16(s[q8kBsumOff+4*k+2:]))
			binary.LittleEndian.PutUint16(d[16+4*qk+8*k+2*j:], uint16(s0+s1))
		}
	}
}
