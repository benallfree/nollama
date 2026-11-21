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

// gemmF16x2x4, when set, computes out[r*4+c] = dot(F16 row r, y[c*stride:])
// for rows w0, w1 and 4 float activation rows stride floats apart.
var gemmF16x2x4 func(w0, w1 []byte, y []float32, stride int, out *[8]float32)

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

// The q8k6 layout holds 4 activation rows' Q8_K blocks interleaved for the
// Q6_K prompt kernel. Per 256-value block:
//
//	d[4]          float32, one per row
//	qs[4][4][64]  int8: group g (values 64g..64g+63), row j, transposed in
//	              4-byte lanes: byte 16*l + 4*i + b is value 16*i + 4*l + b
//
// matching Q6_K weights whose 4 sub-blocks per group are transposed the same
// way.
const q8k6Bytes = 16 + 4*qk

// gemmQ6K1x4, when set, computes out[c] = dot(Q6_K row w, column c) against
// one q8k6 group of 4 columns.
var gemmQ6K1x4 func(w, y []byte, nb int, out *[4]float32)

// packQ8K6 copies nb plain Q8_K blocks of one activation row into lane j of a
// q8k6 group.
func packQ8K6(src, dst []byte, j, nb int) {
	for b := 0; b < nb; b++ {
		s, d := src[b*q8kBytes:], dst[b*q8k6Bytes:]
		copy(d[4*j:4*j+4], s[:4])
		for g := 0; g < qk/64; g++ {
			sq, dq := s[q8kQsOff+64*g:], d[16+256*g+64*j:]
			for l := 0; l < 4; l++ {
				for i := 0; i < 4; i++ {
					copy(dq[16*l+4*i:16*l+4*i+4], sq[16*i+4*l:16*i+4*l+4])
				}
			}
		}
	}
}

// promptKernelsOff turns the 2-row prompt kernels off (for before/after
// comparisons) and returns a function that turns them back on.
func promptKernelsOff() (restore func()) {
	q, f, q6 := gemmQ4K2x4, gemmF16x2x4, gemmQ6K1x4
	gemmQ4K2x4, gemmF16x2x4, gemmQ6K1x4 = nil, nil, nil
	return func() { gemmQ4K2x4, gemmF16x2x4, gemmQ6K1x4 = q, f, q6 }
}
