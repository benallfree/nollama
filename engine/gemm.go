package engine

import (
	"encoding/binary"
	"math"
)

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

// gemmQ4K2x4Plain is gemmQ4K2x4 reading 4 plain Q8_K activation rows stride bytes apart (amd64), with scales
// from q4kScalesPlain, q4kScaleSize bytes per block.
var (
	gemmQ4K2x4Plain func(w0, w1, s0, s1, y []byte, stride, nb int, out *[8]float32)
	q4kScalesPlain  func(w, dst []byte, nb int)
	q4kScaleSize    int
)

// gemmQ6K2x4Plain is the Q6_K counterpart (amd64), with scales from q6kScales, q6kScaleSize bytes per block.
var (
	gemmQ6K2x4Plain func(w0, w1, s0, s1, y []byte, stride, nb int, out *[8]float32)
	q6kScales       func(w, dst []byte, nb int)
	q6kScaleSize    int
)

// gemmQ4K4x16, when set (amd64 with AVX-512BW), computes 4 Q4_K rows x 16 tokens: w is the 4 rows packed by
// q4kPack4x16, y 4 token groups packed by q8kPack16, gstride bytes apart; out[c*ostride+r] is row r times token c,
// in the reference order.
var gemmQ4K4x16 func(w, y []byte, gstride, nb int, out []float32, ostride int)

const (
	q4k4x16Bytes = 1408 // per 4 rows per block: qs 512, scale vectors 8*64, min vectors 4*64, d 64, dmin 64
	q8k16GBytes  = 1104 // per 4 tokens per block: yd 16, bsum pairs 4*16, qs 64*16
)

// q4kPack4x16 packs 4 Q4_K rows (nb blocks each) for gemmQ4K4x16, q4k4x16Bytes per block:
//
//	qs[8][4][16]  chunk c holds bytes 16c..16c+15 of each row's qs (row r at 16r)
//	sc[8][4][8]   int16: sub-block s, row r: the row's scale for s, 8 times (lane r of a VPMADDWD operand)
//	mn[4][4][8]   int16: sub-block pair p, row r: (min 2p, min 2p+1) 4 times
//	d[4][4]       float32: row r's d, 4 times
//	dmin[4][4]    float32
func q4kPack4x16(rows *[4][]byte, dst []byte, nb int) {
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

// q8kPack16 packs one Q8_K token row (nb blocks) into slot t of a token group for gemmQ4K4x16, q8k16GBytes
// per block:
//
//	yd[4]         float32: token t's scale
//	bs[4][4][2]   int16: sub-block pair p, token t: (sum of sub-block 2p, sum of sub-block 2p+1)
//	qs[64][4][4]  int8: group of 4 values k (values 4k..4k+3), token t
func q8kPack16(src, dst []byte, t, nb int) {
	for b := 0; b < nb; b++ {
		o, yb := dst[b*q8k16GBytes:], src[b*q8kBytes:]
		copy(o[4*t:4*t+4], yb[:4])
		bs := func(k int) int16 { return int16(binary.LittleEndian.Uint16(yb[q8kBsumOff+2*k:])) }
		for s := 0; s < 8; s++ {
			binary.LittleEndian.PutUint16(o[16+16*(s/2)+4*t+2*(s%2):], uint16(bs(2*s)+bs(2*s+1)))
		}
		for k := 0; k < 64; k++ {
			copy(o[80+16*k+4*t:80+16*k+4*t+4], yb[q8kQsOff+4*k:])
		}
	}
}

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
	q, f, q6, qp, q6p, q16 := gemmQ4K2x4, gemmF16x2x4, gemmQ6K1x4, gemmQ4K2x4Plain, gemmQ6K2x4Plain, gemmQ4K4x16
	gemmQ4K2x4, gemmF16x2x4, gemmQ6K1x4, gemmQ4K2x4Plain, gemmQ6K2x4Plain, gemmQ4K4x16 = nil, nil, nil, nil, nil, nil
	return func() {
		gemmQ4K2x4, gemmF16x2x4, gemmQ6K1x4, gemmQ4K2x4Plain, gemmQ6K2x4Plain, gemmQ4K4x16 = q, f, q6, qp, q6p, q16
	}
}
