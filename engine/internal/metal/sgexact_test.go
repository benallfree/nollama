//go:build darwin

package metal

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// One simdgroup per case: C(8x8) = A(8xK) * B(Kx8), K = 64, inputs as shorts converted to half, accumulated in float
// through simdgroup_multiply_accumulate. Exact whenever every partial sum is an integer below 2^24.
const sgExactSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void sgx(device const short *A [[buffer(0)]], device const short *B [[buffer(1)]], device float *C [[buffer(2)]],
                uint cs [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
	threadgroup half a[8 * 64], b[64 * 8];
	for (int i = lane; i < 512; i += 32) {
		a[i] = (half)A[cs * 512 + i];
		b[i] = (half)B[cs * 512 + i];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	simdgroup_matrix<float, 8, 8> c = simdgroup_matrix<float, 8, 8>(0.0f);
	for (int k = 0; k < 64; k += 8) {
		simdgroup_matrix<half, 8, 8> ma, mb;
		simdgroup_load(ma, a + k, 64);
		simdgroup_load(mb, b + k * 8, 8);
		simdgroup_multiply_accumulate(c, ma, mb, c);
	}
	simdgroup_store(c, C + cs * 64, 8);
}
`

func TestSGExact(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	ps, err := d.Compile(sgExactSrc, "sgx")
	if err != nil {
		t.Fatal(err)
	}
	type gen struct {
		name  string
		a, b  func(r *rand.Rand, i int) int16
		cases int
	}
	unif := func(lo, hi int) func(r *rand.Rand, i int) int16 {
		return func(r *rand.Rand, i int) int16 { return int16(lo + r.Intn(hi-lo+1)) }
	}
	ext := func(lo, hi int) func(r *rand.Rand, i int) int16 { // extremes only, worst-case sums
		return func(r *rand.Rand, i int) int16 {
			if r.Intn(8) == 0 {
				return int16(lo)
			}
			return int16(hi)
		}
	}
	gens := []gen{
		{"w8 x a12 (AQ=12 GEMM)", unif(-127, 127), unif(-2047, 2047), 4096},
		{"w8 x a12 extremes", ext(-127, 127), ext(-2047, 2047), 4096},
		{"w8 x a16 lo byte", ext(-127, 127), ext(0, 255), 4096},
		{"w8 x a16 hi byte", ext(-127, 127), ext(-128, 127), 4096},
	}
	r := rand.New(rand.NewSource(1))
	for _, g := range gens {
		n := g.cases
		A := make([]int16, n*512)
		B := make([]int16, n*512)
		for i := range A {
			A[i] = g.a(r, i)
			B[i] = g.b(r, i)
		}
		ab, _ := d.NewBuffer(len(A) * 2)
		bb, _ := d.NewBuffer(len(B) * 2)
		cb, _ := d.NewBuffer(n * 64 * 4)
		for i, v := range A {
			binary.LittleEndian.PutUint16(ab.Bytes()[2*i:], uint16(v))
		}
		for i, v := range B {
			binary.LittleEndian.PutUint16(bb.Bytes()[2*i:], uint16(v))
		}
		if err := d.Run(Dispatch{P: ps["sgx"], Args: []Arg{{Buf: ab}, {Buf: bb}, {Buf: cb}}, Groups: Size{W: uint(n), H: 1, D: 1},
			Threads: Size{W: 32, H: 1, D: 1}}); err != nil {
			t.Fatal(err)
		}
		bad, maxAbs := 0, int64(0)
		for c := 0; c < n; c++ {
			for i := 0; i < 8; i++ {
				for j := 0; j < 8; j++ {
					var want int64
					for k := 0; k < 64; k++ {
						want += int64(A[c*512+i*64+k]) * int64(B[c*512+k*8+j])
					}
					maxAbs = max(maxAbs, want, -want)
					got := math.Float32frombits(binary.LittleEndian.Uint32(cb.Bytes()[4*(c*64+i*8+j):]))
					if float64(got) != float64(want) {
						if bad < 3 {
							t.Errorf("%s case %d [%d,%d]: got %v want %d", g.name, c, i, j, got, want)
						}
						bad++
					}
				}
			}
		}
		t.Logf("%s: %d cases, %d mismatches, max |sum| %d (2^24 = %d)", g.name, n*64, bad, maxAbs, 1<<24)
		ab.Release()
		bb.Release()
		cb.Release()
	}
}
