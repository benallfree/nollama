//go:build darwin

package engine

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
)

// The arithmetic the deterministic kernels rely on, as gpu_det.metal compiles it: fast math off, no contraction.
const ieeeProbeSrc = `
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void probe(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                  device const float *c [[buffer(2)]], device float *o [[buffer(3)]],
                  constant uint &n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
	if (i >= n) return;
	float x = a[i], y = b[i], z = c[i];
	float p = x * y;
	o[8*i+0] = p + z;
	o[8*i+1] = x * y + z; // not fused
	o[8*i+2] = fma(x, y, z);
	o[8*i+3] = x / y;
	o[8*i+4] = sqrt(fabs(x));
	o[8*i+5] = rint(x * 1000.0f);
	o[8*i+6] = 1.0f / y;
	o[8*i+7] = x * y - z * x;
}`

// TestMetalIEEE checks that Metal, compiled as the deterministic kernels are, rounds +, *, fma, / and sqrt as IEEE
// float32 does (Go's float32 and fma32), with subnormal inputs and results flushed to zero (ftz32): the model of GPU
// arithmetic the reference code is written against.
func TestMetalIEEE(t *testing.T) {
	d, err := metal.Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	ps, err := d.CompilePrecise(ieeeProbeSrc, "probe")
	if err != nil {
		t.Fatal(err)
	}
	const n = 1 << 18
	bufs := make([]*metal.Buffer, 4)
	for i := range bufs {
		sz := 4 * n
		if i == 3 {
			sz *= 8
		}
		if bufs[i], err = d.NewBuffer(sz); err != nil {
			t.Fatal(err)
		}
		defer bufs[i].Release()
	}
	f := func(b *metal.Buffer) []float32 {
		return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b.Bytes()))), len(b.Bytes())/4)
	}
	a, bb, c, o := f(bufs[0]), f(bufs[1]), f(bufs[2]), f(bufs[3])
	r := rand.New(rand.NewSource(1))
	wide := func() float32 { return float32(r.NormFloat64() * math.Exp(r.NormFloat64()*3)) }
	for i := 0; i < n; i++ {
		a[i], bb[i], c[i] = wide(), wide(), wide()
		switch i % 4 { // subnormal inputs, and products that land below 2^-126
		case 1:
			a[i] = float32(r.NormFloat64() * 1e-39)
		case 2:
			a[i], bb[i] = float32(r.NormFloat64()*1e-20), float32(r.NormFloat64()*1e-19)
			c[i] = float32(r.NormFloat64() * 1e-39)
		}
	}
	err = d.Run(metal.Dispatch{P: ps["probe"], Args: []metal.Arg{{Buf: bufs[0]}, {Buf: bufs[1]}, {Buf: bufs[2]}, {Buf: bufs[3]}, {Bytes: u32(n)}},
		Groups: metal.Size{W: n / 256, H: 1, D: 1}, Threads: metal.Size{W: 256, H: 1, D: 1}})
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"p+z", "x*y+z", "fma", "x/y", "sqrt", "rint", "1/y", "x*y-z*x"}
	var bad [8]int
	for i := 0; i < n; i++ {
		x, y, z := ftz32(a[i]), ftz32(bb[i]), ftz32(c[i])
		want := [8]float32{ftz32(ftz32(x*y) + z), ftz32(ftz32(x*y) + z), ftz32(fma32(x, y, z)), ftz32(x / y),
			ftz32(float32(math.Sqrt(float64(float32(math.Abs(float64(x))))))), float32(math.RoundToEven(float64(ftz32(x * 1000)))),
			ftz32(1 / y), ftz32(ftz32(x*y) - ftz32(z*x))}
		for k := range want {
			if math.Float32bits(o[8*i+k]) != math.Float32bits(want[k]) && !(want[k] != want[k] && o[8*i+k] != o[8*i+k]) {
				bad[k]++
			}
		}
	}
	for k, nm := range names {
		if bad[k] > 0 {
			t.Errorf("%s: %d of %d results differ from IEEE float32 with flush to zero", nm, bad[k], n)
		}
	}
}
