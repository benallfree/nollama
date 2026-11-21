//go:build linux || windows

package engine

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/cuda"
)

// TestCUDAIEEE checks that the deterministic PTX (gpu_det.cu as gen_ptx.sh compiles it) rounds +, *, fma, / and sqrt
// exactly as IEEE float32 does (Go's float32 and fma32), subnormals included: the CPU's arithmetic, which the
// kernels rely on everywhere except where they flush explicitly (ftz).
func TestCUDAIEEE(t *testing.T) {
	d, err := cuda.Open(0)
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	mod, ks, err := d.Load(cudaDetPTX, "ieee_probe_det")
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Unload()
	const n = 1 << 18
	a, bb, c := make([]float32, n), make([]float32, n), make([]float32, n)
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
	bytes := func(f []float32) []byte { return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f))), 4*len(f)) }
	var ptrs [4]cuda.Ptr
	for i, f := range [][]float32{a, bb, c, nil} {
		sz := 4 * n
		if f == nil {
			sz *= 8
		}
		if ptrs[i], err = d.Alloc(sz); err != nil {
			t.Fatal(err)
		}
		defer d.Free(ptrs[i])
		if f != nil {
			if err := d.Write(ptrs[i], bytes(f)); err != nil {
				t.Fatal(err)
			}
		}
	}
	err = d.Run(cuda.Launch{K: ks["ieee_probe_det"], Grid: [3]uint32{n / 256, 1, 1}, Block: [3]uint32{256, 1, 1},
		Args: []cuda.Arg{{Ptr: ptrs[0]}, {Ptr: ptrs[1]}, {Ptr: ptrs[2]}, {Ptr: ptrs[3]}, {Bytes: u32(n)}}})
	if err != nil {
		t.Fatal(err)
	}
	o := make([]float32, 8*n)
	if err := d.Read(ptrs[3], bytes(o)); err != nil {
		t.Fatal(err)
	}
	names := []string{"p+z", "x*y+z", "fma", "x/y", "sqrt", "rint", "1/y", "x*y-z*x"}
	var bad [8]int
	for i := 0; i < n; i++ {
		x, y, z := a[i], bb[i], c[i]
		p := float32(x * y) // float32() keeps Go from fusing the product into the add
		want := [8]float32{p + z, p + z, fma32(x, y, z), x / y, float32(math.Sqrt(float64(float32(math.Abs(float64(x)))))),
			float32(math.RoundToEven(float64(float32(x * 1000)))), 1 / y, p - float32(z*x)}
		for k := range want {
			if math.Float32bits(o[8*i+k]) != math.Float32bits(want[k]) && !(want[k] != want[k] && o[8*i+k] != o[8*i+k]) {
				bad[k]++
			}
		}
	}
	for k, nm := range names {
		if bad[k] > 0 {
			t.Errorf("%s: %d of %d results differ from IEEE float32", nm, bad[k], n)
		}
	}
}
