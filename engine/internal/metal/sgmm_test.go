//go:build darwin

package metal

import (
	"os"
	"testing"
)

// Peak simdgroup-matrix multiply-accumulate rates: each simdgroup keeps 4 independent 8x8 accumulators in registers.
const sgmmSrc = `
#include <metal_stdlib>
using namespace metal;
template <typename TI, typename TA>
static inline void body(device float *o, uint id, uint lane) {
	simdgroup_matrix<TI, 8, 8> a = simdgroup_matrix<TI, 8, 8>((TI)(id & 7) * (TI)0.001);
	simdgroup_matrix<TI, 8, 8> b = simdgroup_matrix<TI, 8, 8>((TI)0.5);
	simdgroup_matrix<TA, 8, 8> c0 = simdgroup_matrix<TA, 8, 8>(0), c1 = c0, c2 = c0, c3 = c0;
	simdgroup_matrix<TA, 8, 8> c4 = c0, c5 = c0, c6 = c0, c7 = c0;
	for (int i = 0; i < 1024; i++) {
		simdgroup_multiply_accumulate(c0, a, b, c0);
		simdgroup_multiply_accumulate(c1, b, a, c1);
		simdgroup_multiply_accumulate(c2, a, a, c2);
		simdgroup_multiply_accumulate(c3, b, b, c3);
		simdgroup_multiply_accumulate(c4, a, b, c4);
		simdgroup_multiply_accumulate(c5, b, a, c5);
		simdgroup_multiply_accumulate(c6, a, a, c6);
		simdgroup_multiply_accumulate(c7, b, b, c7);
	}
	simdgroup_matrix<TA, 8, 8> s;
	simdgroup_multiply_accumulate(s, c0, c1, c2);
	simdgroup_multiply_accumulate(s, c3, s, c4);
	simdgroup_multiply_accumulate(s, c5, s, c6);
	simdgroup_multiply_accumulate(s, c7, s, s);
	o[id] = (float)s.thread_elements()[0] + (float)s.thread_elements()[1];
}
kernel void sg_hf(device float *o [[buffer(0)]], uint id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { body<half, float>(o, id, lane); }
kernel void sg_ff(device float *o [[buffer(0)]], uint id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { body<float, float>(o, id, lane); }
kernel void sg_hh(device float *o [[buffer(0)]], uint id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { body<half, half>(o, id, lane); }
`

// TestSGMM measures simdgroup_matrix multiply-accumulate throughput (NOLLAMA_GPU_TPUT=1), in scalar MACs per second.
func TestSGMM(t *testing.T) {
	if os.Getenv("NOLLAMA_GPU_TPUT") == "" {
		t.Skip("set NOLLAMA_GPU_TPUT=1")
	}
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	ks := []string{"sg_hf", "sg_ff", "sg_hh"}
	ps, err := d.Compile(sgmmSrc, ks...)
	if err != nil {
		t.Fatal(err)
	}
	n := 1 << 20 // threads
	b, _ := d.NewBuffer(4 * n)
	defer b.Release()
	for _, k := range ks {
		ds := Dispatch{P: ps[k], Args: []Arg{{Buf: b}}, Groups: Size{W: uint(n / 256), H: 1, D: 1}, Threads: Size{W: 256, H: 1, D: 1}}
		if err := d.Run(ds); err != nil {
			t.Fatal(err)
		}
		best := 1e9
		for i := 0; i < 3; i++ {
			d.Run(ds)
			best = min(best, d.LastGPU.Seconds())
		}
		macs := float64(n/32) * 1024 * 8 * 512 // simdgroups x iters x mmas x 8*8*8
		t.Logf("%s: %.2f T mac/s", k, macs/best/1e12)
	}
}
