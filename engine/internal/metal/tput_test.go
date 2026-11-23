//go:build darwin

package metal

import (
	"os"
	"testing"
)

const tputSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void imad(device int *o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
	int4 a = int4(id, id+1, id+2, id+3), b = int4(3, 5, 7, 9), c0 = 0, c1 = 1, c2 = 2, c3 = 3, c4 = 4, c5 = 5, c6 = 6, c7 = 7;
	for (int i = 0; i < 1024; i++) {
		c0 = c0 * a + b; c1 = c1 * a + b; c2 = c2 * a + b; c3 = c3 * a + b;
		c4 = c4 * a + b; c5 = c5 * a + b; c6 = c6 * a + b; c7 = c7 * a + b;
	}
	int4 s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
	o[id] = s.x + s.y + s.z + s.w;
}
kernel void fmad(device float *o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
	float4 a = float4(id, id+1, id+2, id+3) * 1e-9f, b = float4(3, 5, 7, 9), c0 = 0, c1 = 1, c2 = 2, c3 = 3, c4 = 4, c5 = 5, c6 = 6, c7 = 7;
	for (int i = 0; i < 1024; i++) {
		c0 = fma(c0, a, b); c1 = fma(c1, a, b); c2 = fma(c2, a, b); c3 = fma(c3, a, b);
		c4 = fma(c4, a, b); c5 = fma(c5, a, b); c6 = fma(c6, a, b); c7 = fma(c7, a, b);
	}
	float4 s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
	o[id] = s.x + s.y + s.z + s.w;
}
kernel void smad(device short *o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
	short4 a = short4(id, id+1, id+2, id+3), b = short4(3, 5, 7, 9), c0 = 0, c1 = 1, c2 = 2, c3 = 3, c4 = 4, c5 = 5, c6 = 6, c7 = 7;
	for (int i = 0; i < 1024; i++) {
		c0 = c0 * a + b; c1 = c1 * a + b; c2 = c2 * a + b; c3 = c3 * a + b;
		c4 = c4 * a + b; c5 = c5 * a + b; c6 = c6 * a + b; c7 = c7 * a + b;
	}
	short4 s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
	o[id] = s.x + s.y + s.z + s.w;
}
`

// TestThroughput measures the GPU's int32, float and int16 multiply-add rates (NOLLAMA_GPU_TPUT=1).
func TestThroughput(t *testing.T) {
	if os.Getenv("NOLLAMA_GPU_TPUT") == "" {
		t.Skip("set NOLLAMA_GPU_TPUT=1")
	}
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	ps, err := d.Compile(tputSrc, "imad", "fmad", "smad")
	if err != nil {
		t.Fatal(err)
	}
	n := 1 << 20
	b, _ := d.NewBuffer(4 * n)
	defer b.Release()
	for _, k := range []string{"imad", "fmad", "smad"} {
		ds := Dispatch{P: ps[k], Args: []Arg{{Buf: b}}, Groups: Size{W: uint(n / 256), H: 1, D: 1}, Threads: Size{W: 256, H: 1, D: 1}}
		d.Run(ds)
		best := 1e9
		for i := 0; i < 3; i++ {
			d.Run(ds)
			best = min(best, d.LastGPU.Seconds())
		}
		ops := float64(n) * 1024 * 8 * 4
		t.Logf("%s: %.2f T mad/s", k, ops/best/1e12)
	}
}
