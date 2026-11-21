//go:build darwin

package metal

import (
	"encoding/binary"
	"math"
	"testing"
)

const addSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void add(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                device float *c [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
	if (i < n) c[i] = a[i] + b[i];
}`

func TestAdd(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	t.Logf("device %q unified=%v maxWorking=%d MB", d.Name, d.Unified, d.MaxWorking>>20)
	ps, err := d.Compile(addSrc, "add")
	if err != nil {
		t.Fatal(err)
	}
	const n = 1000
	a, _ := d.NewBuffer(4 * n)
	b, _ := d.NewBuffer(4 * n)
	c, _ := d.NewBuffer(4 * n)
	defer a.Release()
	defer b.Release()
	defer c.Release()
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(a.Bytes()[4*i:], math.Float32bits(float32(i)))
		binary.LittleEndian.PutUint32(b.Bytes()[4*i:], math.Float32bits(0.5))
	}
	nb := binary.LittleEndian.AppendUint32(nil, n)
	err = d.Run(Dispatch{P: ps["add"], Args: []Arg{{Buf: a}, {Buf: b}, {Buf: c}, {Bytes: nb}},
		Groups: Size{(n + 255) / 256, 1, 1}, Threads: Size{256, 1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(c.Bytes()[4*i:])); got != float32(i)+0.5 {
			t.Fatalf("c[%d] = %v", i, got)
		}
	}
}

func TestCompileError(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	if _, err := d.Compile("kernel void x( {", "x"); err == nil {
		t.Fatal("want compile error")
	}
}
