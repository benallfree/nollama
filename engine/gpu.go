package engine

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
)

//go:embed gpu.metal
var gpuSource string

// gpuKernels are the kernels gpuSource defines.
var gpuKernels = []string{"matvec_q4k_f32", "matvec_q8_0_f32"}

// gpu is a model's GPU state: the device, compiled kernels and the mmapped file wrapped as one buffer.
// Experimental: nothing in forward uses it yet.
type gpu struct {
	dev     *metal.Device
	ps      map[string]*metal.Pipeline
	weights *metal.Buffer // the whole GGUF file, zero-copy
	base    uintptr       // address of the file's first byte
}

func openGPU(g *ggufFile) (*gpu, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if !dev.Unified {
		dev.Close()
		return nil, errors.New("engine: GPU without unified memory isn't supported yet")
	}
	ps, err := dev.Compile(gpuSource, gpuKernels...)
	if err != nil {
		dev.Close()
		return nil, err
	}
	wb, err := dev.Wrap(g.data)
	if err != nil {
		dev.Close()
		return nil, err
	}
	return &gpu{dev: dev, ps: ps, weights: wb, base: uintptr(unsafe.Pointer(unsafe.SliceData(g.data)))}, nil
}

func (u *gpu) close() {
	u.weights.Release()
	u.dev.Close()
}

// weightArg is w's bytes as a kernel argument: an offset into the wrapped file.
func (u *gpu) weightArg(w *weight) metal.Arg {
	return metal.Arg{Buf: u.weights, Offset: int(uintptr(unsafe.Pointer(unsafe.SliceData(w.data))) - u.base)}
}

// gpuMatvec reports whether matvecDispatch handles weights of type t.
func gpuMatvec(t uint32) bool { return t == typeQ4_K || t == typeQ8_0 }

// matvecDispatch computes out[t*rows+r] = W·x[t] for n tokens; x holds n rows of w.cols floats.
func (u *gpu) matvecDispatch(w *weight, x, out *metal.Buffer, n int) metal.Dispatch {
	var kernel string
	var nb int
	switch w.typ {
	case typeQ4_K:
		kernel, nb = "matvec_q4k_f32", w.cols/qk
	case typeQ8_0:
		kernel, nb = "matvec_q8_0_f32", w.cols/32
	default:
		panic("gpu: matvec supports Q4_K and Q8_0, got " + typeName(w.typ))
	}
	const nsg = 4 // rows (simdgroups) per threadgroup
	args := make([]byte, 16)
	binary.LittleEndian.PutUint32(args[0:], uint32(w.rows))
	binary.LittleEndian.PutUint32(args[4:], uint32(nb))
	binary.LittleEndian.PutUint32(args[8:], uint32(w.rowBytes))
	binary.LittleEndian.PutUint32(args[12:], uint32(w.cols))
	return metal.Dispatch{
		P:       u.ps[kernel],
		Args:    []metal.Arg{u.weightArg(w), {Buf: x}, {Buf: out}, {Bytes: args}},
		Groups:  metal.Size{W: uint((w.rows + nsg - 1) / nsg), H: uint(n), D: 1},
		Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
	}
}
