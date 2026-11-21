package engine

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"math"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
)

//go:embed gpu.metal
var gpuSource string

// gpuKernels are the kernels gpuSource defines.
var gpuKernels = []string{"matvec_q4k_f32", "matvec_q8_0_f32", "rmsnorm_f32", "rope_neox_f32", "swiglu_f32", "add_f32"}

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

func u32(vs ...uint32) []byte {
	b := make([]byte, 4*len(vs))
	for i, v := range vs {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	return b
}

// rmsnormDispatch normalizes rows rows of n floats from x into out (may be x), scaled by w (n floats).
func (u *gpu) rmsnormDispatch(x, w, out metal.Arg, rows, n int, eps float64) metal.Dispatch {
	p := u.ps["rmsnorm_f32"]
	nt := min(p.MaxThreads, 1024, (n+31)&^31)
	return metal.Dispatch{
		P:       p,
		Args:    []metal.Arg{x, w, out, {Bytes: u32(uint32(n), math.Float32bits(float32(eps)))}},
		Groups:  metal.Size{W: uint(rows), H: 1, D: 1},
		Threads: metal.Size{W: uint(nt), H: 1, D: 1},
		Shared:  32 * 4,
	}
}

// ropeDispatch applies NeoX RoPE in place to x ([tokens][heads][headDim]) with per-token cos/sin tables.
func (u *gpu) ropeDispatch(x, cs, sn metal.Arg, tokens, heads, headDim int) metal.Dispatch {
	half := headDim / 2
	tw := min(half, 64)
	return metal.Dispatch{
		P:       u.ps["rope_neox_f32"],
		Args:    []metal.Arg{x, cs, sn, {Bytes: u32(uint32(heads), uint32(headDim))}},
		Groups:  metal.Size{W: uint((half + tw - 1) / tw), H: uint(heads), D: uint(tokens)},
		Threads: metal.Size{W: uint(tw), H: 1, D: 1},
	}
}

// elementwise is a 1-D dispatch over n elements.
func (u *gpu) elementwise(kernel string, n int, args ...metal.Arg) metal.Dispatch {
	args = append(args, metal.Arg{Bytes: u32(uint32(n))})
	return metal.Dispatch{
		P:       u.ps[kernel],
		Args:    args,
		Groups:  metal.Size{W: uint((n + 255) / 256), H: 1, D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
	}
}
