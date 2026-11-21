package engine

//go:generate sh gen_ptx.sh

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/cuda"
)

// gpu_cuda.ptx is gpu.cu compiled for compute capability 7.5 (T4 and newer: A10, A100, L4, L40S, H100, RTX 20xx+).
//
//go:embed gpu_cuda.ptx
var cudaPTX string

var cudaKernels = []string{"matvec_q4k_f32", "matvec_q8_0_f32", "matmul_q8_0_f32", "rmsnorm_f32", "rope_neox_f32",
	"swiglu_f32", "add_f32", "copy_f32", "kv_store_f32", "attention_f32",
	"attention_split_f32", "attention_combine_f32"}

// cudaBackend is the gpuBackend for NVIDIA GPUs, through the driver API (no toolkit needed).
type cudaBackend struct {
	dev     *cuda.Device
	mod     *cuda.Module
	ks      map[string]*cuda.Kernel
	weights cuda.Ptr // the model file, uploaded once
	base    uintptr  // address of the file's first byte in host memory
}

func openCUDA(det bool) (gpuBackend, error) {
	if det {
		return nil, errors.New("cuda: no deterministic kernels yet")
	}
	dev, err := cuda.Open(0)
	if err != nil {
		return nil, err
	}
	if dev.Major < 7 || dev.Major == 7 && dev.Minor < 5 {
		dev.Close()
		return nil, fmt.Errorf("cuda: %s is compute %d.%d; nollama needs 7.5 or newer", dev.Name, dev.Major, dev.Minor)
	}
	mod, ks, err := dev.Load(cudaPTX, cudaKernels...)
	if err != nil {
		dev.Close()
		return nil, err
	}
	return &cudaBackend{dev: dev, mod: mod, ks: ks}, nil
}

func (c *cudaBackend) name() string {
	return fmt.Sprintf("cuda (%s, %d GB, compute %d.%d)", c.dev.Name, c.dev.Memory>>30, c.dev.Major, c.dev.Minor)
}
func (c *cudaBackend) unified() bool          { return false }
func (c *cudaBackend) supports(t uint32) bool { return t == typeQ4_K || t == typeQ8_0 }

func (c *cudaBackend) loadWeights(file []byte) error {
	if uint64(len(file)) > c.dev.Memory {
		return fmt.Errorf("model (%d MB) is larger than the GPU's memory (%d MB)", len(file)>>20, c.dev.Memory>>20)
	}
	p, err := c.dev.Alloc(len(file))
	if err != nil {
		return err
	}
	const chunk = 256 << 20
	for off := 0; off < len(file); off += chunk {
		if err := c.dev.Write(p+cuda.Ptr(off), file[off:min(off+chunk, len(file))]); err != nil {
			c.dev.Free(p)
			return err
		}
	}
	c.weights, c.base = p, uintptr(unsafe.Pointer(unsafe.SliceData(file)))
	return nil
}

func (c *cudaBackend) close() {
	if c.weights != 0 {
		c.dev.Free(c.weights)
	}
	c.mod.Unload()
	c.dev.Close()
}

type cudaBuf struct {
	c *cudaBackend
	p cuda.Ptr
	n int
}

func (b *cudaBuf) size() int { return b.n }
func (b *cudaBuf) release() {
	if b.p != 0 {
		b.c.dev.Free(b.p)
		b.p = 0
	}
}

func (c *cudaBackend) alloc(n int) (gpuBuf, error) {
	p, err := c.dev.Alloc(n)
	if err != nil {
		return nil, err
	}
	return &cudaBuf{c: c, p: p, n: n}, nil
}

func cptr(r gpuRef) cuda.Ptr { return r.buf.(*cudaBuf).p + cuda.Ptr(r.off) }

// write and read panic on failure: a failed copy means a lost device (the driver reports it on every later call).
func (c *cudaBackend) write(dst gpuRef, src []byte) {
	if err := c.dev.Write(cptr(dst), src); err != nil {
		panic(err)
	}
}

func (c *cudaBackend) read(src gpuRef, dst []byte) {
	if err := c.dev.Read(cptr(src), dst); err != nil {
		panic(err)
	}
}

func (c *cudaBackend) mapped(gpuBuf) []byte { return nil }
func (c *cudaBackend) batch() gpuBatch      { return &cudaBatch{c: c} }

type cudaBatch struct {
	c  *cudaBackend
	ls []cuda.Launch
}

func carg(r gpuRef) cuda.Arg { return cuda.Arg{Ptr: cptr(r)} }

func (b *cudaBatch) launch(k string, grid, block [3]uint32, shared int, args ...cuda.Arg) {
	b.ls = append(b.ls, cuda.Launch{K: b.c.ks[k], Args: args, Grid: grid, Block: block, SharedB: shared})
}

func (b *cudaBatch) submit() error {
	err := b.c.dev.Run(b.ls...)
	b.ls = b.ls[:0]
	return err
}

func (b *cudaBatch) weightArg(w *weight) cuda.Arg {
	return cuda.Arg{Ptr: b.c.weights + cuda.Ptr(uintptr(unsafe.Pointer(unsafe.SliceData(w.data)))-b.c.base)}
}

func (b *cudaBatch) matmul(w *weight, x, out gpuRef, n int, _ gpuRef) {
	if n >= matmulTiledMin && w.typ == typeQ8_0 {
		b.launch("matmul_q8_0_f32", [3]uint32{uint32((w.rows + 63) / 64), uint32((n + 63) / 64), 1}, [3]uint32{256, 1, 1}, 0,
			b.weightArg(w), carg(x), carg(out), cuda.Arg{Bytes: u32(uint32(w.rows), uint32(w.cols/32), uint32(w.rowBytes), uint32(w.cols), uint32(n))})
		return
	}
	var k string
	var nb int
	switch w.typ {
	case typeQ4_K:
		k, nb = "matvec_q4k_f32", w.cols/qk
	case typeQ8_0:
		k, nb = "matvec_q8_0_f32", w.cols/32
	default:
		panic("cuda: matmul doesn't support " + typeName(w.typ))
	}
	b.launch(k, [3]uint32{uint32((w.rows + 3) / 4), uint32(n), 1}, [3]uint32{128, 1, 1}, 0,
		b.weightArg(w), carg(x), carg(out), cuda.Arg{Bytes: u32(uint32(w.rows), uint32(nb), uint32(w.rowBytes), uint32(w.cols))})
}

func (b *cudaBatch) rmsnorm(x, w, out gpuRef, rows, n int, eps float64) {
	nt := min(1024, (n+31)&^31)
	b.launch("rmsnorm_f32", [3]uint32{uint32(rows), 1, 1}, [3]uint32{uint32(nt), 1, 1}, 0,
		carg(x), carg(w), carg(out), cuda.Arg{Bytes: u32(uint32(n), math.Float32bits(float32(eps)))})
}

func (b *cudaBatch) rope(x, cs, sn gpuRef, tokens, heads, headDim int) {
	half := headDim / 2
	tw := min(half, 64)
	b.launch("rope_neox_f32", [3]uint32{uint32((half + tw - 1) / tw), uint32(heads), uint32(tokens)}, [3]uint32{uint32(tw), 1, 1}, 0,
		carg(x), carg(cs), carg(sn), cuda.Arg{Bytes: u32(uint32(heads), uint32(headDim))})
}

func (b *cudaBatch) elementwise(k string, n int, args ...cuda.Arg) {
	b.launch(k, [3]uint32{uint32((n + 255) / 256), 1, 1}, [3]uint32{256, 1, 1}, 0, append(args, cuda.Arg{Bytes: u32(uint32(n))})...)
}

func (b *cudaBatch) swiglu(gate, up, out gpuRef, n int) {
	b.elementwise("swiglu_f32", n, carg(gate), carg(up), carg(out))
}
func (b *cudaBatch) add(x, y gpuRef, n int)      { b.elementwise("add_f32", n, carg(x), carg(y)) }
func (b *cudaBatch) copy(dst, src gpuRef, n int) { b.elementwise("copy_f32", n, carg(dst), carg(src)) }

func (b *cudaBatch) storeKV(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int) {
	b.launch("kv_store_f32", [3]uint32{uint32((kvd + 255) / 256), uint32(n), 1}, [3]uint32{256, 1, 1}, 0,
		carg(k), carg(v), carg(kc), carg(vc), cuda.Arg{Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos))})
}

func (b *cudaBatch) attention(a gpuAttention) {
	if a.kvType != typeF32 {
		panic(fmt.Sprintf("cuda: attention over a %s cache isn't supported", kvTypeName(a.kvType)))
	}
	const nt = 256
	if chunk, splits, ok := a.split(); ok {
		args := cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos),
			a.scale(), uint32(chunk), uint32(splits))}
		b.launch("attention_split_f32", [3]uint32{uint32(a.heads), uint32(a.n), uint32(splits)}, [3]uint32{nt, 1, 1},
			4*(a.headDim+chunk+nt/32*a.headDim), carg(a.q), carg(a.k), carg(a.v), carg(a.scratch), carg(a.mask), args)
		b.launch("attention_combine_f32", [3]uint32{uint32(a.heads), uint32(a.n), 1},
			[3]uint32{uint32((a.headDim + 31) &^ 31), 1, 1}, 0, carg(a.scratch), carg(a.out), args)
		return
	}
	b.launch("attention_f32", [3]uint32{uint32(a.heads), uint32(a.n), 1}, [3]uint32{nt, 1, 1}, 4*(a.headDim+nt),
		carg(a.q), carg(a.k), carg(a.v), carg(a.out), carg(a.scratch), carg(a.mask),
		cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos),
			a.scale())})
}
