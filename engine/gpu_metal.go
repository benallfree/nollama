package engine

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/metal"
)

//go:embed gpu.metal
var metalSource string

//go:embed gpu_det.metal
var metalDetSource string

// metalDetKernels are the kernels metalDetSource adds for deterministic mode.
var metalDetKernels = []string{"rmsnorm_det", "swiglu_det", "quant_q8_0_det", "quant_q8_k_det", "matvec_q8_0_det", "matvec_q4k_det", "attention_det"}

// metalKernels are the kernels metalSource defines.
var metalKernels = []string{"matvec_q4k_f32", "matvec_q8_0_f32", "rmsnorm_f32", "rope_neox_f32", "swiglu_f32", "add_f32", "matmul_q8_0_f32", "attention_f32", "attention_split_f32", "attention_combine_f32", "kv_store_f32", "copy_f32"}

// metalBackend is the gpuBackend for Apple GPUs (unified memory only).
type metalBackend struct {
	dev     *metal.Device
	ps      map[string]*metal.Pipeline
	weights *metal.Buffer // the whole GGUF file, zero-copy
	base    uintptr       // address of the file's first byte
	det     bool          // the deterministic kernels (gpu_det.metal): the CPU's reference results, bit for bit
}

func openMetal(det bool) (gpuBackend, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if !dev.Unified {
		dev.Close()
		return nil, errors.New("metal: GPUs without unified memory aren't supported")
	}
	var ps map[string]*metal.Pipeline
	if det { // fast math off; gpu_det.metal comes first, so its no-contraction pragma covers the shared kernels too
		ps, err = dev.CompilePrecise(metalDetSource+"\n"+metalSource, append(metalDetKernels, metalKernels...)...)
	} else {
		ps, err = dev.Compile(metalSource, metalKernels...)
	}
	if err != nil {
		dev.Close()
		return nil, err
	}
	return &metalBackend{dev: dev, ps: ps, det: det}, nil
}

func (m *metalBackend) name() string {
	if m.det {
		return "metal (" + m.dev.Name + ", deterministic)"
	}
	return "metal (" + m.dev.Name + ")"
}
func (m *metalBackend) unified() bool { return true }

func (m *metalBackend) supports(t uint32) bool { return t == typeQ4_K || t == typeQ8_0 }

func (m *metalBackend) loadWeights(file []byte) error {
	wb, err := m.dev.Wrap(file)
	if err != nil {
		return err
	}
	m.weights, m.base = wb, uintptr(unsafe.Pointer(unsafe.SliceData(file)))
	return nil
}

func (m *metalBackend) close() {
	if m.weights != nil {
		m.weights.Release()
	}
	m.dev.Close()
}

type metalBuf struct{ b *metal.Buffer }

func (b metalBuf) size() int { return len(b.b.Bytes()) }
func (b metalBuf) release()  { b.b.Release() }

func (m *metalBackend) alloc(n int) (gpuBuf, error) {
	b, err := m.dev.NewBuffer(n)
	if err != nil {
		return nil, err
	}
	return metalBuf{b}, nil
}

// Shared memory: write and read are copies the CPU does itself.
func (m *metalBackend) write(dst gpuRef, src []byte) {
	copy(dst.buf.(metalBuf).b.Bytes()[dst.off:], src)
}
func (m *metalBackend) read(src gpuRef, dst []byte) {
	copy(dst, src.buf.(metalBuf).b.Bytes()[src.off:])
}

func (m *metalBackend) mapped(buf gpuBuf) []byte { return buf.(metalBuf).b.Bytes() }

func (m *metalBackend) batch() gpuBatch { return &metalBatch{m: m} }

// metalBatch records dispatches and runs them in one command buffer.
type metalBatch struct {
	m  *metalBackend
	ds []metal.Dispatch
}

func marg(r gpuRef) metal.Arg { return metal.Arg{Buf: r.buf.(metalBuf).b, Offset: r.off} }

func (b *metalBatch) submit() error {
	err := b.m.dev.Run(b.ds...)
	b.ds = b.ds[:0]
	return err
}

// weightArg is w's bytes as a kernel argument: an offset into the wrapped file.
func (b *metalBatch) weightArg(w *weight) metal.Arg {
	return metal.Arg{Buf: b.m.weights, Offset: int(uintptr(unsafe.Pointer(unsafe.SliceData(w.data))) - b.m.base)}
}

// matmulTiledMin is the token count from which matmul uses the tiled kernel.
const matmulTiledMin = 8

func (b *metalBatch) matmul(w *weight, x, out gpuRef, n int, act gpuRef) {
	if b.m.det {
		b.matmulDet(w, x, out, n, act)
	} else if n >= matmulTiledMin && w.typ == typeQ8_0 {
		b.matmulTiled(w, x, out, n)
	} else {
		b.matvec(w, x, out, n)
	}
}

// matvec is the one-token kernel (one simdgroup per row), also usable for small batches.
func (b *metalBatch) matvec(w *weight, x, out gpuRef, n int) {
	var kernel string
	var nb int
	switch w.typ {
	case typeQ4_K:
		kernel, nb = "matvec_q4k_f32", w.cols/qk
	case typeQ8_0:
		kernel, nb = "matvec_q8_0_f32", w.cols/32
	default:
		panic("metal: matmul doesn't support " + typeName(w.typ))
	}
	const nsg = 4 // rows (simdgroups) per threadgroup
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[kernel],
		Args:    []metal.Arg{b.weightArg(w), marg(x), marg(out), {Bytes: u32(uint32(w.rows), uint32(nb), uint32(w.rowBytes), uint32(w.cols))}},
		Groups:  metal.Size{W: uint((w.rows + nsg - 1) / nsg), H: uint(n), D: 1},
		Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
	})
}

// matmulTiled is the 32x32-tile kernel for prompt batches (Q8_0 only so far).
func (b *metalBatch) matmulTiled(w *weight, x, out gpuRef, n int) {
	if w.typ != typeQ8_0 {
		panic("metal: tiled matmul doesn't support " + typeName(w.typ))
	}
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps["matmul_q8_0_f32"],
		Args:    []metal.Arg{b.weightArg(w), marg(x), marg(out), {Bytes: u32(uint32(w.rows), uint32(w.cols/32), uint32(w.rowBytes), uint32(w.cols), uint32(n))}},
		Groups:  metal.Size{W: uint((w.rows + 31) / 32), H: uint((n + 31) / 32), D: 1},
		Threads: metal.Size{W: 128, H: 1, D: 1},
		Shared:  2 * 1024 * 4,
	})
}

// matmulDet quantizes x into act as the CPU does (Q8_0 or Q8_K blocks, by weight type), then runs the integer
// matvec that follows the CPU's dot product order, for each token.
func (b *metalBatch) matmulDet(w *weight, x, out gpuRef, n int, act gpuRef) {
	var quant, kernel string
	var bs, abytes int
	switch w.typ {
	case typeQ8_0:
		quant, kernel, bs, abytes = "quant_q8_0_det", "matvec_q8_0_det", 32, q80aBytes
	case typeQ4_K:
		quant, kernel, bs, abytes = "quant_q8_k_det", "matvec_q4k_det", qk, q8kBytes
	default:
		panic("metal: deterministic matmul doesn't support " + typeName(w.typ))
	}
	nb := w.cols / bs
	const nsg = 4 // blocks or rows (simdgroups) per threadgroup
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[quant],
		Args:    []metal.Arg{marg(x), marg(act), {Bytes: u32(uint32(n * nb))}},
		Groups:  metal.Size{W: uint((n*nb + nsg - 1) / nsg), H: 1, D: 1},
		Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
	}, metal.Dispatch{
		P:       b.m.ps[kernel],
		Args:    []metal.Arg{b.weightArg(w), marg(act), marg(out), {Bytes: u32(uint32(w.rows), uint32(nb), uint32(w.rowBytes), uint32(nb*abytes))}},
		Groups:  metal.Size{W: uint((w.rows + nsg - 1) / nsg), H: uint(n), D: 1},
		Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
	})
}

func (b *metalBatch) rmsnorm(x, w, out gpuRef, rows, n int, eps float64) {
	if b.m.det {
		b.ds = append(b.ds, metal.Dispatch{
			P:       b.m.ps["rmsnorm_det"],
			Args:    []metal.Arg{marg(x), marg(w), marg(out), {Bytes: u32(uint32(n), math.Float32bits(float32(eps)))}},
			Groups:  metal.Size{W: uint(rows), H: 1, D: 1},
			Threads: metal.Size{W: 32, H: 1, D: 1},
		})
		return
	}
	p := b.m.ps["rmsnorm_f32"]
	nt := min(p.MaxThreads, 1024, (n+31)&^31)
	b.ds = append(b.ds, metal.Dispatch{
		P:       p,
		Args:    []metal.Arg{marg(x), marg(w), marg(out), {Bytes: u32(uint32(n), math.Float32bits(float32(eps)))}},
		Groups:  metal.Size{W: uint(rows), H: 1, D: 1},
		Threads: metal.Size{W: uint(nt), H: 1, D: 1},
		Shared:  32 * 4,
	})
}

func (b *metalBatch) rope(x, cs, sn gpuRef, tokens, heads, headDim int) {
	half := headDim / 2
	tw := min(half, 64)
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps["rope_neox_f32"],
		Args:    []metal.Arg{marg(x), marg(cs), marg(sn), {Bytes: u32(uint32(heads), uint32(headDim))}},
		Groups:  metal.Size{W: uint((half + tw - 1) / tw), H: uint(heads), D: uint(tokens)},
		Threads: metal.Size{W: uint(tw), H: 1, D: 1},
	})
}

// elementwise is a 1-D dispatch over n elements.
func (b *metalBatch) elementwise(kernel string, n int, args ...metal.Arg) {
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[kernel],
		Args:    append(args, metal.Arg{Bytes: u32(uint32(n))}),
		Groups:  metal.Size{W: uint((n + 255) / 256), H: 1, D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
	})
}

func (b *metalBatch) swiglu(gate, up, out gpuRef, n int) {
	if b.m.det {
		b.elementwise("swiglu_det", n, marg(gate), marg(up), marg(out))
		return
	}
	b.elementwise("swiglu_f32", n, marg(gate), marg(up), marg(out))
}

func (b *metalBatch) add(x, y gpuRef, n int) { b.elementwise("add_f32", n, marg(x), marg(y)) }

func (b *metalBatch) copy(dst, src gpuRef, n int) { b.elementwise("copy_f32", n, marg(dst), marg(src)) }

func (b *metalBatch) attention(a gpuAttention) {
	if a.kvType != typeF32 {
		panic(fmt.Sprintf("metal: attention over a %s cache isn't supported", kvTypeName(a.kvType)))
	}
	if b.m.det { // the CPU's order: one pass per (token, head), no split (its partial sums would be added in another order)
		b.ds = append(b.ds, metal.Dispatch{
			P: b.m.ps["attention_det"],
			Args: []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.out), marg(a.scratch), marg(a.mask),
				{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale())}},
			Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: 1},
			Threads: metal.Size{W: 128, H: 1, D: 1},
			Shared:  4 * (a.headDim + 4),
		})
		return
	}
	const nt = 256
	if chunk, splits, ok := a.split(); ok {
		args := metal.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos),
			a.scale(), uint32(chunk), uint32(splits))}
		b.ds = append(b.ds, metal.Dispatch{
			P:       b.m.ps["attention_split_f32"],
			Args:    []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.scratch), marg(a.mask), args},
			Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: uint(splits)},
			Threads: metal.Size{W: nt, H: 1, D: 1},
			Shared:  4 * (a.headDim + chunk + nt/32*a.headDim),
		}, metal.Dispatch{
			P:       b.m.ps["attention_combine_f32"],
			Args:    []metal.Arg{marg(a.scratch), marg(a.out), args},
			Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: 1},
			Threads: metal.Size{W: uint((a.headDim + 31) &^ 31), H: 1, D: 1},
		})
		return
	}
	b.ds = append(b.ds, metal.Dispatch{
		P: b.m.ps["attention_f32"],
		Args: []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.out), marg(a.scratch), marg(a.mask),
			{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale())}},
		Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: 1},
		Threads: metal.Size{W: nt, H: 1, D: 1},
		Shared:  4 * (a.headDim + nt),
	})
}

func (b *metalBatch) storeKV(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int) {
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps["kv_store_f32"],
		Args:    []metal.Arg{marg(k), marg(v), marg(kc), marg(vc), {Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos))}},
		Groups:  metal.Size{W: uint((kvd + 255) / 256), H: uint(n), D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
	})
}
