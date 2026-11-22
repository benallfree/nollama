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
var metalDetKernels = []string{"rmsnorm_det", "swiglu_det", "quant_q8_0_det", "quant_q8_k_det", "matvec_q8_0_det", "matvec_q4k_det", "matmul_q8_0_det", "matvec1_q8_0_det", "attention_det"}

// metalKernels are the kernels metalSource defines.
var metalKernels = []string{"matvec_q4k_f32", "matvec_q8_0_f32", "rmsnorm_f32", "rope_neox_f32", "swiglu_f32", "add_f32", "matmul_q8_0_f32", "matmul_q4k_f32", "matmul_q6k_f32", "matvec_q6k_f32", "matvec_f16_f32", "matmul_f16_f32", "matvec_q4k_pair", "matvec_q8_0_pair", "matvec_q6k_pair", "matvec_f16_pair", "norm_rope_f32", "add_norm_f32", "attention_f32", "attention_tile_f32", "attention_flash_f32", "attention_split_f32", "attention_combine_f32", "kv_store_f32", "kv_store_f16", "copy_f32", "attention_f16", "attention_split_f16", "attention_tile_f16", "attention_flash_f16"}

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

func (m *metalBackend) halfPrompts() bool        { return false }
func (m *metalBackend) int8Prompts() bool        { return false }
func (m *metalBackend) supportsKV(t uint32) bool { return t == typeF32 || t == typeF16 && !m.det }

func (m *metalBackend) supports(t uint32) bool {
	return t == typeQ4_K || t == typeQ8_0 || (t == typeQ6_K || t == typeF16) && !m.det // deterministic mode: no Q6_K/F16 kernels yet
}

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
	// quantized: what act holds (deterministic mode), so matmuls sharing an input (q/k/v, gate/up) quantize it once.
	// Every other operation clears it: it may write that input.
	quantized struct {
		x, act gpuRef
		n      int
		kernel string
	}
}

// wrote forgets the quantized activations: the operation being recorded may change their source.
func (b *metalBatch) wrote() { b.quantized.kernel = "" }

func marg(r gpuRef) metal.Arg { return metal.Arg{Buf: r.buf.(metalBuf).b, Offset: r.off} }

func (b *metalBatch) submit() error {
	b.wrote()
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
	} else if n >= matmulTiledMin {
		b.matmulTiled(w, x, out, n)
	} else {
		b.matvec(w, x, out, n)
	}
}

// matvec is the one-token kernel (a simdgroup per mvRows rows), also usable for small batches.
func (b *metalBatch) matvec(w *weight, x, out gpuRef, n int) { b.matvecPair(w, nil, x, out, n) }

// gateUp fuses the matvecs and swiglu for small batches (same weight types); prompts run the three steps.
func (b *metalBatch) gateUp(wg, wu *weight, x, gate, up, out gpuRef, n int, act gpuRef) {
	if b.m.det || n >= matmulTiledMin || wg.typ != wu.typ || wg.rows != wu.rows || wg.cols != wu.cols {
		b.matmul(wg, x, gate, n, act)
		b.matmul(wu, x, up, n, act)
		b.swiglu(gate, up, out, n*wg.rows)
		return
	}
	b.wrote()
	b.matvecPair(wg, wu, x, out, n)
}

// matvecPair is matvec, or with w2 the pair kernel: silu(w x) * (w2 x).
func (b *metalBatch) matvecPair(w, w2 *weight, x, out gpuRef, n int) {
	var kernel string
	var nb int
	switch w.typ {
	case typeQ4_K:
		kernel, nb = "matvec_q4k_f32", w.cols/qk
	case typeQ8_0:
		kernel, nb = "matvec_q8_0_f32", w.cols/32
	case typeQ6_K:
		kernel, nb = "matvec_q6k_f32", w.cols/qk
	case typeF16:
		kernel, nb = "matvec_f16_f32", w.cols/256
	default:
		panic("metal: matmul doesn't support " + typeName(w.typ))
	}
	nsg := mvSimdgroups
	rows := mvRows // output rows per simdgroup
	w2arg := b.weightArg(w)
	if w2 != nil {
		kernel = kernel[:len(kernel)-len("f32")] + "pair"
		rows /= 2
		w2arg = b.weightArg(w2)
	}
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[kernel],
		Args:    []metal.Arg{b.weightArg(w), marg(x), marg(out), {Bytes: u32(uint32(w.rows), uint32(nb), uint32(w.rowBytes), uint32(w.cols))}, w2arg},
		Writes:  1 << 2,
		Groups:  metal.Size{W: uint((w.rows + nsg*rows - 1) / (nsg * rows)), H: uint(n), D: 1},
		Threads: metal.Size{W: uint(32 * nsg), H: 1, D: 1},
	})
}

// metalAttnOnePass uses the one-pass attention kernel instead of the tiled one (tests compare the two).
var metalAttnOnePass = false

// metalAttnTile uses the scalar tiled prompt kernel instead of the matrix-unit one (other head sizes use it anyway).
var metalAttnTile = false

// mvRows is MV_ROWS in gpu.metal: rows per simdgroup in the matvecs.
const mvRows = 8

// mvSimdgroups is the matvecs' simdgroups per threadgroup: 2 measured best on an M1 Max (more threadgroups for the
// small matrices), 4 close behind, 8 worse.
var mvSimdgroups = 2

// matmulTiled is the 32x32-tile kernel for prompt batches (Q8_0 only so far).
func (b *metalBatch) matmulTiled(w *weight, x, out gpuRef, n int) {
	kernel := map[uint32]string{typeQ8_0: "matmul_q8_0_f32", typeQ4_K: "matmul_q4k_f32", typeQ6_K: "matmul_q6k_f32", typeF16: "matmul_f16_f32"}[w.typ]
	if kernel == "" {
		panic("metal: tiled matmul doesn't support " + typeName(w.typ))
	}
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[kernel],
		Args:    []metal.Arg{b.weightArg(w), marg(x), marg(out), {Bytes: u32(uint32(w.rows), uint32(w.cols/32), uint32(w.rowBytes), uint32(w.cols), uint32(n))}},
		Writes:  1 << 2,
		Groups:  metal.Size{W: uint((w.rows + 63) / 64), H: uint((n + 63) / 64), D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
		Shared:  64 * 68 * 4, // the two 64x32 input tiles, then the 64x64 output tile (rows padded: MM_LD, MM_OLD)
	})
}

// detT is DET_T in gpu_det.metal and gpu_det.cu: tokens per simdgroup (warp) in the deterministic matvecs.
const detT = 8

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
	q := &b.quantized
	if q.kernel != quant || q.x != x || q.act != act || q.n != n {
		b.ds = append(b.ds, metal.Dispatch{
			P:       b.m.ps[quant],
			Args:    []metal.Arg{marg(x), marg(act), {Bytes: u32(uint32(n * nb))}},
			Writes:  1 << 1,
			Groups:  metal.Size{W: uint((n*nb + nsg - 1) / nsg), H: 1, D: 1},
			Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
		})
		q.x, q.act, q.n, q.kernel = x, act, n, quant
	}
	if out.buf == x.buf || out.buf == act.buf {
		b.wrote()
	}
	args := metal.Arg{Bytes: u32(uint32(w.rows), uint32(nb), uint32(w.rowBytes), uint32(nb*abytes), uint32(n))}
	if n >= matmulTiledMin && w.typ == typeQ8_0 { // prompts: the matrix units (exact integer totals per block)
		b.ds = append(b.ds, metal.Dispatch{
			P:       b.m.ps["matmul_q8_0_det"],
			Args:    []metal.Arg{b.weightArg(w), marg(act), marg(out), args},
			Writes:  1 << 2,
			Groups:  metal.Size{W: uint((w.rows + 31) / 32), H: uint((n + 31) / 32), D: 1},
			Threads: metal.Size{W: 128, H: 1, D: 1},
			Shared:  4 * (3072 + 64),
		})
		return
	}
	if n == 1 && w.typ == typeQ8_0 && nb <= 1024 { // decode: coalesced loads
		b.ds = append(b.ds, metal.Dispatch{
			P:       b.m.ps["matvec1_q8_0_det"],
			Args:    []metal.Arg{b.weightArg(w), marg(act), marg(out), args},
			Writes:  1 << 2,
			Groups:  metal.Size{W: uint((w.rows + nsg - 1) / nsg), H: 1, D: 1},
			Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
			Shared:  nsg * 2 * nb * 4,
		})
		return
	}
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[kernel],
		Args:    []metal.Arg{b.weightArg(w), marg(act), marg(out), args},
		Writes:  1 << 2,
		Groups:  metal.Size{W: uint((w.rows + nsg - 1) / nsg), H: uint((n + detT - 1) / detT), D: 1},
		Threads: metal.Size{W: 32 * nsg, H: 1, D: 1},
		Shared:  nsg * 64 * min(detT, n) * 4,
	})
}

func (b *metalBatch) rmsnorm(x, w, out gpuRef, rows, n int, eps float64) {
	b.wrote()
	if b.m.det {
		b.ds = append(b.ds, metal.Dispatch{
			P:       b.m.ps["rmsnorm_det"],
			Args:    []metal.Arg{marg(x), marg(w), marg(out), {Bytes: u32(uint32(n), math.Float32bits(float32(eps)))}},
			Writes:  1 << 2,
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
		Writes:  1 << 2,
		Groups:  metal.Size{W: uint(rows), H: 1, D: 1},
		Threads: metal.Size{W: uint(nt), H: 1, D: 1},
		Shared:  32 * 4,
	})
}

func (b *metalBatch) rope(x, cs, sn gpuRef, tokens, heads, headDim int) {
	b.wrote()
	half := headDim / 2
	tw := min(half, 64)
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps["rope_neox_f32"],
		Args:    []metal.Arg{marg(x), marg(cs), marg(sn), {Bytes: u32(uint32(heads), uint32(headDim))}},
		Writes:  1 << 0,
		Groups:  metal.Size{W: uint((half + tw - 1) / tw), H: uint(heads), D: uint(tokens)},
		Threads: metal.Size{W: uint(tw), H: 1, D: 1},
	})
}

func (b *metalBatch) normRope(x, w, cs, sn gpuRef, tokens, heads, headDim int, eps float64) {
	if b.m.det { // the separate kernels: the CPU's order
		b.rmsnorm(x, w, x, tokens*heads, headDim, eps)
		b.rope(x, cs, sn, tokens, heads, headDim)
		return
	}
	b.normRopeKV(x, w, cs, sn, gpuRef{}, tokens, heads, headDim, 0, 0, eps)
}

func (b *metalBatch) normRopeKV(x, w, cs, sn, kc gpuRef, tokens, heads, headDim, pos, ctxP int, eps float64) {
	if b.m.det {
		panic("metal: normRopeKV in deterministic mode")
	}
	b.wrote()
	store, writes := uint32(0), uint32(1<<0)
	kcArg := marg(x)
	if kc.buf != nil {
		store, writes, kcArg = 1, 1<<0|1<<5, marg(kc)
	}
	b.ds = append(b.ds, metal.Dispatch{
		P: b.m.ps["norm_rope_f32"],
		Args: []metal.Arg{marg(x), marg(w), marg(cs), marg(sn),
			{Bytes: u32(uint32(heads), uint32(headDim), math.Float32bits(float32(eps)), store, uint32(ctxP), uint32(pos))}, kcArg},
		Writes:  writes,
		Groups:  metal.Size{W: uint(tokens * heads), H: 1, D: 1},
		Threads: metal.Size{W: uint(headDim / 2), H: 1, D: 1},
		Shared:  32 * 4,
	})
}

func (b *metalBatch) addNorm(x, y, w, out gpuRef, rows, n int, eps float64) {
	if b.m.det {
		b.add(x, y, rows*n)
		b.rmsnorm(x, w, out, rows, n, eps)
		return
	}
	b.wrote()
	p := b.m.ps["add_norm_f32"]
	b.ds = append(b.ds, metal.Dispatch{
		P:       p,
		Args:    []metal.Arg{marg(x), marg(y), marg(w), marg(out), {Bytes: u32(uint32(n), math.Float32bits(float32(eps)))}},
		Writes:  1<<0 | 1<<3,
		Groups:  metal.Size{W: uint(rows), H: 1, D: 1},
		Threads: metal.Size{W: uint(min(p.MaxThreads, 1024, (n+31)&^31)), H: 1, D: 1},
		Shared:  32 * 4,
	})
}

// elementwise is a 1-D dispatch over n elements; writes marks the args it writes (metal.Dispatch.Writes).
func (b *metalBatch) elementwise(kernel string, n int, writes uint32, args ...metal.Arg) {
	b.wrote()
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps[kernel],
		Writes:  writes,
		Args:    append(args, metal.Arg{Bytes: u32(uint32(n))}),
		Groups:  metal.Size{W: uint((n + 255) / 256), H: 1, D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
	})
}

func (b *metalBatch) swiglu(gate, up, out gpuRef, n int) {
	if b.m.det {
		b.elementwise("swiglu_det", n, 1<<2, marg(gate), marg(up), marg(out))
		return
	}
	b.elementwise("swiglu_f32", n, 1<<2, marg(gate), marg(up), marg(out))
}

func (b *metalBatch) add(x, y gpuRef, n int) { b.elementwise("add_f32", n, 1<<0, marg(x), marg(y)) }

func (b *metalBatch) copy(dst, src gpuRef, n int) {
	b.elementwise("copy_f32", n, 1<<0, marg(dst), marg(src))
}

func (b *metalBatch) attention(a gpuAttention) {
	b.wrote()
	kv := map[uint32]string{typeF32: "_f32", typeF16: "_f16"}[a.kvType]
	if kv == "" || kv != "_f32" && b.m.det {
		panic(fmt.Sprintf("metal: attention over a %s cache isn't supported", kvTypeName(a.kvType)))
	}
	if b.m.det { // the CPU's order: one pass per (token, head), no split (its partial sums would be added in another order)
		b.ds = append(b.ds, metal.Dispatch{
			P: b.m.ps["attention_det"],
			Args: []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.out), marg(a.scratch), marg(a.mask),
				{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale())}},
			Writes:  1<<3 | 1<<4,
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
			P:       b.m.ps["attention_split"+kv],
			Args:    []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.scratch), marg(a.mask), args},
			Writes:  1 << 3,
			Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: uint(splits)},
			Threads: metal.Size{W: nt, H: 1, D: 1},
			Shared:  4 * (a.headDim + chunk + nt/32*a.headDim),
		}, metal.Dispatch{
			P:       b.m.ps["attention_combine_f32"],
			Args:    []metal.Arg{marg(a.scratch), marg(a.out), args},
			Writes:  1 << 1,
			Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: 1},
			Threads: metal.Size{W: uint((a.headDim + 31) &^ 31), H: 1, D: 1},
		})
		return
	}
	if a.n > attnSplitTokens && a.headDim == 128 && !metalAttnOnePass && !metalAttnTile { // prompts on the matrix units
		b.ds = append(b.ds, metal.Dispatch{
			P: b.m.ps["attention_flash"+kv],
			Args: []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.out), marg(a.mask),
				{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale(), uint32(a.n))}},
			Writes:  1 << 3,
			Groups:  metal.Size{W: uint(a.heads), H: uint((a.n + 31) / 32), D: 1},
			Threads: metal.Size{W: 128, H: 1, D: 1},
		})
		return
	}
	if a.n > attnSplitTokens && a.headDim <= 128 && !metalAttnOnePass { // prompts; one-token steps keep the one-pass kernel
		const qt = 16 // ATT_QT
		b.ds = append(b.ds, metal.Dispatch{
			P: b.m.ps["attention_tile"+kv],
			Args: []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.out), marg(a.mask),
				{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale(), uint32(a.n))}},
			Writes:  1 << 3,
			Groups:  metal.Size{W: uint(a.heads), H: uint((a.n + qt - 1) / qt), D: 1},
			Threads: metal.Size{W: 256, H: 1, D: 1},
		})
		return
	}
	b.ds = append(b.ds, metal.Dispatch{
		P: b.m.ps["attention"+kv],
		Args: []metal.Arg{marg(a.q), marg(a.k), marg(a.v), marg(a.out), marg(a.scratch), marg(a.mask),
			{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale())}},
		Writes:  1<<3 | 1<<4,
		Groups:  metal.Size{W: uint(a.heads), H: uint(a.n), D: 1},
		Threads: metal.Size{W: nt, H: 1, D: 1},
		Shared:  4 * (a.headDim + nt),
	})
}

func (b *metalBatch) storeKV16(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int) {
	b.wrote()
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps["kv_store_f16"],
		Args:    []metal.Arg{marg(k), marg(v), marg(kc), marg(vc), {Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos))}},
		Writes:  1<<2 | 1<<3,
		Groups:  metal.Size{W: uint((kvd + 255) / 256), H: uint(n), D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
	})
}

func (b *metalBatch) storeKV(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int) {
	b.wrote()
	b.ds = append(b.ds, metal.Dispatch{
		P:       b.m.ps["kv_store_f32"],
		Args:    []metal.Arg{marg(k), marg(v), marg(kc), marg(vc), {Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos))}},
		Writes:  1<<2 | 1<<3,
		Groups:  metal.Size{W: uint((kvd + 255) / 256), H: uint(n), D: 1},
		Threads: metal.Size{W: 256, H: 1, D: 1},
	})
}
