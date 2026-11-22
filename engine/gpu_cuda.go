package engine

//go:generate sh gen_ptx.sh

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/benallfree/nollama/engine/internal/cuda"
)

// gpu_cuda.ptx is gpu.cu compiled for compute capability 7.5 (T4 and newer: A10, A100, L4, L40S, H100, RTX 20xx+).
//
//go:embed gpu_cuda.ptx
var cudaPTX string

// gpu_cuda_det.ptx is gpu_det.cu (which includes gpu.cu) compiled for IEEE float32 arithmetic: deterministic mode.
//
//go:embed gpu_cuda_det.ptx
var cudaDetPTX string

// gpu_cuda_mma.ptx is gpu_mma.cu compiled for compute capability 8.0 (A100, A10, A16, L4, L40S, H100, RTX 30xx+): the
// int8 tensor-core prompt matmuls, loaded on GPUs that have them.
//
//go:embed gpu_cuda_mma.ptx
var cudaMMAPTX string

var cudaMMAKernels = []string{"quant_act_mma", "matmul_q4k_mmq"}

// cudaNoMMQ keeps prompts on the half-precision tensor-core kernels even where the int8 ones load (tests compare).
var cudaNoMMQ = false

// cudaDetKernels are the kernels gpu_det.cu adds for deterministic mode.
var cudaDetKernels = []string{"rmsnorm_det", "swiglu_det", "quant_q8_0_det", "quant_q8_k_det", "matvec_q8_0_det",
	"matvec1_q8_0_det", "matmul_q8_0_det", "matvec_q4k_det", "attention_det"}

var cudaKernels = []string{"matvec_q4k_f32", "matvec_q8_0_f32", "matvec_q6k_f32", "matvec_f16_f32",
	"matvec_q4k_pair", "matvec_q8_0_pair", "matvec_q6k_pair", "matvec_f16_pair",
	"matmul_q8_0_f32", "matmul_q4k_f32", "matmul_q6k_f32", "matmul_f16_f32",
	"matmul_q8_0_tc", "matmul_q4k_tc", "matmul_q6k_tc", "matmul_f16_tc", "rmsnorm_f32", "rope_neox_f32",
	"swiglu_f32", "add_f32", "norm_rope_f32", "add_norm_f32", "copy_f32", "kv_store_f32", "kv_store_f16", "attention_f32", "attention_f16", "attention_split_f16", "attention_tile_f16", "attention_flash_f16", "attention_q8", "attention_q4", "attention_split_q8", "attention_split_q4", "attention_tile_q8", "attention_tile_q4", "attention_flash_q8", "attention_flash_q4", "kv_store_qk", "kv_store_qv", "copy_rows_u8", "attention_flash_f32",
	"attention_split_f32", "attention_combine_f32", "attention_tile_f32"}

// cudaBackend is the gpuBackend for NVIDIA GPUs, through the driver API (no toolkit needed).
type cudaBackend struct {
	dev     *cuda.Device
	mod     *cuda.Module
	mmaMod  *cuda.Module // the int8 prompt kernels (compute 8.0+, not deterministic mode), or nil
	ks      map[string]*cuda.Kernel
	weights cuda.Ptr // the model file, uploaded once
	base    uintptr  // address of the file's first byte in host memory
	det     bool     // the deterministic kernels (gpu_det.cu): the CPU's reference results, bit for bit
}

func openCUDA(det bool) (gpuBackend, error) {
	dev, err := cuda.Open(0)
	if err != nil {
		return nil, err
	}
	if dev.Major < 7 || dev.Major == 7 && dev.Minor < 5 {
		dev.Close()
		return nil, fmt.Errorf("cuda: %s is compute %d.%d; nollama needs 7.5 or newer", dev.Name, dev.Major, dev.Minor)
	}
	ptx, names := cudaPTX, cudaKernels
	if det {
		ptx, names = cudaDetPTX, append(append([]string(nil), cudaKernels...), cudaDetKernels...)
	}
	mod, ks, err := dev.Load(ptx, names...)
	if err != nil {
		dev.Close()
		return nil, err
	}
	c := &cudaBackend{dev: dev, mod: mod, ks: ks, det: det}
	if !det && dev.Major >= 8 {
		if mm, mks, err := dev.Load(cudaMMAPTX, cudaMMAKernels...); err == nil { // without it, prompts use the half kernels
			c.mmaMod = mm
			for k, v := range mks {
				ks[k] = v
			}
		}
	}
	return c, nil
}

func (c *cudaBackend) name() string {
	det := ""
	if c.det {
		det = ", deterministic"
	}
	return fmt.Sprintf("cuda (%s, %d GB, compute %d.%d%s)", c.dev.Name, c.dev.Memory>>30, c.dev.Major, c.dev.Minor, det)
}
func (c *cudaBackend) unified() bool     { return false }
func (c *cudaBackend) halfPrompts() bool { return !c.det && !cudaPromptF32 }

// supportsKV: f32, and the compact caches (f16, q8_0, q4_0) outside deterministic mode (checked on an NVIDIA A16).
func (c *cudaBackend) supportsKV(t uint32) bool {
	return t == typeF32 || (t == typeF16 || t == typeQ8_0 || t == typeQ4_0) && !c.det
}

func (c *cudaBackend) int8Prompts() bool { return c.mmaMod != nil && !cudaNoMMQ && !cudaPromptF32 }

func (c *cudaBackend) supports(t uint32) bool {
	return t == typeQ4_K || t == typeQ8_0 || (t == typeQ6_K || t == typeF16) && !c.det // deterministic mode: no Q6_K/F16 kernels yet
}

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
	if c.mmaMod != nil {
		c.mmaMod.Unload()
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
	// quantized: what act holds (deterministic mode), so matmuls sharing an input (q/k/v, gate/up) quantize it once.
	quantized struct {
		x, act gpuRef
		n      int
		kernel string
	}
}

// wrote forgets the quantized activations: the operation being recorded may change their source.
func (b *cudaBatch) wrote() { b.quantized.kernel = "" }

func carg(r gpuRef) cuda.Arg { return cuda.Arg{Ptr: cptr(r)} }

func (b *cudaBatch) launch(k string, grid, block [3]uint32, shared int, args ...cuda.Arg) {
	b.ls = append(b.ls, cuda.Launch{K: b.c.ks[k], Args: args, Grid: grid, Block: block, SharedB: shared})
}

func (b *cudaBatch) submit() error {
	b.wrote()
	err := b.c.dev.Run(b.ls...)
	b.ls = b.ls[:0]
	return err
}

func (b *cudaBatch) weightArg(w *weight) cuda.Arg {
	return cuda.Arg{Ptr: b.c.weights + cuda.Ptr(uintptr(unsafe.Pointer(unsafe.SliceData(w.data)))-b.c.base)}
}

func (b *cudaBatch) matmul(w *weight, x, out gpuRef, n int, act gpuRef) {
	if b.c.det {
		b.matmulDet(w, x, out, n, act)
		return
	}
	k, ok := cudaMatmuls[w.typ]
	if !ok {
		panic("cuda: matmul doesn't support " + typeName(w.typ))
	}
	if out.buf == b.quantized.x.buf || out.buf == b.quantized.act.buf {
		b.wrote()
	}
	if n >= matmulTiledMin && b.mmq(w, x, out, n, act) {
		return
	}
	if n >= matmulTiledMin {
		tiled, rowsPer, tokensPer := k.tc, 128, 64
		if cudaPromptF32 || w.cols%64 != 0 {
			tiled, rowsPer, tokensPer = k.tiled, 64, 64
		}
		b.launch(tiled, [3]uint32{uint32((w.rows + rowsPer - 1) / rowsPer), uint32((n + tokensPer - 1) / tokensPer), 1}, [3]uint32{256, 1, 1}, 0,
			b.weightArg(w), carg(x), carg(out), cuda.Arg{Bytes: u32(uint32(w.rows), uint32(w.cols/32), uint32(w.rowBytes), uint32(w.cols), uint32(n))})
		return
	}
	const rows = 4 * cudaMVRows // 4 warps per block
	b.launch(k.matvec, [3]uint32{uint32((w.rows + rows - 1) / rows), uint32(n), 1}, [3]uint32{128, 1, 1}, 0,
		b.weightArg(w), carg(x), carg(out), cuda.Arg{Bytes: u32(uint32(w.rows), uint32(w.cols/k.step), uint32(w.rowBytes), uint32(w.cols))})
}

// mmq runs a prompt matmul on the int8 tensor cores when the GPU has them and w is Q4_K: x is quantized into
// act (once for matmuls sharing it), then multiplied with the weights' quants. It reports false to leave the matmul
// to the half-precision kernels.
func (b *cudaBatch) mmq(w *weight, x, out gpuRef, n int, act gpuRef) bool {
	// Q8_0 measured no faster this way on an A16; Q6_K (16-element scales, so k16 MMAs) slower.
	kernel := map[uint32]string{typeQ4_K: "matmul_q4k_mmq"}[w.typ]
	qbytes := n * w.cols // int8 quants, then a (scale, scaled sum) pair per 32
	if b.c.mmaMod == nil || cudaNoMMQ || cudaPromptF32 || kernel == "" || w.cols%128 != 0 || act.buf == nil ||
		act.buf.size()-act.off < qbytes+qbytes/32*8 {
		return false
	}
	scales := act.at(qbytes)
	q := &b.quantized
	if q.kernel != "quant_act_mma" || q.x != x || q.act != act || q.n != n {
		nblk := qbytes / 32
		b.launch("quant_act_mma", [3]uint32{uint32((nblk + 31) / 32), 1, 1}, [3]uint32{256, 1, 1}, 0,
			carg(x), carg(act), carg(scales), cuda.Arg{Bytes: u32(uint32(nblk))})
		q.x, q.act, q.n, q.kernel = x, act, n, "quant_act_mma"
	}
	b.launch(kernel, [3]uint32{uint32((w.rows + 127) / 128), uint32((n + 63) / 64), 1}, [3]uint32{256, 1, 1}, 0,
		b.weightArg(w), carg(act), carg(scales), carg(out),
		cuda.Arg{Bytes: u32(uint32(w.rows), uint32(w.cols/32), uint32(w.rowBytes), uint32(w.cols), uint32(n))})
	return true
}

// cudaMatmuls are each weight type's one-token and tiled kernels, and the elements per one-token step.
var cudaMatmuls = map[uint32]struct {
	matvec, tiled, tc string // tc: tensor cores (half inputs, f32 sums)
	step              int
}{
	typeQ8_0: {"matvec_q8_0_f32", "matmul_q8_0_f32", "matmul_q8_0_tc", 32},
	typeQ4_K: {"matvec_q4k_f32", "matmul_q4k_f32", "matmul_q4k_tc", qk},
	typeQ6_K: {"matvec_q6k_f32", "matmul_q6k_f32", "matmul_q6k_tc", qk},
	typeF16:  {"matvec_f16_f32", "matmul_f16_f32", "matmul_f16_tc", 256},
}

// cudaAttnOnePass uses the one-pass attention kernel instead of the tiled one (tests compare the two).
var cudaAttnOnePass = false

// cudaAttnTile uses the scalar tiled prompt kernel instead of the tensor-core one (other head sizes use it anyway).
var cudaAttnTile = false

// cudaPromptF32 runs prompts with the plain f32 kernels instead of the tensor cores (tests compare the two).
var cudaPromptF32 = false

// cudaMVRows is MV_ROWS in gpu.cu: rows per warp in the matvecs.
const cudaMVRows = 8

// matmulDet quantizes x into act as the CPU does (once for matmuls sharing x), then runs the integer dot products in
// the CPU's order (gpu_det.cu).
func (b *cudaBatch) matmulDet(w *weight, x, out gpuRef, n int, act gpuRef) {
	var quant, kernel string
	var bs, abytes int
	switch w.typ {
	case typeQ8_0:
		quant, kernel, bs, abytes = "quant_q8_0_det", "matvec_q8_0_det", 32, q80aBytes
	case typeQ4_K:
		quant, kernel, bs, abytes = "quant_q8_k_det", "matvec_q4k_det", qk, q8kBytes
	default:
		panic("cuda: deterministic matmul doesn't support " + typeName(w.typ))
	}
	nb := w.cols / bs
	const nsg = 4 // blocks or rows (warps) per thread block
	q := &b.quantized
	if q.kernel != quant || q.x != x || q.act != act || q.n != n {
		b.launch(quant, [3]uint32{uint32((n*nb + nsg - 1) / nsg), 1, 1}, [3]uint32{32 * nsg, 1, 1}, 0,
			carg(x), carg(act), cuda.Arg{Bytes: u32(uint32(n * nb))})
		q.x, q.act, q.n, q.kernel = x, act, n, quant
	}
	if out.buf == x.buf || out.buf == act.buf {
		b.wrote()
	}
	args := cuda.Arg{Bytes: u32(uint32(w.rows), uint32(nb), uint32(w.rowBytes), uint32(nb*abytes), uint32(n))}
	if n >= matmulTiledMin && w.typ == typeQ8_0 { // prompts: 64x64 tiles
		b.launch("matmul_q8_0_det", [3]uint32{uint32((w.rows + 63) / 64), uint32((n + 63) / 64), 1}, [3]uint32{256, 1, 1}, 0,
			b.weightArg(w), carg(act), carg(out), args)
		return
	}
	if n == 1 && w.typ == typeQ8_0 && nb <= 1024 { // decode: coalesced loads
		b.launch("matvec1_q8_0_det", [3]uint32{uint32((w.rows + nsg - 1) / nsg), 1, 1}, [3]uint32{32 * nsg, 1, 1}, nsg*2*nb*4,
			b.weightArg(w), carg(act), carg(out), args)
		return
	}
	b.launch(kernel, [3]uint32{uint32((w.rows + nsg - 1) / nsg), uint32((n + detT - 1) / detT), 1}, [3]uint32{32 * nsg, 1, 1},
		nsg*64*min(detT, n)*4, b.weightArg(w), carg(act), carg(out), args)
}

func (b *cudaBatch) rmsnorm(x, w, out gpuRef, rows, n int, eps float64) {
	b.wrote()
	if b.c.det {
		b.launch("rmsnorm_det", [3]uint32{uint32(rows), 1, 1}, [3]uint32{32, 1, 1}, 0,
			carg(x), carg(w), carg(out), cuda.Arg{Bytes: u32(uint32(n), math.Float32bits(float32(eps)))})
		return
	}
	nt := min(1024, (n+31)&^31)
	b.launch("rmsnorm_f32", [3]uint32{uint32(rows), 1, 1}, [3]uint32{uint32(nt), 1, 1}, 0,
		carg(x), carg(w), carg(out), cuda.Arg{Bytes: u32(uint32(n), math.Float32bits(float32(eps)))})
}

func (b *cudaBatch) rope(x, cs, sn gpuRef, tokens, heads, headDim int) {
	b.wrote()
	half := headDim / 2
	tw := min(half, 64)
	b.launch("rope_neox_f32", [3]uint32{uint32((half + tw - 1) / tw), uint32(heads), uint32(tokens)}, [3]uint32{uint32(tw), 1, 1}, 0,
		carg(x), carg(cs), carg(sn), cuda.Arg{Bytes: u32(uint32(heads), uint32(headDim))})
}

func (b *cudaBatch) elementwise(k string, n int, args ...cuda.Arg) {
	b.launch(k, [3]uint32{uint32((n + 255) / 256), 1, 1}, [3]uint32{256, 1, 1}, 0, append(args, cuda.Arg{Bytes: u32(uint32(n))})...)
}

func (b *cudaBatch) swiglu(gate, up, out gpuRef, n int) {
	b.wrote()
	if b.c.det {
		b.elementwise("swiglu_det", n, carg(gate), carg(up), carg(out))
		return
	}
	b.elementwise("swiglu_f32", n, carg(gate), carg(up), carg(out))
}
func (b *cudaBatch) normRope(x, w, cs, sn gpuRef, tokens, heads, headDim int, eps float64) {
	if b.c.det { // the separate kernels: the CPU's order
		b.rmsnorm(x, w, x, tokens*heads, headDim, eps)
		b.rope(x, cs, sn, tokens, heads, headDim)
		return
	}
	b.normRopeKV(x, w, cs, sn, gpuRef{}, tokens, heads, headDim, 0, 0, eps)
}

func (b *cudaBatch) normRopeKV(x, w, cs, sn, kc gpuRef, tokens, heads, headDim, pos, ctxP int, eps float64) {
	if b.c.det {
		panic("cuda: normRopeKV in deterministic mode")
	}
	b.wrote()
	store, kcArg := uint32(0), carg(x)
	if kc.buf != nil {
		store, kcArg = 1, carg(kc)
	}
	b.launch("norm_rope_f32", [3]uint32{uint32(tokens * heads), 1, 1}, [3]uint32{uint32(headDim / 2), 1, 1}, 0,
		carg(x), carg(w), carg(cs), carg(sn),
		cuda.Arg{Bytes: u32(uint32(heads), uint32(headDim), math.Float32bits(float32(eps)), store, uint32(ctxP), uint32(pos))}, kcArg)
}

// gateUp fuses the matvecs and swiglu for small batches (same weight types); prompts run the three steps.
func (b *cudaBatch) gateUp(wg, wu *weight, x, gate, up, out gpuRef, n int, act gpuRef) {
	k, ok := cudaMatmuls[wg.typ]
	if b.c.det || !ok || n >= matmulTiledMin || wg.typ != wu.typ || wg.rows != wu.rows || wg.cols != wu.cols {
		b.matmul(wg, x, gate, n, act)
		b.matmul(wu, x, up, n, act)
		b.swiglu(gate, up, out, n*wg.rows)
		return
	}
	b.wrote()
	const rows = 4 * cudaMVRows / 2 // 4 warps per block, half the virtual rows each
	b.launch(k.matvec[:len(k.matvec)-len("f32")]+"pair", [3]uint32{uint32((wg.rows + rows - 1) / rows), uint32(n), 1}, [3]uint32{128, 1, 1}, 0,
		b.weightArg(wg), carg(x), carg(out), cuda.Arg{Bytes: u32(uint32(wg.rows), uint32(wg.cols/k.step), uint32(wg.rowBytes), uint32(wg.cols))},
		b.weightArg(wu))
}

func (b *cudaBatch) addNorm(x, y, w, out gpuRef, rows, n int, eps float64) {
	if b.c.det {
		b.add(x, y, rows*n)
		b.rmsnorm(x, w, out, rows, n, eps)
		return
	}
	b.wrote()
	b.launch("add_norm_f32", [3]uint32{uint32(rows), 1, 1}, [3]uint32{uint32(min(1024, (n+31)&^31)), 1, 1}, 0,
		carg(x), carg(y), carg(w), carg(out), cuda.Arg{Bytes: u32(uint32(n), math.Float32bits(float32(eps)))})
}

func (b *cudaBatch) add(x, y gpuRef, n int) {
	b.wrote()
	b.elementwise("add_f32", n, carg(x), carg(y))
}

func (b *cudaBatch) copy(dst, src gpuRef, n int) {
	b.wrote()
	b.elementwise("copy_f32", n, carg(dst), carg(src))
}

func (b *cudaBatch) storeKVQ(k, v, kc, vc, stage gpuRef, n, pos, kvd, ctxP, oldStageBlk int, vtyp uint32) {
	b.wrote()
	vb := uint32(8)
	if vtyp == typeQ4_0 {
		vb = 4
	}
	args := cuda.Arg{Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos), uint32(n), uint32(oldStageBlk), vb)}
	b.launch("kv_store_qk", [3]uint32{uint32((n*kvd/32 + 7) / 8), 1, 1}, [3]uint32{256, 1, 1}, 0, carg(k), carg(kc), args)
	b.launch("kv_store_qv", [3]uint32{uint32((kvd + 7) / 8), 1, 1}, [3]uint32{256, 1, 1}, 0, carg(v), carg(vc), carg(stage), args)
}

func (b *cudaBatch) packRows(src, dst gpuRef, rows, stride, off, length int) {
	b.wrote()
	b.launch("copy_rows_u8", [3]uint32{uint32((rows*length + 255) / 256), 1, 1}, [3]uint32{256, 1, 1}, 0,
		carg(src), carg(dst), cuda.Arg{Bytes: u32(uint32(rows), uint32(stride), uint32(off), uint32(length))})
}

func (b *cudaBatch) storeKV16(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int) {
	b.wrote()
	b.launch("kv_store_f16", [3]uint32{uint32((kvd + 255) / 256), uint32(n), 1}, [3]uint32{256, 1, 1}, 0,
		carg(k), carg(v), carg(kc), carg(vc), cuda.Arg{Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos))})
}

func (b *cudaBatch) storeKV(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int) {
	b.wrote()
	b.launch("kv_store_f32", [3]uint32{uint32((kvd + 255) / 256), uint32(n), 1}, [3]uint32{256, 1, 1}, 0,
		carg(k), carg(v), carg(kc), carg(vc), cuda.Arg{Bytes: u32(uint32(kvd), uint32(ctxP), uint32(pos))})
}

func (b *cudaBatch) attention(a gpuAttention) {
	kv := map[uint32]string{typeF32: "_f32", typeF16: "_f16", typeQ8_0: "_q8", typeQ4_0: "_q4"}[a.kvType]
	if kv == "" || kv != "_f32" && b.c.det {
		panic(fmt.Sprintf("cuda: attention over a %s cache isn't supported", kvTypeName(a.kvType)))
	}
	stage := carg(a.k) // block caches' stage (unused by the others)
	if a.stage.buf != nil {
		stage = carg(a.stage)
	}
	ss := uint32(a.stageStart)
	b.wrote()
	if b.c.det { // the CPU's order: one pass per (token, head), no split (its partial sums would be added in another order)
		b.launch("attention_det", [3]uint32{uint32(a.heads), uint32(a.n), 1}, [3]uint32{128, 1, 1}, 4*(a.headDim+4),
			carg(a.q), carg(a.k), carg(a.v), carg(a.out), carg(a.scratch), carg(a.mask),
			cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale())})
		return
	}
	const nt = 256
	if chunk, splits, ok := a.split(); ok {
		args := cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos),
			a.scale(), uint32(chunk), uint32(splits), ss)}
		b.launch("attention_split"+kv, [3]uint32{uint32(a.heads), uint32(a.n), uint32(splits)}, [3]uint32{nt, 1, 1},
			4*(a.headDim+chunk+nt/32*a.headDim), carg(a.q), carg(a.k), carg(a.v), carg(a.scratch), carg(a.mask), args, stage)
		b.launch("attention_combine_f32", [3]uint32{uint32(a.heads), uint32(a.n), 1},
			[3]uint32{uint32((a.headDim + 31) &^ 31), 1, 1}, 0, carg(a.scratch), carg(a.out), args)
		return
	}
	if a.n > attnSplitTokens && a.headDim == 128 && !cudaAttnOnePass && !cudaAttnTile { // prompts on the tensor cores
		b.launch("attention_flash"+kv, [3]uint32{uint32(a.heads), uint32((a.n + 63) / 64), 1}, [3]uint32{128, 1, 1}, 0,
			carg(a.q), carg(a.k), carg(a.v), carg(a.out), carg(a.mask),
			cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale(), uint32(a.n), ss)}, stage)
		return
	}
	if a.n > attnSplitTokens && a.headDim <= 128 && !cudaAttnOnePass { // prompts; one-token steps keep the one-pass kernel
		const qt = 16 // ATT_QT
		b.launch("attention_tile"+kv, [3]uint32{uint32(a.heads), uint32((a.n + qt - 1) / qt), 1}, [3]uint32{256, 1, 1}, 0,
			carg(a.q), carg(a.k), carg(a.v), carg(a.out), carg(a.mask),
			cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos), a.scale(), uint32(a.n), ss)}, stage)
		return
	}
	b.launch("attention"+kv, [3]uint32{uint32(a.heads), uint32(a.n), 1}, [3]uint32{nt, 1, 1}, 4*(a.headDim+nt),
		carg(a.q), carg(a.k), carg(a.v), carg(a.out), carg(a.scratch), carg(a.mask),
		cuda.Arg{Bytes: u32(uint32(a.heads), uint32(a.kvHeads), uint32(a.headDim), uint32(a.ctxP), uint32(a.pos),
			a.scale(), ss)}, stage)
}
