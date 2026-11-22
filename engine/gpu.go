package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
)

// GPU backends. forward talks to gpuBackend and gpuBatch only; each GPU API (Metal today, CUDA or Vulkan
// later) implements them. The contract holds for discrete GPUs too: buffers live on the device and move
// with write/read, and unified-memory backends make those copies cheap (Metal wraps the mmapped model
// file instead of uploading it). Every backend runs the same kernels with the same layouts:
//
//   - activations are f32, row-major, one row per token;
//   - weights stay in their GGUF formats, rows as in the file;
//   - the f32 KV cache keeps the CPU's layout (keys [kvHeads][headDim][ctxP], values [ctxP][kvHeads*headDim]),
//     so prompt caching and threads work unchanged.
//
// gpuModel (gpu_forward.go) runs the forward pass on it.
type gpuBackend interface {
	name() string // "metal (Apple M1 Max)"
	unified() bool
	supports(weightType uint32) bool // weights of other types run on the CPU
	supportsKV(kvType uint32) bool   // KV cache formats the attention kernels take (typeF32 always)
	// halfPrompts reports that prompt-sized matmuls round their inputs to half (tensor cores): about 4e-5 of a
	// row's magnitude instead of f32's 1e-7, still far closer to the float reference than the CPU's 8-bit path.
	halfPrompts() bool
	// int8Prompts reports that prompt-sized matmuls of some weight types quantize their inputs to 8 bits (int8
	// tensor cores), as the CPU's fast path does: about 1% of the logits' spread.
	int8Prompts() bool
	// loadWeights makes the model file's weights available to batches: zero-copy on unified memory,
	// one upload otherwise. Called once, before any batch.
	loadWeights(file []byte) error
	alloc(n int) (gpuBuf, error)
	write(dst gpuRef, src []byte) // host to device
	read(src gpuRef, dst []byte)  // device to host
	// mapped returns buf's memory as the CPU sees it on unified memory (the KV cache is shared this way
	// with the prompt cache and threads), or nil on a discrete GPU.
	mapped(buf gpuBuf) []byte
	batch() gpuBatch
	close()
}

// gpuBuf is device memory.
type gpuBuf interface {
	size() int
	release()
}

// gpuRef is a position in a buffer.
type gpuRef struct {
	buf gpuBuf
	off int // bytes
}

func (r gpuRef) at(off int) gpuRef { return gpuRef{r.buf, r.off + off} }

// gpuBatch records operations to run in order; submit runs them and waits.
type gpuBatch interface {
	// matmul: out[t*w.rows + r] = W row r · x[t*w.cols:], for n tokens. The backend picks its
	// one-token or tiled kernel by n. act is scratch for the quantized activations of deterministic
	// backends (gpuActBytes); others ignore it.
	matmul(w *weight, x, out gpuRef, n int, act gpuRef)
	// rmsnorm normalizes rows rows of n floats from x into out (may equal x), scaled by w (n floats).
	rmsnorm(x, w, out gpuRef, rows, n int, eps float64)
	// rope applies NeoX RoPE in place to x [tokens][heads][headDim] with cos/sin tables [tokens][headDim/2].
	rope(x, cs, sn gpuRef, tokens, heads, headDim int)
	// normRope is rmsnorm(x, w, x, tokens*heads, headDim, eps) then rope(x, cs, sn, tokens, heads, headDim).
	normRope(x, w, cs, sn gpuRef, tokens, heads, headDim int, eps float64)
	// normRopeKV is normRope that also writes the result into a layer's f32 key cache kc at positions pos.. (the
	// keys half of storeKV). Not in deterministic mode.
	normRopeKV(x, w, cs, sn, kc gpuRef, tokens, heads, headDim, pos, ctxP int, eps float64)
	// gateUp is matmul(wg, x, gate), matmul(wu, x, up), swiglu(gate, up, out): one dispatch where the backend can.
	gateUp(wg, wu *weight, x, gate, up, out gpuRef, n int, act gpuRef)
	// addNorm is add(x, y, rows*n) then rmsnorm(x, w, out, rows, n, eps).
	addNorm(x, y, w, out gpuRef, rows, n int, eps float64)
	swiglu(gate, up, out gpuRef, n int) // out = silu(gate) * up
	add(x, y gpuRef, n int)             // x += y
	copy(dst, src gpuRef, n int)        // n floats
	// storeKV writes n tokens' keys and values (each [n][kvd]) into a layer's f32 cache at positions pos..
	storeKV(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int)
	// storeKV16 writes them into a layer's f16 cache in the compact layout: keys by position (kc[p*kvd + i]), values
	// by channel (vc[i*ctxP + p]), as halves.
	storeKV16(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int)
	// storeKVQ writes them into a layer's block cache (q8_0 keys; vtyp q8_0 or q4_0 values) as the CPU's storeKey and
	// storeValue do: keys quantized in their position rows, values into their channel rows' blocks, the newest block
	// at full precision in stage ([kvd][32]). oldStageBlk is the block the stage held before.
	storeKVQ(k, v, kc, vc, stage gpuRef, n, pos, kvd, ctxP, oldStageBlk int, vtyp uint32)
	// packRows copies bytes [off, off+length) of rows rows (stride bytes apart) of src into dst, contiguous.
	packRows(src, dst gpuRef, rows, stride, off, length int)
	attention(a gpuAttention)
	submit() error
}

// gpuAttention is one layer's causal attention for n tokens at positions pos..pos+n-1.
type gpuAttention struct {
	q, out     gpuRef // [n][heads][headDim]
	k, v       gpuRef // the layer's cache, in kvType's layout
	scratch    gpuRef // n*heads*ctxP floats
	mask       gpuRef // n pairs of uint32: positions [lo, hi) token t must not see (packed sequences)
	kvType     uint32 // typeF32, or typeF16, typeQ8_0, typeQ4_0 (the compact layouts: kvcache.go)
	stage      gpuRef // block caches: the newest block's values ([kvd][32] floats), positions from stageStart
	stageStart int
	n, pos     int
	heads      int
	kvHeads    int
	headDim    int
	ctxP       int // cache rows, a multiple of 16
}

// gpuOpeners are the backends to try, in order of preference. det asks for the deterministic kernels: the CPU's
// reference results bit for bit (a backend without them fails to open).
var gpuOpeners = []func(det bool) (gpuBackend, error){openMetal, openCUDA}

// gpuActBytes bounds the quantized form of n activation rows of k floats (Q8_0 or Q8_K blocks).
func gpuActBytes(n, k int) int { return n * (k/qk + 1) * q8kBytes * 2 }

// openGPU returns the first backend that works on this machine, with the model's weights loaded,
// or an error listing why each failed.
func openGPU(g *ggufFile, det bool) (gpuBackend, error) {
	var why []string
	for _, open := range gpuOpeners {
		b, err := open(det)
		if err != nil {
			why = append(why, err.Error())
			continue
		}
		if err := b.loadWeights(g.data); err != nil {
			b.close()
			why = append(why, fmt.Sprintf("%s: %v", b.name(), err))
			continue
		}
		return b, nil
	}
	if len(why) == 0 {
		return nil, errors.New("engine: no GPU backends")
	}
	return nil, errors.New("engine: no usable GPU: " + strings.Join(why, "; "))
}

func u32(vs ...uint32) []byte {
	b := make([]byte, 4*len(vs))
	for i, v := range vs {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	return b
}

// Split attention ("flash decoding"): one token attending to a long context is a few threadgroups doing long
// serial loops, so batches of up to attnSplitTokens tokens split the context into chunks that run in
// parallel, then merge the chunks' partial results. The partials go in the scratch buffer.
const (
	attnSplitTokens = 8
	attnSplitChunk  = 256 // positions per chunk, one per thread
)

var attnSplitOff = false // tests compare against the one-pass kernel

// split returns the chunk size and count for split attention, or ok false for the one-pass kernel.
func (a gpuAttention) split() (chunk, splits int, ok bool) {
	if a.n > attnSplitTokens || a.headDim > 256 || attnSplitOff {
		return 0, 0, false
	}
	ctx := a.pos + a.n
	for chunk = attnSplitChunk; chunk <= 2048; chunk *= 2 {
		splits = (ctx + chunk - 1) / chunk
		if splits < 2 {
			break
		}
		if splits*(a.headDim+2) <= a.ctxP { // partials fit in the scratch (n*heads*ctxP floats)
			return chunk, splits, true
		}
	}
	return 0, 0, false
}

func (a gpuAttention) scale() uint32 {
	return math.Float32bits(float32(1 / math.Sqrt(float64(a.headDim))))
}
