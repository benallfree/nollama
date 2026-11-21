package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
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
// Experimental: nothing in forward uses it yet.
type gpuBackend interface {
	name() string // "metal (Apple M1 Max)"
	unified() bool
	supports(weightType uint32) bool // weights of other types run on the CPU
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
	// one-token or tiled kernel by n.
	matmul(w *weight, x, out gpuRef, n int)
	// rmsnorm normalizes rows rows of n floats from x into out (may equal x), scaled by w (n floats).
	rmsnorm(x, w, out gpuRef, rows, n int, eps float64)
	// rope applies NeoX RoPE in place to x [tokens][heads][headDim] with cos/sin tables [tokens][headDim/2].
	rope(x, cs, sn gpuRef, tokens, heads, headDim int)
	swiglu(gate, up, out gpuRef, n int) // out = silu(gate) * up
	add(x, y gpuRef, n int)             // x += y
	// storeKV writes n tokens' keys and values (each [n][kvd]) into a layer's f32 cache at positions pos..
	storeKV(k, v, kc, vc gpuRef, n, pos, kvd, ctxP int)
	attention(a gpuAttention)
	submit() error
}

// gpuAttention is one layer's causal attention for n tokens at positions pos..pos+n-1.
type gpuAttention struct {
	q, out  gpuRef // [n][heads][headDim]
	k, v    gpuRef // the layer's cache, in kvType's layout
	scratch gpuRef // n*heads*ctxP floats
	mask    gpuRef // n pairs of uint32: positions [lo, hi) token t must not see (packed sequences)
	kvType  uint32 // typeF32 only, so far
	n, pos  int
	heads   int
	kvHeads int
	headDim int
	ctxP    int // cache rows, a multiple of 16
}

// gpuOpeners are the backends to try, in order of preference.
var gpuOpeners = []func() (gpuBackend, error){openMetal}

// openGPU returns the first backend that works on this machine, with the model's weights loaded,
// or an error listing why each failed.
func openGPU(g *ggufFile) (gpuBackend, error) {
	var why []string
	for _, open := range gpuOpeners {
		b, err := open()
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
