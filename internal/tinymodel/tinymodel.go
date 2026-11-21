// Package tinymodel writes a tiny Qwen3 GGUF with random weights, for tests (nollamatest re-exports it; the engine's
// own tests use it when GGUF is unset).
package tinymodel

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
)

// Config shapes the tiny model. The zero value is the default: 2 layers, 64-wide, about 1 MB.
type Config struct {
	Seed    uint64 // the random weights' seed (same seed, same model)
	Layers  int    // default 2
	Dim     int    // embedding width, a multiple of 16 (default 64)
	Context int    // context length (default 2048)
	// EndTurns lets the model end its answer on its own. By default it can't: its end-of-turn tokens never win, so an
	// unconstrained answer runs to max tokens ("length"), and a constrained one ends when the constraint is done.
	EndTurns bool
}

// The tiny model's vocabulary: the 256 byte tokens of a GPT-2 byte-level BPE, merges that build common English words
// (so text takes about as many tokens as with a real model), and Qwen3's chat tokens.
var (
	tinyWords = strings.Fields(`the a an and or of to in is it you your for on with that this are be as at by from not can
		will do how what when where who why yes no hi hello say write name one two three four five ten count number
		color sky blue red green note notes sync across devices within few seconds saving answer assistant help app
		below excerpts share notebook fast label ticket bug billing question charged twice terse short sentence word
		words title post about make list give tell me my we our they their there here more most some any all each
		every new old good bad first last next time day year way thing people work use used using get got set run
		read reading writing file files model models test tests text json`)
	tinySpecials = []string{"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"}
)

func gpt2Bytes() [256]string {
	var enc [256]string
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
			r = rune(256 + n)
			n++
		}
		enc[b] = string(r)
	}
	return enc
}

// tinyVocab returns the tokens, their types and the merges, and which tokens the model writes on its own: whole
// words after a space (those that tokenize back to that one token), "." and ",". Each word, plain and capitalized,
// gets a chain of left-to-right merges; then a merge of the space (Ġ) onto the whole word, ranked after every in-word
// merge, as a trained BPE would have it.
func tinyVocab() (tokens []string, types []int32, merges []string, writes map[int]bool) {
	enc := gpt2Bytes()
	ids := map[string]int{}
	add := func(tok string) {
		if _, ok := ids[tok]; !ok {
			ids[tok] = len(tokens)
			tokens = append(tokens, tok)
			types = append(types, 1)
		}
	}
	for _, t := range enc {
		add(t)
	}
	ranks := map[string]int{}
	merge := func(a, b string) {
		if m := a + " " + b; ranks[m] == 0 {
			merges = append(merges, m)
			ranks[m] = len(merges)
		}
		add(a + b)
	}
	var forms []string
	for _, w := range tinyWords {
		forms = append(forms, w, strings.ToUpper(w[:1])+w[1:])
	}
	symbols := func(s string) []string {
		var out []string
		for i := 0; i < len(s); i++ {
			out = append(out, enc[s[i]])
		}
		return out
	}
	for _, f := range forms {
		syms := symbols(f)
		cur := syms[0]
		for _, next := range syms[1:] {
			merge(cur, next)
			cur += next
		}
	}
	space := enc[' ']
	for _, f := range forms {
		merge(space, strings.Join(symbols(f), ""))
	}
	writes = map[int]bool{'.': true, ',': true}
	for _, f := range forms {
		if got := bpe(symbols(" "+f), ranks); len(got) == 1 {
			writes[ids[got[0]]] = true
		}
	}
	for _, s := range tinySpecials {
		tokens = append(tokens, s)
		types = append(types, 3)
	}
	return tokens, types, merges, writes
}

// bpe merges syms the way the engine's tokenizer does: the lowest-ranked pair first.
func bpe(syms []string, ranks map[string]int) []string {
	for len(syms) > 1 {
		best, at := 0, -1
		for i := 0; i+1 < len(syms); i++ {
			if r := ranks[syms[i]+" "+syms[i+1]]; r > 0 && (at < 0 || r < best) {
				best, at = r, i
			}
		}
		if at < 0 {
			break
		}
		syms[at] += syms[at+1]
		syms = append(syms[:at+1], syms[at+2:]...)
	}
	return syms
}

// GGUF returns a complete Qwen3 GGUF file with random weights: a real model the engine loads and runs, with real
// tokenization, sampling and constraints, small enough to build in every test. Its answers are gibberish, but at
// temperature 0 they are the same every run on a given machine (floating-point details differ between CPUs, so don't
// compare exact text across machines).
func GGUF(c Config) []byte {
	if c.Layers <= 0 {
		c.Layers = 2
	}
	if c.Dim <= 0 {
		c.Dim = 64
	}
	if c.Context <= 0 {
		c.Context = 2048
	}
	heads, kvHeads := 4, 2
	headDim := c.Dim / heads
	ffn := 2 * c.Dim
	tokens, types, merges, writes := tinyVocab()
	vocab := len(tokens)
	eos := vocab - len(tinySpecials) // <|endoftext|>

	w := &ggufWriter{}
	w.kv("general.architecture", "qwen3")
	w.kv("general.name", "nollama-tiny")
	w.kv("general.file_type", uint32(0)) // F32
	w.kv("general.alignment", uint32(32))
	w.kv("qwen3.context_length", uint32(c.Context))
	w.kv("qwen3.embedding_length", uint32(c.Dim))
	w.kv("qwen3.block_count", uint32(c.Layers))
	w.kv("qwen3.feed_forward_length", uint32(ffn))
	w.kv("qwen3.attention.head_count", uint32(heads))
	w.kv("qwen3.attention.head_count_kv", uint32(kvHeads))
	w.kv("qwen3.attention.key_length", uint32(headDim))
	w.kv("qwen3.attention.value_length", uint32(headDim))
	w.kv("qwen3.attention.layer_norm_rms_epsilon", float32(1e-6))
	w.kv("qwen3.rope.freq_base", float32(1e6))
	w.kv("tokenizer.ggml.model", "gpt2")
	w.kv("tokenizer.ggml.pre", "qwen2")
	w.kv("tokenizer.ggml.tokens", tokens)
	w.kv("tokenizer.ggml.token_type", types)
	w.kv("tokenizer.ggml.merges", merges)
	w.kv("tokenizer.ggml.eos_token_id", uint32(eos))

	rng := rand.New(rand.NewPCG(c.Seed, c.Seed^0x6e6f6c6c616d61))
	random := func(n int, scale float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64() * scale)
		}
		return out
	}
	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}
	w.tensor("token_embd.weight", []int{c.Dim, vocab}, random(c.Dim*vocab, 1))
	// Only whole words after a space, "." and "," (and, with EndTurns, the chat tokens) get random output rows; the
	// rest have logits of 0, below the best of the ~600 random ones. So the model writes words, and what it writes
	// tokenizes back to the same tokens, as a real model's answers mostly do (threads rely on it).
	out := random(c.Dim*vocab, 1)
	for id := 0; id < vocab; id++ {
		if !writes[id] && !(id >= eos && c.EndTurns) {
			clear(out[id*c.Dim : (id+1)*c.Dim])
		}
	}
	w.tensor("output.weight", []int{c.Dim, vocab}, out)
	w.tensor("output_norm.weight", []int{c.Dim}, ones(c.Dim))
	qd, kvd := heads*headDim, kvHeads*headDim
	s := 1 / math.Sqrt(float64(c.Dim))
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		w.tensor(p+"attn_norm.weight", []int{c.Dim}, ones(c.Dim))
		w.tensor(p+"ffn_norm.weight", []int{c.Dim}, ones(c.Dim))
		w.tensor(p+"attn_q_norm.weight", []int{headDim}, ones(headDim))
		w.tensor(p+"attn_k_norm.weight", []int{headDim}, ones(headDim))
		w.tensor(p+"attn_q.weight", []int{c.Dim, qd}, random(c.Dim*qd, s))
		w.tensor(p+"attn_k.weight", []int{c.Dim, kvd}, random(c.Dim*kvd, s))
		w.tensor(p+"attn_v.weight", []int{c.Dim, kvd}, random(c.Dim*kvd, s))
		w.tensor(p+"attn_output.weight", []int{qd, c.Dim}, random(qd*c.Dim, s))
		w.tensor(p+"ffn_gate.weight", []int{c.Dim, ffn}, random(c.Dim*ffn, s))
		w.tensor(p+"ffn_up.weight", []int{c.Dim, ffn}, random(c.Dim*ffn, s))
		w.tensor(p+"ffn_down.weight", []int{ffn, c.Dim}, random(ffn*c.Dim, 1/math.Sqrt(float64(ffn))))
	}
	return w.bytes()
}

// Write writes GGUF(c) to path.
func Write(path string, c Config) error {
	return os.WriteFile(path, GGUF(c), 0o644)
}

// ggufWriter builds a GGUF v3 file with F32 tensors.
type ggufWriter struct {
	kvs     bytes.Buffer
	nkv     int
	infos   bytes.Buffer
	data    bytes.Buffer
	ntensor int
}

func putStr(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func (w *ggufWriter) kv(key string, v any) {
	b := &w.kvs
	putStr(b, key)
	le := binary.LittleEndian
	switch v := v.(type) {
	case uint32:
		_ = binary.Write(b, le, uint32(4))
		_ = binary.Write(b, le, v)
	case float32:
		_ = binary.Write(b, le, uint32(6))
		_ = binary.Write(b, le, v)
	case string:
		_ = binary.Write(b, le, uint32(8))
		putStr(b, v)
	case []string:
		_ = binary.Write(b, le, uint32(9))
		_ = binary.Write(b, le, uint32(8))
		_ = binary.Write(b, le, uint64(len(v)))
		for _, s := range v {
			putStr(b, s)
		}
	case []int32:
		_ = binary.Write(b, le, uint32(9))
		_ = binary.Write(b, le, uint32(5))
		_ = binary.Write(b, le, uint64(len(v)))
		_ = binary.Write(b, le, v)
	default:
		panic(fmt.Sprintf("ggufWriter: %T", v))
	}
	w.nkv++
}

func (w *ggufWriter) tensor(name string, dims []int, data []float32) {
	for w.data.Len()%32 != 0 {
		w.data.WriteByte(0)
	}
	le := binary.LittleEndian
	putStr(&w.infos, name)
	_ = binary.Write(&w.infos, le, uint32(len(dims)))
	for _, d := range dims {
		_ = binary.Write(&w.infos, le, uint64(d))
	}
	_ = binary.Write(&w.infos, le, uint32(0)) // F32
	_ = binary.Write(&w.infos, le, uint64(w.data.Len()))
	_ = binary.Write(&w.data, le, data)
	w.ntensor++
}

func (w *ggufWriter) bytes() []byte {
	var b bytes.Buffer
	le := binary.LittleEndian
	b.WriteString("GGUF")
	_ = binary.Write(&b, le, uint32(3))
	_ = binary.Write(&b, le, uint64(w.ntensor))
	_ = binary.Write(&b, le, uint64(w.nkv))
	b.Write(w.kvs.Bytes())
	b.Write(w.infos.Bytes())
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(w.data.Bytes())
	return b.Bytes()
}
