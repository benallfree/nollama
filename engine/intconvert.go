package engine

// Converting official Qwen2.5 bf16 weights (a Hugging Face snapshot: config.json, tokenizer.json, *.safetensors) into
// the integer codec's GGUF, byte for byte the same file as the reference converter (compress13k intref.py build), so
// its SHA-256 matches the manifest. Every step is IEEE arithmetic that Go reproduces exactly (division, power-of-two
// scaling, round-half-to-even, frexp); the lookup tables are generated with Go's math and checked against pinned
// hashes.

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// ---- safetensors (bf16)

type stTensor struct {
	file  int
	off   int64
	shape []int
}

type stSet struct {
	files []*os.File
	data  [][]byte // mmapped
	maps  [][]byte
	t     map[string]stTensor
}

func openSafetensors(dir string) (*stSet, error) {
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, fmt.Errorf("intcodec: no .safetensors in %s", dir)
	}
	s := &stSet{t: map[string]stTensor{}}
	for i, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			s.close()
			return nil, err
		}
		st, _ := f.Stat()
		mp, d, err := mmapRange(f, 0, st.Size())
		if err != nil {
			f.Close()
			s.close()
			return nil, err
		}
		s.files, s.maps, s.data = append(s.files, f), append(s.maps, mp), append(s.data, d)
		n := binary.LittleEndian.Uint64(d[:8])
		var hdr map[string]json.RawMessage
		if err := json.Unmarshal(d[8:8+n], &hdr); err != nil {
			s.close()
			return nil, err
		}
		for k, raw := range hdr {
			if k == "__metadata__" {
				continue
			}
			var v struct {
				Dtype   string  `json:"dtype"`
				Shape   []int   `json:"shape"`
				Offsets []int64 `json:"data_offsets"`
			}
			if err := json.Unmarshal(raw, &v); err != nil {
				s.close()
				return nil, err
			}
			if v.Dtype != "BF16" {
				s.close()
				return nil, fmt.Errorf("intcodec: %s is %s, want BF16", k, v.Dtype)
			}
			s.t[k] = stTensor{file: i, off: 8 + int64(n) + v.Offsets[0], shape: v.Shape}
		}
	}
	return s, nil
}

func (s *stSet) close() {
	for i := range s.files {
		if s.maps[i] != nil {
			munmap(s.maps[i])
		}
		s.files[i].Close()
	}
}

// rows returns the tensor's bf16 rows (each row: cols values) as float64.
func (s *stSet) get(name string) (rows, cols int, at func(r, c int) float64, err error) {
	t, ok := s.t[name]
	if !ok {
		return 0, 0, nil, fmt.Errorf("intcodec: missing tensor %s", name)
	}
	rows, cols = 1, t.shape[len(t.shape)-1]
	for _, d := range t.shape[:len(t.shape)-1] {
		rows *= d
	}
	d := s.data[t.file][t.off:]
	return rows, cols, func(r, c int) float64 {
		i := 2 * (r*cols + c)
		return float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(d[i:])) << 16))
	}, nil
}

// ---- the converter's view of a (possibly fused) matrix

type convMat struct {
	name  string
	parts []string // safetensors tensors, rows concatenated in order
	rows  int
	cols  int
	e     int
}

func (m *convMat) at(s *stSet) (func(r, c int) float64, error) {
	type part struct {
		start int
		at    func(r, c int) float64
	}
	var ps []part
	r0 := 0
	for _, p := range m.parts {
		rows, cols, at, err := s.get(p)
		if err != nil {
			return nil, err
		}
		if r0 == 0 {
			m.cols = cols
		} else if cols != m.cols {
			return nil, fmt.Errorf("intcodec: %s: column mismatch", m.name)
		}
		ps = append(ps, part{r0, at})
		r0 += rows
	}
	m.rows = r0
	return func(r, c int) float64 {
		i := len(ps) - 1
		for ps[i].start > r {
			i--
		}
		return ps[i].at(r-ps[i].start, c)
	}, nil
}

func parallelRows(n int, body func(lo, hi int)) {
	w := max(1, min(n, 16))
	var wg sync.WaitGroup
	for k := 0; k < w; k++ {
		lo, hi := k*n/w, (k+1)*n/w
		wg.Add(1)
		go func() { defer wg.Done(); body(lo, hi) }()
	}
	wg.Wait()
}

// groupScale is max|w| / 127 over a group of 64 (1e-30 for an all-zero group), as the reference computes it.
func groupScale(at func(r, c int) float64, r, g int) float64 {
	var mx float64
	for k := 0; k < 64; k++ {
		mx = max(mx, math.Abs(at(r, g*64+k)))
	}
	sr := mx / 127
	if sr == 0 {
		sr = 1e-30
	}
	return sr
}

// exponent: floor(log2(max scale)) - 14, from frexp (exact, no libm).
func (m *convMat) exponent(s *stSet) error {
	at, err := m.at(s)
	if err != nil {
		return err
	}
	if m.cols%128 != 0 {
		return fmt.Errorf("intcodec: %s: %d columns, want a multiple of 128", m.name, m.cols)
	}
	ng := m.cols / 64
	mxs := make([]float64, 16)
	var mu sync.Mutex
	parallelRows(m.rows, func(lo, hi int) {
		var mx float64
		for r := lo; r < hi; r++ {
			for g := 0; g < ng; g++ {
				mx = max(mx, groupScale(at, r, g))
			}
		}
		mu.Lock()
		mxs = append(mxs, mx)
		mu.Unlock()
	})
	var mx float64
	for _, v := range mxs {
		mx = max(mx, v)
	}
	_, e := math.Frexp(mx)
	m.e = e - 1 - 14
	return nil
}

// quantize writes q (int8, rows x cols) and s (int16, rows x cols/64).
func (m *convMat) quantize(s *stSet) (q []byte, sc []byte, err error) {
	at, err := m.at(s)
	if err != nil {
		return nil, nil, err
	}
	ng := m.cols / 64
	q, sc = make([]byte, m.rows*m.cols), make([]byte, m.rows*ng*2)
	pow := math.Ldexp(1, m.e)
	parallelRows(m.rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for g := 0; g < ng; g++ {
				sv := min(max(math.RoundToEven(groupScale(at, r, g)/pow), 1), 32767)
				binary.LittleEndian.PutUint16(sc[2*(r*ng+g):], uint16(int16(sv)))
				den := sv * pow
				for k := 0; k < 64; k++ {
					c := g*64 + k
					q[r*m.cols+c] = byte(int8(min(max(math.RoundToEven(at(r, c)/den), -127), 127)))
				}
			}
		}
	})
	return q, sc, nil
}

// fixed-point vector: rint(v * 2^k) as int64, rows of several tensors concatenated.
func convFixed(s *stSet, names []string, k int) ([]byte, error) {
	var out []byte
	scale := math.Ldexp(1, k)
	for _, n := range names {
		rows, cols, at, err := s.get(n)
		if err != nil {
			return nil, err
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				out = binary.LittleEndian.AppendUint64(out, uint64(int64(math.RoundToEven(at(r, c)*scale))))
			}
		}
	}
	return out, nil
}

// ---- lookup tables (generated, checked against the frozen tables' hashes)

var intTableSHA = map[string]string{
	"EXPA":       "6a5c35d8aab15670",
	"EXPF":       "f1bc8936cb9711d3",
	"SIG":        "bf265d6da83c93a9",
	"ROPE_C_64":  "d8095cb9491c2b30",
	"ROPE_S_64":  "f2f65aefb6003b23",
	"ROPE_C_128": "b23e31ed33b550c6",
	"ROPE_S_128": "6f1899b8e578f28f",
}

func intTables(hd int) (map[string][]byte, error) {
	i32 := func(n int, f func(i int) float64) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(math.RoundToEven(f(i)))))
		}
		return b
	}
	t := map[string][]byte{
		"EXPA": i32(intNA, func(i int) float64 { return 32768 * math.Exp(-float64(i)/1024) }),
		"EXPF": i32(intNF, func(i int) float64 { return 1073741824 * math.Exp(-float64(i)/1024) }),
		"SIG":  i32(2*intNS+1, func(i int) float64 { return 65536 / (1 + math.Exp(-float64(i-intNS)/1024)) }),
	}
	c, s := make([]byte, 32768*hd), make([]byte, 32768*hd) // int16, [32768][hd/2]
	for p := 0; p < 32768; p++ {
		for j := 0; j < hd/2; j++ {
			a := float64(p) * math.Pow(1e6, -float64(2*j)/float64(hd))
			binary.LittleEndian.PutUint16(c[2*(p*hd/2+j):], uint16(int16(math.RoundToEven(math.Cos(a)*16384))))
			binary.LittleEndian.PutUint16(s[2*(p*hd/2+j):], uint16(int16(math.RoundToEven(math.Sin(a)*16384))))
		}
	}
	t[fmt.Sprint("ROPE_C_", hd)], t[fmt.Sprint("ROPE_S_", hd)] = c, s
	for k, b := range t {
		h := sha256.Sum256(b)
		if want := intTableSHA[k]; hex.EncodeToString(h[:])[:16] != want {
			return nil, fmt.Errorf("intcodec: this platform computes table %s differently (sha %x, want %s); "+
				"converting here would not give the manifest's file", k, h[:8], want)
		}
	}
	return t, nil
}

// ---- GGUF writer (the reference's exact layout)

type ggufW struct {
	b bytes.Buffer
}

func (w *ggufW) str(s string) {
	binary.Write(&w.b, binary.LittleEndian, uint64(len(s)))
	w.b.WriteString(s)
}

func (w *ggufW) u32(v uint32) { binary.Write(&w.b, binary.LittleEndian, v) }

func (w *ggufW) kvStr(k, v string) { w.str(k); w.u32(8); w.str(v) }
func (w *ggufW) kvU32(k string, v uint32) {
	w.str(k)
	w.u32(4)
	w.u32(v)
}
func (w *ggufW) kvI32(k string, v int32) {
	w.str(k)
	w.u32(5)
	binary.Write(&w.b, binary.LittleEndian, v)
}
func (w *ggufW) kvF64(k string, v float64) {
	w.str(k)
	w.u32(12)
	binary.Write(&w.b, binary.LittleEndian, v)
}
func (w *ggufW) kvBool(k string, v bool) {
	w.str(k)
	w.u32(7)
	if v {
		w.b.WriteByte(1)
	} else {
		w.b.WriteByte(0)
	}
}
func (w *ggufW) kvStrs(k string, v []string) {
	w.str(k)
	w.u32(9)
	w.u32(8)
	binary.Write(&w.b, binary.LittleEndian, uint64(len(v)))
	for _, s := range v {
		w.str(s)
	}
}
func (w *ggufW) kvI32s(k string, v []int32) {
	w.str(k)
	w.u32(9)
	w.u32(5)
	binary.Write(&w.b, binary.LittleEndian, uint64(len(v)))
	binary.Write(&w.b, binary.LittleEndian, v)
}

type convTensor struct {
	name string
	dims []uint64 // GGUF order (dims[0] contiguous)
	typ  uint32
	size int64
	data func() ([]byte, error)
}

// ConvertIntModel converts a Qwen2.5 bf16 snapshot directory into the integer codec's GGUF at dst.
func ConvertIntModel(srcDir, dst string, logf func(format string, args ...any)) error {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	cfgRaw, err := os.ReadFile(filepath.Join(srcDir, "config.json"))
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(cfgRaw))
	dec.UseNumber()
	var cfg map[string]any
	if err := dec.Decode(&cfg); err != nil {
		return err
	}
	num := func(k string) (json.Number, error) {
		n, ok := cfg[k].(json.Number)
		if !ok {
			return "", fmt.Errorf("intcodec: config.json: %s missing", k)
		}
		return n, nil
	}
	ival := func(k string) int {
		n, _ := num(k)
		v, _ := n.Int64()
		return int(v)
	}
	D, L := ival("hidden_size"), ival("num_hidden_layers")
	nq, V := ival("num_attention_heads"), ival("vocab_size")
	if th, _ := num("rope_theta"); th.String() != "1000000.0" && th.String() != "1000000" {
		return fmt.Errorf("intcodec: rope_theta %s, the tables cover 1e6 only", th)
	}
	if ival("max_position_embeddings") != 32768 {
		return errors.New("intcodec: the tables cover 32,768 positions only")
	}
	tie, _ := cfg["tie_word_embeddings"].(bool)
	hd := D / nq

	st, err := openSafetensors(srcDir)
	if err != nil {
		return err
	}
	defer st.close()
	tables, err := intTables(hd)
	if err != nil {
		return err
	}

	// matrices in the spec's order, with their exponents (a first pass over the weights)
	mats := []*convMat{{name: "emb", parts: []string{"model.embed_tokens.weight"}}}
	if !tie {
		mats = append(mats, &convMat{name: "head", parts: []string{"lm_head.weight"}})
	}
	for l := 0; l < L; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		a := p + "self_attn."
		mats = append(mats,
			&convMat{name: fmt.Sprint(l, ".qkv"), parts: []string{a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"}},
			&convMat{name: fmt.Sprint(l, ".o"), parts: []string{a + "o_proj.weight"}},
			&convMat{name: fmt.Sprint(l, ".gu"), parts: []string{p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}},
			&convMat{name: fmt.Sprint(l, ".d"), parts: []string{p + "mlp.down_proj.weight"}})
	}
	for i, m := range mats {
		if err := m.exponent(st); err != nil {
			return err
		}
		logf("exponents %d/%d", i+1, len(mats))
	}
	byName := map[string]*convMat{}
	for _, m := range mats {
		byName[m.name] = m
	}

	// tensors in the spec's order
	var ts []convTensor
	addMat := func(name string) {
		m := byName[name]
		var qd, sd []byte
		quant := func() error {
			if qd == nil {
				var err error
				qd, sd, err = m.quantize(st)
				return err
			}
			return nil
		}
		ts = append(ts,
			convTensor{name + ".q", []uint64{uint64(m.cols), uint64(m.rows)}, typeI8, int64(m.rows * m.cols),
				func() ([]byte, error) { err := quant(); return qd, err }},
			convTensor{name + ".s", []uint64{uint64(m.cols / 64), uint64(m.rows)}, typeI16, int64(m.rows * m.cols / 64 * 2),
				func() ([]byte, error) { err := quant(); b := sd; qd, sd = nil, nil; return b, err }})
	}
	addFixed := func(name string, parts []string, k, n int) {
		ts = append(ts, convTensor{name, []uint64{uint64(n)}, typeI64, int64(8 * n),
			func() ([]byte, error) { return convFixed(st, parts, k) }})
	}
	addMat("emb")
	if !tie {
		addMat("head")
	}
	addFixed("gf", []string{"model.norm.weight"}, 16, D)
	nkv := ival("num_key_value_heads")
	for l := 0; l < L; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		a := p + "self_attn."
		addMat(fmt.Sprint(l, ".qkv"))
		addFixed(fmt.Sprint(l, ".bqkv"), []string{a + "q_proj.bias", a + "k_proj.bias", a + "v_proj.bias"}, 24, (nq+2*nkv)*hd)
		addMat(fmt.Sprint(l, ".o"))
		addMat(fmt.Sprint(l, ".gu"))
		addMat(fmt.Sprint(l, ".d"))
		addFixed(fmt.Sprint(l, ".g1"), []string{p + "input_layernorm.weight"}, 16, D)
		addFixed(fmt.Sprint(l, ".g2"), []string{p + "post_attention_layernorm.weight"}, 16, D)
	}
	tab := func(name, key string, dims []uint64, typ uint32) {
		b := tables[key]
		ts = append(ts, convTensor{name, dims, typ, int64(len(b)), func() ([]byte, error) { return b, nil }})
	}
	tab("ROPE_C", fmt.Sprint("ROPE_C_", hd), []uint64{uint64(hd / 2), 32768}, typeI16)
	tab("ROPE_S", fmt.Sprint("ROPE_S_", hd), []uint64{uint64(hd / 2), 32768}, typeI16)
	tab("EXPA", "EXPA", []uint64{intNA}, typeI32)
	tab("EXPF", "EXPF", []uint64{intNF}, typeI32)
	tab("SIG", "SIG", []uint64{2*intNS + 1}, typeI32)

	// metadata, in the reference's order
	var w ggufW
	var kv ggufW
	nkvs := 0
	count := func() { nkvs++ }
	kv.kvStr("general.architecture", "qwen2-intcodec")
	count()
	kv.kvU32("general.alignment", 32)
	count()
	kv.kvU32("intcodec.spec_version", 1)
	count()
	for _, k := range []string{"bos_token_id", "hidden_size", "intermediate_size", "max_position_embeddings",
		"num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta",
		"tie_word_embeddings", "vocab_size"} {
		switch v := cfg[k].(type) {
		case bool:
			kv.kvBool("intcodec."+k, v)
		case json.Number:
			if strings.ContainsAny(v.String(), ".eE") { // a JSON float, as Python reads it
				f, _ := v.Float64()
				kv.kvF64("intcodec."+k, f)
			} else {
				n, _ := v.Int64()
				kv.kvU32("intcodec."+k, uint32(n))
			}
		default:
			return fmt.Errorf("intcodec: config.json: %s missing", k)
		}
		count()
	}
	for _, m := range mats {
		kv.kvI32("intcodec.exp."+m.name, int32(m.e))
		count()
	}
	toks, types, merges, err := tokenizerArrays(filepath.Join(srcDir, "tokenizer.json"), V)
	if err != nil {
		return err
	}
	kv.kvStr("tokenizer.ggml.model", "gpt2")
	kv.kvStr("tokenizer.ggml.pre", "qwen2")
	kv.kvStrs("tokenizer.ggml.tokens", toks)
	kv.kvI32s("tokenizer.ggml.token_type", types)
	kv.kvStrs("tokenizer.ggml.merges", merges)
	nkvs += 5

	w.b.WriteString("GGUF")
	w.u32(3)
	binary.Write(&w.b, binary.LittleEndian, uint64(len(ts)))
	binary.Write(&w.b, binary.LittleEndian, uint64(nkvs))
	w.b.Write(kv.b.Bytes())
	var off uint64
	for _, t := range ts {
		w.str(t.name)
		w.u32(uint32(len(t.dims)))
		for _, d := range t.dims {
			binary.Write(&w.b, binary.LittleEndian, d)
		}
		w.u32(t.typ)
		binary.Write(&w.b, binary.LittleEndian, off)
		off += uint64(t.size+31) / 32 * 32
	}
	for w.b.Len()%32 != 0 {
		w.b.WriteByte(0)
	}

	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<22)
	werr := func() error {
		if _, err := bw.Write(w.b.Bytes()); err != nil {
			return err
		}
		for i, t := range ts {
			b, err := t.data()
			if err != nil {
				return err
			}
			if int64(len(b)) != t.size {
				return fmt.Errorf("intcodec: %s: %d bytes, want %d", t.name, len(b), t.size)
			}
			if _, err := bw.Write(b); err != nil {
				return err
			}
			if pad := (32 - len(b)%32) % 32; pad > 0 {
				bw.Write(make([]byte, pad))
			}
			if i%20 == 0 {
				logf("tensors %d/%d", i+1, len(ts))
			}
		}
		return bw.Flush()
	}()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		os.Remove(tmp)
		return werr
	}
	return os.Rename(tmp, dst)
}

// tokenizerArrays builds GGUF tokenizer arrays from a Hugging Face tokenizer.json, padded to vocabSize.
func tokenizerArrays(path string, vocabSize int) (toks []string, types []int32, merges []string, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, err
	}
	var t struct {
		Model struct {
			Vocab  map[string]int    `json:"vocab"`
			Merges []json.RawMessage `json:"merges"`
		} `json:"model"`
		Added []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, nil, nil, err
	}
	toks, types = make([]string, vocabSize), make([]int32, vocabSize)
	set := make([]bool, vocabSize)
	for s, i := range t.Model.Vocab {
		toks[i], types[i], set[i] = s, 1, true
	}
	for _, a := range t.Added {
		toks[a.ID], set[a.ID] = a.Content, true
		types[a.ID] = 4
		if a.Special {
			types[a.ID] = 3
		}
	}
	for i := range toks {
		if !set[i] {
			toks[i], types[i] = fmt.Sprintf("[PAD%d]", i), 5
		}
	}
	for _, m := range t.Model.Merges {
		var s string
		if json.Unmarshal(m, &s) == nil {
			merges = append(merges, s)
			continue
		}
		var pair []string
		if err := json.Unmarshal(m, &pair); err != nil {
			return nil, nil, nil, err
		}
		merges = append(merges, strings.Join(pair, " "))
	}
	return toks, types, merges, nil
}

// FileSHA256 returns a file's SHA-256 in hex.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, bufio.NewReaderSize(f, 1<<22)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
