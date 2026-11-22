package engine

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/benallfree/nollama/audio"
)

// Whisper is a loaded OpenAI Whisper speech-recognition model (an encoder-decoder transformer over a log-mel
// spectrogram), read from whisper.cpp's ggml model files (ggml-tiny.bin, ggml-base.bin, ...). It runs on the CPU with
// the same matmul and attention kernels as the text models. Experimental.
//
// A Whisper is safe for one Transcribe at a time; calls on one Whisper are serialized.
type Whisper struct {
	f    *whisperFile
	hp   whisperHParams
	tok  *whisperTokenizer
	pool *pool
	info WhisperInfo
	busy chan struct{}

	conv1w, conv2w *weight // conv kernels as matrices: row o is W[o][i][k], i-major, columns padded to 16
	conv1b, conv2b []float32
	encPos         []float32 // audioCtx rows of audioState
	enc            []whisperLayer
	encLnW, encLnB []float32
	decTok         *weight   // token embeddings, also the output projection
	decPos         []float32 // textCtx rows of textState
	dec            []whisperLayer
	decLnW, decLnB []float32
}

type whisperLayer struct {
	attnLnW, attnLnB []float32
	q, k, v, o       *weight
	qb, vb, ob       []float32
	xLnW, xLnB       []float32 // cross-attention (decoder only)
	xq, xk, xv, xo   *weight
	xqb, xvb, xob    []float32
	mlpLnW, mlpLnB   []float32
	fc1, fc2         *weight
	fc1b, fc2b       []float32
}

// WhisperOptions configures LoadWhisper.
type WhisperOptions struct {
	Threads int // worker threads; 0 = GOMAXPROCS capped at 8
}

// WhisperInfo describes a loaded Whisper model.
type WhisperInfo struct {
	Name         string
	Multilingual bool
	Languages    []string
	Mels         int
	AudioLayers  int
	TextLayers   int
	Width        int // audio_state = text_state
	Vocab        int
	Type         string // weight type, e.g. F16
	FileSize     int64
	Threads      int
	Kernels      string
}

// LoadWhisper opens a whisper.cpp ggml model file (F16, F32 or Q8_0 weights).
func LoadWhisper(path string, opts WhisperOptions) (*Whisper, error) {
	f, err := openWhisperFile(path)
	if err != nil {
		return nil, err
	}
	w, err := newWhisper(f, opts)
	if err != nil {
		munmap(f.mapping)
		return nil, err
	}
	w.info.Name = strings.TrimSuffix(filepath.Base(path), ".bin")
	return w, nil
}

func newWhisper(f *whisperFile, opts WhisperOptions) (*Whisper, error) {
	hp := f.hp
	if hp.audioState != hp.textState {
		return nil, errors.New("engine: whisper audio and text widths differ")
	}
	if hp.mels*f.nFFT != len(f.filters) || f.nFFT != audio.NFFT/2+1 {
		return nil, errors.New("engine: whisper mel filterbank has an unexpected shape")
	}
	tok, err := newWhisperTokenizer(f.vocab, hp.vocab)
	if err != nil {
		return nil, err
	}
	threads := opts.Threads
	if threads <= 0 {
		threads = min(runtime.GOMAXPROCS(0), 8)
	}
	w := &Whisper{f: f, hp: hp, tok: tok, busy: make(chan struct{}, 1)}
	var missing []string
	ten := func(name string) *tensorInfo {
		t := f.tensors[name]
		if t == nil {
			missing = append(missing, name)
		}
		return t
	}
	vec := func(name string) []float32 {
		t := ten(name)
		if t == nil {
			return nil
		}
		return tensorF32(t)
	}
	mat := func(name string) *weight {
		t := ten(name)
		if t == nil {
			return nil
		}
		return tensorWeight(t)
	}
	w.conv1w, w.conv2w = convWeight(ten("encoder.conv1.weight")), convWeight(ten("encoder.conv2.weight"))
	w.conv1b, w.conv2b = vec("encoder.conv1.bias"), vec("encoder.conv2.bias")
	w.encPos = vec("encoder.positional_embedding")
	w.encLnW, w.encLnB = vec("encoder.ln_post.weight"), vec("encoder.ln_post.bias")
	w.decTok = mat("decoder.token_embedding.weight")
	w.decPos = vec("decoder.positional_embedding")
	w.decLnW, w.decLnB = vec("decoder.ln.weight"), vec("decoder.ln.bias")
	block := func(pre string, cross bool) whisperLayer {
		l := whisperLayer{
			attnLnW: vec(pre + "attn_ln.weight"), attnLnB: vec(pre + "attn_ln.bias"),
			q: mat(pre + "attn.query.weight"), qb: vec(pre + "attn.query.bias"),
			k: mat(pre + "attn.key.weight"),
			v: mat(pre + "attn.value.weight"), vb: vec(pre + "attn.value.bias"),
			o: mat(pre + "attn.out.weight"), ob: vec(pre + "attn.out.bias"),
			mlpLnW: vec(pre + "mlp_ln.weight"), mlpLnB: vec(pre + "mlp_ln.bias"),
			fc1: mat(pre + "mlp.0.weight"), fc1b: vec(pre + "mlp.0.bias"),
			fc2: mat(pre + "mlp.2.weight"), fc2b: vec(pre + "mlp.2.bias"),
		}
		if cross {
			l.xLnW, l.xLnB = vec(pre+"cross_attn_ln.weight"), vec(pre+"cross_attn_ln.bias")
			l.xq, l.xqb = mat(pre+"cross_attn.query.weight"), vec(pre+"cross_attn.query.bias")
			l.xk = mat(pre + "cross_attn.key.weight")
			l.xv, l.xvb = mat(pre+"cross_attn.value.weight"), vec(pre+"cross_attn.value.bias")
			l.xo, l.xob = mat(pre+"cross_attn.out.weight"), vec(pre+"cross_attn.out.bias")
		}
		return l
	}
	for i := 0; i < hp.audioLayer; i++ {
		w.enc = append(w.enc, block(fmt.Sprintf("encoder.blocks.%d.", i), false))
	}
	for i := 0; i < hp.textLayer; i++ {
		w.dec = append(w.dec, block(fmt.Sprintf("decoder.blocks.%d.", i), true))
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("engine: whisper model is missing tensors: %s", strings.Join(missing, ", "))
	}
	if w.conv1w == nil || w.conv2w == nil {
		return nil, errors.New("engine: whisper conv weights must be F16 or F32")
	}
	for _, wt := range []*weight{w.decTok, w.enc[0].q, w.enc[0].fc1} {
		if wt.typ != typeF16 && wt.typ != typeF32 && wt.typ != typeQ8_0 {
			return nil, fmt.Errorf("engine: whisper weights of type %s aren't supported (F16, F32, Q8_0)", typeName(wt.typ))
		}
	}
	w.pool = newPool(threads, nil)
	w.info = WhisperInfo{
		Multilingual: tok.nLangs > 0, Languages: whisperLanguages[:tok.nLangs], Mels: hp.mels,
		AudioLayers: hp.audioLayer, TextLayers: hp.textLayer, Width: hp.audioState, Vocab: hp.vocab,
		Type: typeName(w.enc[0].q.typ), FileSize: f.size, Threads: w.pool.n, Kernels: kernelName,
	}
	return w, nil
}

// Close releases the model file.
func (w *Whisper) Close() error {
	w.pool.close()
	return munmap(w.f.mapping)
}

// Info describes the model.
func (w *Whisper) Info() WhisperInfo { return w.info }

func tensorF32(t *tensorInfo) []float32 {
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	out := make([]float32, n)
	switch t.Type {
	case typeF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Data[4*i:]))
		}
	case typeF16:
		for i := range out {
			out[i] = f16(t.Data[2*i:])
		}
	default:
		dequantRow(t.Type, t.Data, out)
	}
	return out
}

func tensorWeight(t *tensorInfo) *weight {
	cols := int(t.Dims[0])
	rows := t.rows()
	return &weight{typ: t.Type, cols: cols, rows: rows, rowBytes: len(t.Data) / rows, data: t.Data}
}

// convWeight reshapes a Conv1d kernel (ggml dims [k, in, out]) into an F16 matrix of out rows, each W[o][i][k] in
// (i, k) order, zero-padded to a multiple of 16 columns so the F16 kernels take it whole.
func convWeight(t *tensorInfo) *weight {
	if t == nil || len(t.Dims) != 3 || (t.Type != typeF16 && t.Type != typeF32) {
		return nil
	}
	k, in, out := int(t.Dims[0]), int(t.Dims[1]), int(t.Dims[2])
	cols := k * in
	pc := (cols + 15) &^ 15
	data := make([]byte, out*pc*2)
	f := tensorF32(t)
	for o := 0; o < out; o++ {
		for c := 0; c < cols; c++ {
			binary.LittleEndian.PutUint16(data[(o*pc+c)*2:], f32ToF16(f[o*cols+c]))
		}
	}
	return &weight{typ: typeF16, cols: pc, rows: out, rowBytes: pc * 2, data: data}
}

// linear computes out[t] = W·x[t] + b for n rows of x (len k each), with the shared matmul kernels.
func (w *Whisper) linear(x []float32, n int, wt *weight, b []float32, out []float32) {
	k := wt.cols
	a := &acts{n: n, k: k, f: x[:n*k]}
	if wt.typ == typeQ8_0 {
		a.q80 = make([]byte, n*k/32*q80aBytes)
		if n >= 4 {
			a.q80x4 = make([]byte, n/4*k/32*q80x4Bytes)
		}
		a.prepare(wt)
		w.pool.parallelFor(n, 4, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				a.quantizeRow(t, 0, k)
			}
		})
	}
	matmul(w.pool, a, 0, mmSeg{wt, out, wt.rows})
	if b != nil {
		rows := wt.rows
		w.pool.parallelFor(n, 16, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				o := out[t*rows : (t+1)*rows]
				for i := range o {
					o[i] += b[i]
				}
			}
		})
	}
}

// layerNorm writes (x-mean)/sqrt(var+1e-5)·g + b for each row of x.
func (w *Whisper) layerNorm(x []float32, n, d int, g, b, out []float32) {
	body := func(lo, hi int) {
		for t := lo; t < hi; t++ {
			xr, o := x[t*d:(t+1)*d], out[t*d:(t+1)*d]
			var s float64
			for _, v := range xr {
				s += float64(v)
			}
			mean := s / float64(d)
			var vs float64
			for _, v := range xr {
				dv := float64(v) - mean
				vs += dv * dv
			}
			inv := 1 / math.Sqrt(vs/float64(d)+1e-5)
			for i, v := range xr {
				o[i] = float32((float64(v)-mean)*inv)*g[i] + b[i]
			}
		}
	}
	if n < 4 {
		body(0, n)
		return
	}
	w.pool.parallelFor(n, 8, body)
}

// gelu applies the exact (erf) GELU in place, as torch.nn.GELU.
func (w *Whisper) gelu(x []float32) {
	const chunk = 4096
	w.pool.parallelFor((len(x)+chunk-1)/chunk, 1, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			for i := c * chunk; i < min((c+1)*chunk, len(x)); i++ {
				v := float64(x[i])
				x[i] = float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
			}
		}
	})
}

// kvHeads holds one attention's keys transposed per head (kT[h][d][ctxP], ctxP a multiple of 16, for scores4) and
// its values by position (v[p][dim], for wsum4).
type kvHeads struct {
	kT   []float32
	v    []float32
	ctxP int
}

func newKVHeads(ctx, dim int) *kvHeads {
	ctxP := (ctx + 15) &^ 15
	return &kvHeads{kT: make([]float32, dim*ctxP), v: make([]float32, ctxP*dim), ctxP: ctxP}
}

// store puts key and value rows for positions pos.. (n rows of dim).
func (kv *kvHeads) store(k, v []float32, pos, n, dim, hd int) {
	for t := 0; t < n; t++ {
		kr := k[t*dim : (t+1)*dim]
		for i, x := range kr {
			h, d := i/hd, i%hd
			kv.kT[(h*hd+d)*kv.ctxP+pos+t] = x
		}
		copy(kv.v[(pos+t)*dim:(pos+t+1)*dim], v[t*dim:(t+1)*dim])
	}
}

// attend computes softmax(q·kᵀ/√hd)·v per head for n query rows (q, out: n rows of dim). Query t sees keys
// [0, ctx) or, when causal, [0, pos+t+1).
func (w *Whisper) attend(q []float32, n int, kv *kvHeads, ctx int, causal bool, pos int, heads, hd int, out []float32) {
	dim := heads * hd
	quads := (n + 3) / 4
	scale := float32(1 / math.Sqrt(float64(hd)))
	ctxP := kv.ctxP
	nw := w.pool.n
	scs, qbs, obs := make([][]float32, nw), make([][]float32, nw), make([][]float32, nw)
	w.pool.parallelForW(heads*quads, 1, func(wk, lo, hi int) {
		if scs[wk] == nil {
			scs[wk], qbs[wk], obs[wk] = make([]float32, 4*ctxP), make([]float32, 4*hd), make([]float32, 4*hd)
		}
		sc, qb, ob := scs[wk], qbs[wk], obs[wk]
		for task := lo; task < hi; task++ {
			h, qi := task/quads, task%quads
			var ts [4]int
			for i := 0; i < 4; i++ {
				ts[i] = min(4*qi+i, n-1)
				copy(qb[i*hd:(i+1)*hd], q[ts[i]*dim+h*hd:ts[i]*dim+(h+1)*hd])
			}
			cmax := ctx
			if causal {
				cmax = pos + ts[3] + 1
			}
			scores4(qb, kv.kT[h*hd*ctxP:], ctxP, (cmax+15)/16, sc, ctxP)
			n4 := (cmax + 3) / 4
			for i := 0; i < 4; i++ {
				c := ctx
				if causal {
					c = pos + ts[i] + 1
				}
				row := sc[i*ctxP : i*ctxP+4*n4]
				mx := float32(math.Inf(-1))
				for j := 0; j < c; j++ {
					row[j] *= scale
					mx = max(mx, row[j])
				}
				var sum float32
				for j := 0; j < c; j++ {
					e := expf(row[j] - mx)
					row[j] = e
					sum += e
				}
				inv := 1 / sum
				for j := 0; j < c; j++ {
					row[j] *= inv
				}
				clear(row[c:])
			}
			wsum4(sc, ctxP, kv.v[h*hd:], dim, n4, ob, hd, hd)
			for i := 0; i < 4; i++ {
				if 4*qi+i < n {
					copy(out[ts[i]*dim+h*hd:ts[i]*dim+(h+1)*hd], ob[i*hd:(i+1)*hd])
				}
			}
		}
	})
}

// encode runs one 30 s window of log-mel frames (mels rows of audio.ChunkFrames) through the encoder, returning
// audioCtx rows of audioState.
func (w *Whisper) encode(mel []float32) []float32 {
	hp := w.hp
	frames := audio.ChunkFrames
	dim, n := hp.audioState, hp.audioCtx // n = frames/2
	// conv1 (k=3, pad 1) as a matmul over im2col rows: x1[t][i*3+k] = mel[i][t+k-1]
	c1 := w.conv1w.cols
	x1 := make([]float32, frames*c1)
	w.pool.parallelFor(frames, 64, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			r := x1[t*c1:]
			for i := 0; i < hp.mels; i++ {
				for k := 0; k < 3; k++ {
					if s := t + k - 1; s >= 0 && s < frames {
						r[i*3+k] = mel[i*frames+s]
					}
				}
			}
		}
	})
	h1 := make([]float32, frames*dim)
	w.linear(x1, frames, w.conv1w, w.conv1b, h1)
	w.gelu(h1)
	// conv2 (k=3, stride 2, pad 1): x2[t][i*3+k] = h1[2t+k-1][i]
	c2 := w.conv2w.cols
	x2 := make([]float32, n*c2)
	w.pool.parallelFor(n, 16, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			r := x2[t*c2:]
			for k := 0; k < 3; k++ {
				s := 2*t + k - 1
				if s < 0 || s >= frames {
					continue
				}
				src := h1[s*dim : (s+1)*dim]
				for i, v := range src {
					r[i*3+k] = v
				}
			}
		}
	})
	x := make([]float32, n*dim)
	w.linear(x2, n, w.conv2w, w.conv2b, x)
	w.gelu(x)
	for i := range x {
		x[i] += w.encPos[i]
	}
	heads := hp.audioHead
	hd := dim / heads
	xn := make([]float32, n*dim)
	q, k, v, att := make([]float32, n*dim), make([]float32, n*dim), make([]float32, n*dim), make([]float32, n*dim)
	ff := make([]float32, n*w.enc[0].fc1.rows)
	tmp := make([]float32, n*dim)
	kv := newKVHeads(n, dim)
	for li := range w.enc {
		l := &w.enc[li]
		w.layerNorm(x, n, dim, l.attnLnW, l.attnLnB, xn)
		w.linear(xn, n, l.q, l.qb, q)
		w.linear(xn, n, l.k, nil, k)
		w.linear(xn, n, l.v, l.vb, v)
		kv.store(k, v, 0, n, dim, hd)
		w.attend(q, n, kv, n, false, 0, heads, hd, att)
		w.linear(att, n, l.o, l.ob, tmp)
		addInto(x, tmp)
		w.layerNorm(x, n, dim, l.mlpLnW, l.mlpLnB, xn)
		w.linear(xn, n, l.fc1, l.fc1b, ff)
		w.gelu(ff)
		w.linear(ff, n, l.fc2, l.fc2b, tmp)
		addInto(x, tmp)
	}
	w.layerNorm(x, n, dim, w.encLnW, w.encLnB, x)
	return x
}

func addInto(x, y []float32) {
	for i, v := range y {
		x[i] += v
	}
}

// whisperDecoder is one decoding pass's state: the cross-attention keys and values from the encoder output and the
// self-attention cache.
type whisperDecoder struct {
	w     *Whisper
	cross []*kvHeads
	self  []*kvHeads
	pos   int
}

func (w *Whisper) newDecoder(encOut []float32) *whisperDecoder {
	hp := w.hp
	dim, n := hp.textState, hp.audioCtx
	hd := dim / hp.textHead
	d := &whisperDecoder{w: w}
	k, v := make([]float32, n*dim), make([]float32, n*dim)
	for li := range w.dec {
		l := &w.dec[li]
		w.linear(encOut, n, l.xk, nil, k)
		w.linear(encOut, n, l.xv, l.xvb, v)
		c := newKVHeads(n, dim)
		c.store(k, v, 0, n, dim, hd)
		d.cross = append(d.cross, c)
		d.self = append(d.self, newKVHeads(hp.textCtx, dim))
	}
	return d
}

// reset forgets the decoded tokens, keeping the cross-attention state.
func (d *whisperDecoder) reset() { d.pos = 0 }

// step runs tokens at positions pos.. and returns the logits for the last one.
func (d *whisperDecoder) step(tokens []int) []float32 {
	w := d.w
	hp := w.hp
	dim, heads := hp.textState, hp.textHead
	hd := dim / heads
	n := len(tokens)
	x := make([]float32, n*dim)
	for t, id := range tokens {
		dequantRow(w.decTok.typ, w.decTok.row(id), x[t*dim:(t+1)*dim])
		addInto(x[t*dim:(t+1)*dim], w.decPos[(d.pos+t)*dim:(d.pos+t+1)*dim])
	}
	xn := make([]float32, n*dim)
	q, k, v, att, tmp := make([]float32, n*dim), make([]float32, n*dim), make([]float32, n*dim), make([]float32, n*dim), make([]float32, n*dim)
	ff := make([]float32, n*w.dec[0].fc1.rows)
	for li := range w.dec {
		l := &w.dec[li]
		w.layerNorm(x, n, dim, l.attnLnW, l.attnLnB, xn)
		w.linear(xn, n, l.q, l.qb, q)
		w.linear(xn, n, l.k, nil, k)
		w.linear(xn, n, l.v, l.vb, v)
		d.self[li].store(k, v, d.pos, n, dim, hd)
		w.attend(q, n, d.self[li], 0, true, d.pos, heads, hd, att)
		w.linear(att, n, l.o, l.ob, tmp)
		addInto(x, tmp)

		w.layerNorm(x, n, dim, l.xLnW, l.xLnB, xn)
		w.linear(xn, n, l.xq, l.xqb, q)
		w.attend(q, n, d.cross[li], hp.audioCtx, false, 0, heads, hd, att)
		w.linear(att, n, l.xo, l.xob, tmp)
		addInto(x, tmp)

		w.layerNorm(x, n, dim, l.mlpLnW, l.mlpLnB, xn)
		w.linear(xn, n, l.fc1, l.fc1b, ff)
		w.gelu(ff)
		w.linear(ff, n, l.fc2, l.fc2b, tmp)
		addInto(x, tmp)
	}
	d.pos += n
	last := make([]float32, dim)
	w.layerNorm(x[(n-1)*dim:], 1, dim, w.decLnW, w.decLnB, last)
	logits := make([]float32, hp.vocab)
	w.linear(last, 1, w.decTok, nil, logits)
	return logits
}

// TranscribeOptions configures Transcribe.
type TranscribeOptions struct {
	// Language is the spoken language's code ("en", "de", ...); "" detects it from the first 30 s window.
	// English-only models ignore it.
	Language string
	// Translate makes a multilingual model write English instead of transcribing.
	Translate bool
	// NoTimestamps decodes plain text: each 30 s window on its own, so a word on a boundary can be split or lost.
	// By default the model also predicts segment timestamps, and the next window starts where the last whole segment
	// ended, as openai/whisper's transcribe does.
	NoTimestamps bool
	// MaxTokens caps the tokens per 30 s window (default and maximum: half the text context, 224).
	MaxTokens int
	// OnSegment, when set, is called with each segment as it is decoded.
	OnSegment func(Segment)
}

// Segment is the text of one stretch of audio.
type Segment struct {
	Start, End time.Duration
	Text       string
	Tokens     []int
}

// Transcript is Transcribe's result.
type Transcript struct {
	Text     string
	Language string
	Duration time.Duration // of the audio
	Segments []Segment
	Timings  TranscribeTimings
}

// TranscribeTimings says where Transcribe's time went.
type TranscribeTimings struct {
	Mel, Encode, Decode time.Duration
	Windows, Tokens     int
}

const whisperFrame = 10 * time.Millisecond // one mel frame; a timestamp step is two

// Transcribe turns 16 kHz mono samples (see audio.DecodeWAV) into text, 30 s window by window.
func (w *Whisper) Transcribe(ctx context.Context, pcm []float32, o TranscribeOptions) (*Transcript, error) {
	select {
	case w.busy <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-w.busy }()
	tok, hp := w.tok, w.hp
	maxTok := hp.textCtx / 2
	if o.MaxTokens > 0 {
		maxTok = min(o.MaxTokens, maxTok)
	}
	tr := Transcript{Duration: time.Duration(len(pcm)) * time.Second / audio.SampleRate}
	t0 := time.Now()
	// pad so the last window is whole, as whisper.cpp does (silence, not zeros in the log-mel domain)
	padded := make([]float32, len(pcm)+audio.ChunkSamples)
	copy(padded, pcm)
	mel := audio.LogMel(padded, w.f.filters, hp.mels)
	tr.Timings.Mel = time.Since(t0)
	frames := len(pcm) / audio.HopLength
	var texts []string
	emit := func(seg Segment) {
		seg.Text = strings.TrimSpace(seg.Text)
		if seg.Text == "" {
			return
		}
		tr.Segments = append(tr.Segments, seg)
		texts = append(texts, seg.Text)
		if o.OnSegment != nil {
			o.OnSegment(seg)
		}
	}
	for seek := 0; seek < max(frames, 1); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		size := min(audio.ChunkFrames, max(frames-seek, 1)) // frames of real audio in this window
		chunk := audio.Chunk(mel, hp.mels, seek, 0)
		te := time.Now()
		enc := w.encode(chunk)
		tr.Timings.Encode += time.Since(te)
		td := time.Now()
		dec := w.newDecoder(enc)
		prompt := []int{tok.sot}
		if tok.nLangs > 0 {
			lang := o.Language
			if lang == "" {
				if tr.Language == "" {
					tr.Language = w.detectLanguage(dec)
				}
				lang = tr.Language
			}
			id := tok.lang(lang)
			if id < 0 {
				return nil, fmt.Errorf("engine: whisper doesn't know language %q", lang)
			}
			tr.Language = lang
			task := tok.transcribe
			if o.Translate {
				task = tok.translate
			}
			prompt = append(prompt, id, task)
		} else {
			tr.Language = "en"
		}
		if o.NoTimestamps {
			prompt = append(prompt, tok.notimestamps)
		}
		ids := w.greedy(ctx, dec, prompt, maxTok, !o.NoTimestamps)
		tr.Timings.Decode += time.Since(td)
		tr.Timings.Windows++
		tr.Timings.Tokens += len(ids)
		at := func(f int) time.Duration { return time.Duration(seek+f) * whisperFrame }
		if o.NoTimestamps {
			emit(Segment{Start: at(0), End: at(size), Text: tok.decode(ids), Tokens: ids})
			seek += size
			continue
		}
		segs, advance := w.splitTimestamps(ids, size)
		for _, sg := range segs {
			emit(Segment{Start: at(sg.from), End: at(sg.to), Text: tok.decode(sg.ids), Tokens: sg.ids})
		}
		seek += max(advance, 1)
	}
	tr.Text = strings.Join(texts, " ")
	return &tr, nil
}

type tsSegment struct {
	from, to int // frames from the window's start
	ids      []int
}

// splitTimestamps cuts one window's tokens into segments at each pair of consecutive timestamps and returns how many
// frames to advance, as openai/whisper's transcribe does: past the whole window when the text ends with a lone
// timestamp (everything said is complete), else only up to the last pair, so an unfinished sentence is decoded again
// in the next window.
func (w *Whisper) splitTimestamps(ids []int, size int) ([]tsSegment, int) {
	tok := w.tok
	ts := func(i int) bool { return tok.isTimestamp(ids[i]) }
	frame := func(id int) int { return 2 * (id - tok.timestampBegin) }
	n := len(ids)
	singleEnding := n >= 2 && ts(n-1) && !ts(n-2)
	var cuts []int // indices i where ids[i-1] and ids[i] are both timestamps
	for i := 1; i < n; i++ {
		if ts(i) && ts(i-1) {
			cuts = append(cuts, i)
		}
	}
	if len(cuts) == 0 { // no pair: one segment for the window
		var text []int
		end := size
		for i, id := range ids {
			if tok.isTimestamp(id) {
				if i == n-1 && frame(id) > 0 {
					end = frame(id)
				}
				continue
			}
			text = append(text, id)
		}
		return []tsSegment{{0, end, text}}, size
	}
	if singleEnding {
		cuts = append(cuts, n)
	}
	var segs []tsSegment
	last := 0
	for _, c := range cuts {
		sl := ids[last:c]
		if len(sl) >= 2 && tok.isTimestamp(sl[0]) {
			var text []int
			for _, id := range sl {
				if !tok.isTimestamp(id) {
					text = append(text, id)
				}
			}
			segs = append(segs, tsSegment{frame(sl[0]), frame(sl[len(sl)-1]), text})
		}
		last = c
	}
	if singleEnding {
		return segs, size
	}
	return segs, frame(ids[cuts[len(cuts)-1]-1])
}

// detectLanguage feeds <|startoftranscript|> and picks the most likely language token.
func (w *Whisper) detectLanguage(d *whisperDecoder) string {
	tok := w.tok
	d.reset()
	logits := d.step([]int{tok.sot})
	d.reset()
	best, bi := float32(math.Inf(-1)), 0
	for i := 0; i < tok.nLangs; i++ {
		if v := logits[tok.langBegin+i]; v > best {
			best, bi = v, i
		}
	}
	return whisperLanguages[bi]
}

// greedy decodes from prompt until end of text, returning the tokens after the prompt. Like openai/whisper's decoder
// it suppresses special tokens other than end of text and timestamps, and a blank or end of text as the first token.
// With timestamps it applies whisper's timestamp rules: the text starts with a timestamp (at most 1 s in),
// timestamps come in pairs and never go backwards, and a timestamp is forced when timestamps together are more likely
// than any one text token.
func (w *Whisper) greedy(ctx context.Context, d *whisperDecoder, prompt []int, maxTok int, timestamps bool) []int {
	tok := w.tok
	d.reset()
	logits := d.step(prompt)
	var out []int
	space := -1
	if id, ok := tok.ids[" "]; ok {
		space = id
	}
	ninf := float32(math.Inf(-1))
	beg := tok.timestampBegin
	for len(out) < maxTok && ctx.Err() == nil {
		for id := tok.eot + 1; id < beg; id++ {
			logits[id] = ninf
		}
		if len(out) == 0 {
			logits[tok.eot] = ninf
			if space >= 0 {
				logits[space] = ninf
			}
		}
		if !timestamps {
			for id := beg; id < len(logits); id++ {
				logits[id] = ninf
			}
		} else {
			w.timestampRules(logits, out)
		}
		best, bi := ninf, -1
		for id, v := range logits {
			if v > best {
				best, bi = v, id
			}
		}
		if bi == tok.eot || bi < 0 {
			break
		}
		out = append(out, bi)
		if d.pos >= w.hp.textCtx {
			break
		}
		logits = d.step([]int{bi})
	}
	return out
}

// timestampRules is openai/whisper's ApplyTimestampRules for greedy decoding.
func (w *Whisper) timestampRules(logits []float32, out []int) {
	tok := w.tok
	beg := tok.timestampBegin
	ninf := float32(math.Inf(-1))
	suppress := func(lo, hi int) {
		for id := max(lo, 0); id < min(hi, len(logits)); id++ {
			logits[id] = ninf
		}
	}
	n := len(out)
	lastTS := n >= 1 && tok.isTimestamp(out[n-1])
	penultTS := n < 2 || tok.isTimestamp(out[n-2])
	if lastTS {
		if penultTS { // a pair just closed: text (or end) comes next
			suppress(beg, len(logits))
		} else { // an opening timestamp: its closing timestamp or the end comes next
			suppress(0, tok.eot)
		}
	}
	lastT := -1
	for i := n - 1; i >= 0; i-- {
		if tok.isTimestamp(out[i]) {
			lastT = out[i]
			break
		}
	}
	if lastT >= 0 {
		if !(lastTS && !penultTS) {
			lastT++ // a new segment starts strictly after the last one ends
		}
		suppress(beg, lastT)
	}
	if n == 0 {
		suppress(0, beg)                // the text starts with a timestamp
		suppress(beg+50+1, len(logits)) // max_initial_timestamp = 1 s
	}
	// timestamps together more likely than any text token: force a timestamp
	mx := ninf
	for _, v := range logits {
		mx = max(mx, v)
	}
	var sum float64
	for _, v := range logits {
		if v > ninf {
			sum += math.Exp(float64(v - mx))
		}
	}
	lse := float64(mx) + math.Log(sum)
	var tsSum float64
	for _, v := range logits[beg:] {
		if v > ninf {
			tsSum += math.Exp(float64(v - mx))
		}
	}
	textMax := ninf
	for _, v := range logits[:beg] {
		textMax = max(textMax, v)
	}
	if tsSum > 0 && float64(mx)+math.Log(tsSum)-lse > float64(textMax)-lse {
		suppress(0, beg)
	}
}
