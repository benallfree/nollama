package engine

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/benallfree/nollama/audio"
	"github.com/benallfree/nollama/models"
)

func TestGPT2PreTokenize(t *testing.T) {
	for in, want := range map[string][]string{
		"Hello world's  test 123!!": {"Hello", " world", "'s", " ", " test", " 123", "!!"},
		" the quick brown fox.":     {" the", " quick", " brown", " fox", "."},
		"a\n\nb  ":                  {"a", "\n", "\n", "b", "  "}, // \s+(?!\S) backs off to one newline
		"I'll they're":              {"I", "'ll", " they", "'re"},
		"Grüße, 東京":                 {"Grüße", ",", " 東京"},
	} {
		if got := gpt2PreTokenize(in); !reflect.DeepEqual(got, want) {
			t.Errorf("gpt2PreTokenize(%q) = %q, want %q", in, got, want)
		}
	}
}

// synthTokenizer has every byte as a token (ids 0-255) plus a few merges, in English-only layout.
func synthTokenizer(t *testing.T, merges ...string) *whisperTokenizer {
	var toks [][]byte
	for b := 0; b < 256; b++ {
		toks = append(toks, []byte{byte(b)})
	}
	for _, m := range merges {
		toks = append(toks, []byte(m))
	}
	for len(toks) < 50256 {
		toks = append(toks, []byte{0xff, byte(len(toks)), byte(len(toks) >> 8), byte(len(toks) >> 16)}) // filler, never matched
	}
	tok, err := newWhisperTokenizer(toks, 51864)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestWhisperBPE(t *testing.T) {
	tok := synthTokenizer(t, "ab", "bc", "abc", "cd", " a")
	// "abcd": "ab" (256) merges before "bc" (257) and "cd" (259); then "abc" (258) beats "cd"
	if got, want := tok.encode("abcd"), []int{258, 'd'}; !reflect.DeepEqual(got, want) {
		t.Errorf("encode(abcd) = %v, want %v", got, want)
	}
	// "bcd": "bc" (257) before "cd" (259)
	if got, want := tok.encode("bcd"), []int{257, 'd'}; !reflect.DeepEqual(got, want) {
		t.Errorf("encode(bcd) = %v, want %v", got, want)
	}
	// " ab": "ab" (256) outranks " a" (260)
	if got, want := tok.encode(" ab"), []int{' ', 256}; !reflect.DeepEqual(got, want) {
		t.Errorf("encode( ab) = %v, want %v", got, want)
	}
	for _, s := range []string{"abcd bcd cab", "héllo wörld", ""} {
		if got := tok.decode(tok.encode(s)); got != s {
			t.Errorf("round trip %q -> %q", s, got)
		}
	}
}

func TestWhisperSpecialTokens(t *testing.T) {
	toks := make([][]byte, 50257)
	for i := range toks {
		toks[i] = []byte{byte(i), byte(i >> 8), 0xfe}
	}
	for _, c := range []struct {
		vocab                                                 int
		eot, sot, en, translate, transcribe, notimestamps, ts int
		langs                                                 int
	}{
		{51864, 50256, 50257, -1, 50357, 50358, 50362, 50363, 0},      // English-only (tiny.en ...)
		{51865, 50257, 50258, 50259, 50358, 50359, 50363, 50364, 99},  // multilingual
		{51866, 50257, 50258, 50259, 50359, 50360, 50364, 50365, 100}, // large-v3
	} {
		tok, err := newWhisperTokenizer(toks, c.vocab)
		if err != nil {
			t.Fatal(err)
		}
		got := []int{tok.eot, tok.sot, tok.lang("en"), tok.translate, tok.transcribe, tok.notimestamps, tok.timestampBegin, tok.nLangs}
		want := []int{c.eot, c.sot, c.en, c.translate, c.transcribe, c.notimestamps, c.ts, c.langs}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("vocab %d: got %v, want %v", c.vocab, got, want)
		}
		if s := tok.tokenString(tok.timestampBegin + 150); s != "<|3.00|>" {
			t.Errorf("vocab %d: timestamp token %q", c.vocab, s)
		}
	}
	if _, err := newWhisperTokenizer(toks, 52000); err == nil {
		t.Error("an unknown vocabulary size should fail")
	}
}

// whisperModel returns the path of a downloaded Whisper model (NOLLAMA_WHISPER_MODEL, or ggml-tiny.bin under the
// models folder's whisper directory), or skips.
func whisperModel(t *testing.T) string {
	p := os.Getenv("NOLLAMA_WHISPER_MODEL")
	if p == "" {
		p = filepath.Join(models.DefaultDir(), "whisper", "ggml-tiny.bin")
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no whisper model at %s (set NOLLAMA_WHISPER_MODEL)", p)
	}
	return p
}

func TestWhisperModelFile(t *testing.T) {
	w, err := LoadWhisper(whisperModel(t), WhisperOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	// the filterbank in the file is Whisper's mel_filters.npz; audio.MelFilters computes the same
	f := audio.MelFilters(w.hp.mels)
	var maxDiff float64
	for i := range f {
		maxDiff = math.Max(maxDiff, math.Abs(float64(f[i]-w.f.filters[i])))
	}
	if maxDiff > 1e-6 {
		t.Errorf("mel filters differ from the file's by %g", maxDiff)
	}
	// a sentence the reference decoder produced, tokenized the same way
	if w.info.Multilingual {
		want := []int{264, 1702, 6292, 21026, 16704, 670, 264, 14847, 3000, 13}
		if got := w.tok.encode(" the quick brown fox jumps over the lazy dog."); !reflect.DeepEqual(got, want) {
			t.Errorf("encode = %v, want %v", got, want)
		}
	}
	for _, s := range []string{"Hello, world!", " Grüße aus 東京 — 123.", "it's"} {
		if got := w.tok.decode(w.tok.encode(s)); got != s {
			t.Errorf("round trip %q -> %q", s, got)
		}
	}
}

// TestWhisperTranscribeTone runs the whole pipeline on a synthetic signal: no crash, no runaway decoding.
func TestWhisperTranscribeTone(t *testing.T) {
	w, err := LoadWhisper(whisperModel(t), WhisperOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	pcm := make([]float32, audio.SampleRate)
	for i := range pcm {
		pcm[i] = float32(0.1 * math.Sin(2*math.Pi*300*float64(i)/audio.SampleRate))
	}
	tr, err := w.Transcribe(context.Background(), pcm, TranscribeOptions{Language: "en", MaxTokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	if tr.Timings.Windows != 1 || tr.Timings.Tokens > 32 {
		t.Fatalf("timings %+v", tr.Timings)
	}
	t.Logf("tone -> %q", tr.Text)
}

// TestWhisperReference compares the encoder output and the first logits with a numpy port of openai/whisper's
// model (NOLLAMA_WHISPER_REF: a folder with mel.bin, enc.bin and logits0.bin for NOLLAMA_WHISPER_MODEL and
// NOLLAMA_WHISPER_REF_PROMPT's clip; see the spike notes).
func TestWhisperReference(t *testing.T) {
	ref := os.Getenv("NOLLAMA_WHISPER_REF")
	if ref == "" {
		t.Skip("NOLLAMA_WHISPER_REF not set")
	}
	w, err := LoadWhisper(whisperModel(t), WhisperOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	read := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		f := make([]float32, len(b)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return f
	}
	diff := func(a, b []float32) (maxAbs, maxRef float64) {
		for i := range a {
			maxAbs = math.Max(maxAbs, math.Abs(float64(a[i]-b[i])))
			maxRef = math.Max(maxRef, math.Abs(float64(b[i])))
		}
		return
	}
	mel := read("mel.bin")
	enc := w.encode(mel)
	encRef := read("enc.bin")
	d, r := diff(enc, encRef)
	t.Logf("encoder: max abs diff %.3g (values up to %.3g)", d, r)
	dec := w.newDecoder(enc)
	tok := w.tok
	prompt := []int{tok.sot, tok.notimestamps}
	if tok.nLangs > 0 {
		prompt = []int{tok.sot, tok.lang("en"), tok.transcribe, tok.notimestamps}
	}
	logits := dec.step(prompt)
	lref := read("logits0.bin")
	dl, rl := diff(logits, lref)
	t.Logf("first logits: max abs diff %.3g (values up to %.3g)", dl, rl)
	if d > 0.05*r || dl > 0.05*rl {
		t.Fatal("outputs differ from the reference by more than 5%")
	}
}

func BenchmarkWhisperEncode(b *testing.B) {
	p := os.Getenv("NOLLAMA_WHISPER_MODEL")
	if p == "" {
		p = filepath.Join(models.DefaultDir(), "whisper", "ggml-tiny.bin")
	}
	w, err := LoadWhisper(p, WhisperOptions{})
	if err != nil {
		b.Skip(err)
	}
	defer w.Close()
	mel := make([]float32, w.hp.mels*audio.ChunkFrames)
	for i := range mel {
		mel[i] = float32(math.Sin(float64(i) * 0.01))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.encode(mel)
	}
}
