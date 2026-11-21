package engine

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	normalize(v)
	if math.Abs(float64(v[0])-0.6) > 1e-7 || math.Abs(float64(v[1])-0.8) > 1e-7 {
		t.Errorf("normalize(3,4) = %v", v)
	}
	z := []float32{0, 0, 0}
	normalize(z) // no NaNs
	for _, x := range z {
		if x != 0 {
			t.Errorf("normalize(0) = %v", z)
		}
	}
	if c := Cosine([]float32{1, 0}, []float32{2, 2}); math.Abs(c-math.Sqrt2/2) > 1e-9 {
		t.Errorf("Cosine = %v", c)
	}
}

func TestEmbedSeq(t *testing.T) {
	m := &Model{hp: hparams{ctx: 5}, addEOS: true, eosID: 9}
	in := []int{1, 2, 3}
	got, err := m.embedSeq(in, 0, false)
	if err != nil || !slices.Equal(got, []int{1, 2, 3, 9}) {
		t.Errorf("embedSeq = %v, %v; want the end-of-text token appended", got, err)
	}
	if !slices.Equal(in, []int{1, 2, 3}) {
		t.Errorf("embedSeq changed its input: %v", in)
	}
	if got, err := m.embedSeq(nil, 0, false); err != nil || !slices.Equal(got, []int{9}) {
		t.Errorf("empty text = %v, %v", got, err)
	}
	long := []int{1, 2, 3, 4, 5, 6}
	if _, err := m.embedSeq(long, 0, false); err == nil {
		t.Error("too long without truncate: no error")
	}
	if got, _ := m.embedSeq(long, 0, true); !slices.Equal(got, []int{1, 2, 3, 4, 9}) {
		t.Errorf("truncated = %v, want the start kept and the end-of-text token last", got)
	}
	if got, _ := m.embedSeq(long, 3, true); !slices.Equal(got, []int{1, 2, 9}) {
		t.Errorf("limit 3 = %v", got)
	}
	m.addEOS = false
	if got, _ := m.embedSeq(long, 0, true); !slices.Equal(got, []int{1, 2, 3, 4, 5}) {
		t.Errorf("truncated without EOS = %v", got)
	}
	if _, err := m.embedSeq(nil, 0, false); err == nil {
		t.Error("empty text without EOS: no error")
	}
}

// loadEmbedModel loads the Qwen3-Embedding GGUF named by EMBED_GGUF, or skips.
func loadEmbedModel(t *testing.T) *Model {
	t.Helper()
	p := os.Getenv("EMBED_GGUF")
	if p == "" {
		t.Skip("EMBED_GGUF not set (path to Qwen3-Embedding-0.6B-Q8_0.gguf)")
	}
	m, err := Load(p, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// The model card's example: two queries (with the retrieval instruction) against two documents. Its reference scores,
// from the full-precision model in Transformers, are [[0.7646, 0.1414], [0.1355, 0.6000]].
func TestEmbedModelCard(t *testing.T) {
	m := loadEmbedModel(t)
	if m.Info().Pooling != "last" {
		t.Errorf("Pooling = %q, want last", m.Info().Pooling)
	}
	texts := []string{
		EmbedQuery("", "What is the capital of China?"),
		EmbedQuery("", "Explain gravity"),
		"The capital of China is Beijing.",
		"Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
	}
	es, err := m.Embed(context.Background(), texts, EmbedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range es {
		if len(e.Vector) != 1024 {
			t.Fatalf("text %d: %d values, want 1024", i, len(e.Vector))
		}
		if n := Cosine(e.Vector, e.Vector); math.Abs(math.Sqrt(dot(e.Vector, e.Vector))-1) > 1e-5 || n < 0.9999 {
			t.Errorf("text %d: not unit length", i)
		}
	}
	want := [2][2]float64{{0.7646, 0.1414}, {0.1355, 0.6000}}
	for q := 0; q < 2; q++ {
		for d := 0; d < 2; d++ {
			got := dot(es[q].Vector, es[2+d].Vector)
			t.Logf("query %d · doc %d = %.4f (reference %.4f)", q, d, got, want[q][d])
			if math.Abs(got-want[q][d]) > 0.02 { // 8-bit weights against the reference's full precision
				t.Errorf("query %d · doc %d = %.4f, want %.4f ± 0.02", q, d, got, want[q][d])
			}
		}
	}
}

func TestEmbedSimilarity(t *testing.T) {
	m := loadEmbedModel(t)
	texts := []string{
		"A man is playing a guitar on stage.",
		"Someone performs music with a guitar in front of an audience.",
		"The stock market fell sharply after the interest rate announcement.",
		"How do I bake sourdough bread at home?",
	}
	es, err := m.Embed(context.Background(), texts, EmbedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	similar := dot(es[0].Vector, es[1].Vector)
	for _, j := range []int{2, 3} {
		if un := dot(es[0].Vector, es[j].Vector); un >= similar {
			t.Errorf("unrelated text %d scores %.3f, the paraphrase only %.3f", j, un, similar)
		}
	}
	t.Logf("paraphrase %.3f, unrelated %.3f %.3f", similar, dot(es[0].Vector, es[2].Vector), dot(es[0].Vector, es[3].Vector))
}

// Last-token pooling must not depend on how the text is split into batches, or on its neighbours in the request.
func TestEmbedPoolingConsistent(t *testing.T) {
	m := loadEmbedModel(t)
	ctx := context.Background()
	text := "The quick brown fox jumps over the lazy dog."
	alone, err := m.Embed(ctx, []string{text}, EmbedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := m.Embed(ctx, []string{"something much longer that comes first in the request, to use the state", text}, EmbedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if c := dot(alone[0].Vector, batch[1].Vector); c < 0.9999 {
		t.Errorf("same text in a batch: cosine %.6f", c)
	}
	// one token at a time (the decode path) gives the same hidden state as one batch
	seq, _ := m.embedSeq(m.Tokenize(text), 0, false)
	if alone[0].Tokens != len(seq) || seq[len(seq)-1] != m.eosID {
		t.Fatalf("Tokens = %d, sequence %v: want the end-of-text token appended", alone[0].Tokens, seq)
	}
	s := m.newState(len(seq), 1)
	s.hidden = make([]float32, m.hp.dim)
	for pos := range seq {
		m.forward(s, seq[pos:pos+1], pos)
	}
	normalize(s.hidden)
	if c := dot(alone[0].Vector, s.hidden); c < 0.999 {
		t.Errorf("token by token vs batch: cosine %.6f", c)
	}
	// Matryoshka: the first 256 values, normalized again
	short, err := m.Embed(ctx, []string{text}, EmbedOptions{Dimensions: 256})
	if err != nil {
		t.Fatal(err)
	}
	want := slices.Clone(alone[0].Vector[:256])
	normalize(want)
	if len(short[0].Vector) != 256 || dot(short[0].Vector, want) < 0.99999 {
		t.Errorf("Dimensions 256: %d values", len(short[0].Vector))
	}
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Against Ollama (llama.cpp's kernels) on the same GGUF file: testdata/embed_ref_ollama.json.
func TestEmbedMatchesOllama(t *testing.T) {
	m := loadEmbedModel(t)
	b, err := os.ReadFile("testdata/embed_ref_ollama.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		PromptEvalCount int         `json:"prompt_eval_count"`
		Inputs          []string    `json:"inputs"`
		Embeddings      [][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	es, err := m.Embed(context.Background(), ref.Inputs, EmbedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	tokens := 0
	for i, e := range es {
		tokens += e.Tokens
		c := dot(e.Vector, ref.Embeddings[i])
		var maxDiff float64
		for j := range e.Vector {
			maxDiff = max(maxDiff, math.Abs(float64(e.Vector[j]-ref.Embeddings[i][j])))
		}
		t.Logf("input %d: cosine %.6f, largest difference %.5f", i, c, maxDiff)
		if c < 0.999 || maxDiff > 0.01 {
			t.Errorf("input %d: cosine %.6f, largest difference %.5f; want >= 0.999 and <= 0.01", i, c, maxDiff)
		}
	}
	if tokens != ref.PromptEvalCount {
		t.Errorf("tokens %d, Ollama read %d", tokens, ref.PromptEvalCount)
	}
}

// The tiny random-weight model runs the whole embedding path in every test run, with no download.
func TestEmbedTiny(t *testing.T) {
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := tinymodel.Write(p, tinymodel.Config{}); err != nil {
		t.Fatal(err)
	}
	m, err := Load(p, LoadOptions{Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	texts := []string{"hello world", "a second, somewhat longer text to embed", "hello world"}
	es, err := m.Embed(ctx, texts, EmbedOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range es {
		if len(e.Vector) != m.Info().Embedding || math.Abs(dot(e.Vector, e.Vector)-1) > 1e-5 || e.Tokens == 0 {
			t.Errorf("text %d: %d values, |v|² %.6f, %d tokens", i, len(e.Vector), dot(e.Vector, e.Vector), e.Tokens)
		}
	}
	if c := dot(es[0].Vector, es[2].Vector); c < 0.99999 {
		t.Errorf("same text twice in a batch: cosine %.6f", c)
	}
	if c := dot(es[0].Vector, es[1].Vector); c > 0.9999 {
		t.Errorf("different texts, same vector")
	}
	short, err := m.Embed(ctx, texts[:1], EmbedOptions{Dimensions: 16})
	if err != nil || len(short[0].Vector) != 16 || math.Abs(dot(short[0].Vector, short[0].Vector)-1) > 1e-5 {
		t.Errorf("Dimensions 16: %v", err)
	}
	if _, err := m.Embed(ctx, texts[:1], EmbedOptions{Dimensions: 10000}); err == nil {
		t.Error("Dimensions past the model's: no error")
	}
	if _, err := m.Embed(ctx, []string{strings.Repeat("word ", 50)}, EmbedOptions{MaxTokens: 8}); err == nil {
		t.Error("too long without Truncate: no error")
	}
	cut, err := m.Embed(ctx, []string{strings.Repeat("word ", 50)}, EmbedOptions{MaxTokens: 8, Truncate: true})
	if err != nil || cut[0].Tokens != 8 {
		t.Errorf("truncated: %v, %v", cut, err)
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := m.Embed(cctx, texts, EmbedOptions{}); err == nil {
		t.Error("canceled: no error")
	}
}
