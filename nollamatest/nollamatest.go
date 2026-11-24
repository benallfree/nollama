// Package nollamatest helps you test code that uses nollama, with no model download.
//
// It offers two kinds of model, both loaded through the real engine:
//
//   - a tiny model (TinyGGUF, Load): a real Qwen3 GGUF with random weights, about 1 MB, built in memory in a few
//     milliseconds. It runs the whole engine (tokenizer, forward pass, sampling, JSON-schema constraints), so it
//     catches plumbing bugs, but what it writes is a jumble of English words.
//   - a scripted model (Scripted): the same file, but each answer comes from a function you write. Use it to test
//     your own logic: "when the model says billing, the ticket goes to the billing queue".
//
// NewServer runs either behind nollama's real HTTP server, speaking Ollama's and OpenAI's APIs, for code that talks to
// nollama (or Ollama) over HTTP.
package nollamatest

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/benallfree/nollama/catalog"
	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/internal/tinymodel"
	"github.com/benallfree/nollama/server"
)

// TinyConfig shapes the tiny model. The zero value is the default: 2 layers, 64-wide, about 1 MB.
//
//   - Seed: the random weights' seed (same seed, same model).
//   - Layers (default 2), Dim (a multiple of 16, default 64), Context (default 2048).
//   - EndTurns lets the model end its answer on its own. By default it can't, so an unconstrained answer runs to max
//     tokens ("length"), and a constrained one ends when the constraint is done.
type TinyConfig = tinymodel.Config

// TinyGGUF returns a complete Qwen3 GGUF file with random weights: a real model the engine loads and runs, with real
// tokenization, sampling and constraints, small enough to build in every test. It writes a jumble of English words,
// the same every run on a given machine at temperature 0 (floating-point details differ between CPUs, so don't
// compare exact text across machines).
func TinyGGUF(c TinyConfig) []byte { return tinymodel.GGUF(c) }

// WriteTinyGGUF writes TinyGGUF(c) to path.
func WriteTinyGGUF(path string, c TinyConfig) error { return tinymodel.Write(path, c) }

// Prompt is what a scripted model was asked: the raw prompt text, and its chat messages when it is a chat prompt
// (Qwen3's ChatML template, which engine.ChatPrompt and the server's chat endpoints use).
type Prompt struct {
	Raw      string
	Messages []engine.Message // nil when Raw isn't a chat prompt
}

// Last returns the content of the last message with the given role ("user", "system", "assistant"), or "".
func (p Prompt) Last(role string) string {
	for i := len(p.Messages) - 1; i >= 0; i-- {
		if p.Messages[i].Role == role {
			return p.Messages[i].Content
		}
	}
	return ""
}

// ParsePrompt reads the chat messages back out of a prompt built with engine.ChatPrompt.
func ParsePrompt(raw string) Prompt {
	p := Prompt{Raw: raw}
	if !strings.Contains(raw, "<|im_start|>") {
		return p
	}
	for _, turn := range strings.Split(raw, "<|im_start|>")[1:] {
		body, closed := strings.CutSuffix(strings.TrimSuffix(turn, "\n"), "<|im_end|>")
		if !closed {
			continue // the open assistant turn the model is asked to write
		}
		role, content, _ := strings.Cut(body, "\n")
		if role == "assistant" {
			content = strings.TrimPrefix(content, "<think>\n\n</think>\n\n")
		}
		p.Messages = append(p.Messages, engine.Message{Role: role, Content: content})
	}
	return p
}

// Responder writes a scripted model's answer to a prompt. It may be called from several goroutines at once.
type Responder func(p Prompt) string

// Reply is a Responder that always answers text.
func Reply(text string) Responder { return func(Prompt) string { return text } }

// TinyPath writes the tiny model into a temporary folder that t removes, and returns its path.
func TinyPath(t testing.TB, c TinyConfig) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := WriteTinyGGUF(p, c); err != nil {
		t.Fatal(err)
	}
	return p
}

// Load loads the tiny model (closed when t ends).
func Load(t testing.TB, c TinyConfig) *engine.Model {
	t.Helper()
	return load(t, TinyPath(t, c), engine.LoadOptions{Threads: 1})
}

// Scripted loads a model that answers with r (closed when t ends). Generate keeps its contract: the answer is
// tokenized and cut at MaxTokens or a stop string, streamed in pieces, and checked against the Constraint (an answer
// that doesn't fit it is an error, so a test finds out its script and its schema disagree).
func Scripted(t testing.TB, r Responder) *engine.Model {
	t.Helper()
	return load(t, TinyPath(t, TinyConfig{}), engine.LoadOptions{Threads: 1, Respond: respond(r)})
}

func respond(r Responder) func(string) string {
	return func(raw string) string { return r(ParsePrompt(raw)) }
}

func load(t testing.TB, path string, o engine.LoadOptions) *engine.Model {
	t.Helper()
	m, err := engine.Load(path, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// Server is nollama's HTTP server running a test model.
type Server struct {
	*httptest.Server
	Nollama *server.Server
	// Model is a name to send in requests. The test model answers to every name in nollama's catalog ("qwen3-1.7b",
	// and Ollama's spelling "qwen3:1.7b"), so code that names a real model works unchanged.
	Model string
}

// Option configures NewServer.
type Option func(*serverOptions)

type serverOptions struct {
	tiny    TinyConfig
	respond Responder
	config  func(*server.Config)
}

// WithResponder makes the server's model answer with r instead of running the tiny model's weights.
func WithResponder(r Responder) Option { return func(o *serverOptions) { o.respond = r } }

// WithTiny shapes the tiny model.
func WithTiny(c TinyConfig) Option { return func(o *serverOptions) { o.tiny = c } }

// WithConfig changes the server's config before it starts (ModelsDir and Load are already set).
func WithConfig(f func(*server.Config)) Option { return func(o *serverOptions) { o.config = f } }

// NewServer starts nollama's server on a test model and stops it when t ends. Point an Ollama or OpenAI client at
// its URL (OpenAI clients at URL + "/v1").
func NewServer(t testing.TB, opts ...Option) *Server {
	t.Helper()
	var o serverOptions
	for _, f := range opts {
		f(&o)
	}
	dir := t.TempDir()
	data := TinyGGUF(o.tiny)
	for _, k := range catalog.Catalog {
		if err := os.WriteFile(filepath.Join(dir, k.Name+".gguf"), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cfg := server.Config{ModelsDir: dir, Threads: 1}
	if o.config != nil {
		o.config(&cfg)
	}
	cfg.ModelsDir = dir
	if o.respond != nil {
		r := respond(o.respond)
		cfg.Load = func(path string, lo engine.LoadOptions) (*engine.Model, error) {
			lo.Respond = r
			return engine.Load(path, lo)
		}
	}
	s := server.New(cfg)
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); _ = s.Close() })
	return &Server{Server: ts, Nollama: s, Model: catalog.Catalog[0].Name}
}
