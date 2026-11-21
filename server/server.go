// Package server answers Ollama's API (/api/...) and OpenAI's (/v1/...) with models the engine runs in this
// process: no Ollama, no llama.cpp, no cgo.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/models"
	"github.com/benallfree/nollama/schema"
)

// OllamaVersion is the Ollama API version /api/version reports. Some clients check it before they use a feature
// (structured outputs need 0.5), so it names the Ollama release whose API nollama follows.
const OllamaVersion = "0.12.0"

// Config sets up a Server.
type Config struct {
	ModelsDir   string        // where models are downloaded and listed (default models.DefaultDir())
	Threads     int           // engine threads (0: the engine's default)
	PromptCache int64         // bytes for the engine's prompt cache (0: off)
	Parallel    int           // answers written at once (default 1); the rest wait their turn
	KeepAlive   time.Duration // how long an idle model stays loaded (default 5m; negative: forever)
	NumCtx      int           // context window when a request doesn't set num_ctx (default 4096, capped at the model's)
	Client      *http.Client  // for pulls (default http.DefaultClient)
	Logf        func(format string, args ...any)
}

// Server holds at most one loaded model and answers both APIs with it, loading the model a request names (and
// unloading the one before) as needed.
type Server struct {
	cfg  Config
	sem  chan struct{}
	mu   sync.Mutex
	cur  *loaded
	pull sync.Mutex // one download at a time
}

type loaded struct {
	name    string
	src     models.Source
	m       *engine.Model
	active  int            // requests using it (guarded by Server.mu)
	refs    sync.WaitGroup // the same, for closing once they finish
	timer   *time.Timer
	expires time.Time // zero: never
	drop    bool      // unload once idle (a keep_alive 0 that came while it was busy)
}

// New returns a server for cfg.
func New(cfg Config) *Server {
	if cfg.ModelsDir == "" {
		cfg.ModelsDir = models.DefaultDir()
	}
	if cfg.Parallel <= 0 {
		cfg.Parallel = 1
	}
	if cfg.KeepAlive == 0 {
		cfg.KeepAlive = 5 * time.Minute
	}
	if cfg.NumCtx <= 0 {
		cfg.NumCtx = 4096
	}
	if cfg.Logf == nil {
		cfg.Logf = func(string, ...any) {}
	}
	return &Server{cfg: cfg, sem: make(chan struct{}, cfg.Parallel)}
}

// Handler serves Ollama's API under /api/, OpenAI's under /v1/, and "nollama is running" at /.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.root)
	mux.HandleFunc("HEAD /{$}", s.root)
	mux.HandleFunc("GET /api/version", s.version)
	mux.HandleFunc("GET /api/tags", s.tags)
	mux.HandleFunc("GET /api/ps", s.ps)
	mux.HandleFunc("POST /api/show", s.show)
	mux.HandleFunc("POST /api/pull", s.pullModel)
	mux.HandleFunc("DELETE /api/delete", s.deleteModel)
	mux.HandleFunc("POST /api/generate", s.ollamaGenerate)
	mux.HandleFunc("POST /api/chat", s.ollamaChat)
	mux.HandleFunc("POST /api/embed", s.notSupported)
	mux.HandleFunc("POST /api/embeddings", s.notSupported)
	mux.HandleFunc("POST /api/create", s.notSupported)
	mux.HandleFunc("POST /api/copy", s.notSupported)
	mux.HandleFunc("POST /api/push", s.notSupported)
	mux.HandleFunc("GET /v1/models", s.openaiModels)
	mux.HandleFunc("GET /v1/models/{model...}", s.openaiModel)
	mux.HandleFunc("POST /v1/chat/completions", s.openaiChat)
	mux.HandleFunc("POST /v1/completions", s.openaiCompletions)
	mux.HandleFunc("POST /v1/embeddings", s.openaiNotSupported)
	return s.logged(mux)
}

func (s *Server) logged(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		h.ServeHTTP(sw, r)
		s.cfg.Logf("%s %s %d %s", r.Method, r.URL.Path, sw.status, time.Since(start).Round(time.Millisecond))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) { w.status = code; w.ResponseWriter.WriteHeader(code) }
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Close unloads the model once its answers finish.
func (s *Server) Close() error {
	s.mu.Lock()
	cur := s.cur
	s.cur = nil
	s.mu.Unlock()
	if cur != nil {
		if cur.timer != nil {
			cur.timer.Stop()
		}
		cur.refs.Wait()
		return cur.m.Close()
	}
	return nil
}

func (s *Server) root(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte("nollama is running"))
}

func (s *Server) version(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]string{"version": OllamaVersion})
}

// errNotFound is a model that isn't on disk; Ollama answers 404 and the client offers to pull it.
type errNotFound struct{ name string }

func (e errNotFound) Error() string {
	return fmt.Sprintf("model %q not found, try pulling it first", e.name)
}

// acquire returns the model named, loading it (and unloading the one before) when it isn't the loaded one. The caller
// calls release when done.
func (s *Server) acquire(name string) (*loaded, time.Duration, error) {
	if strings.TrimSpace(name) == "" {
		return nil, 0, errors.New("model is required")
	}
	src, err := models.Resolve(name, s.cfg.ModelsDir)
	if err != nil {
		return nil, 0, err
	}
	if !src.Present() {
		return nil, 0, errNotFound{name}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur != nil && s.cur.src.Path == src.Path {
		s.hold(s.cur)
		return s.cur, 0, nil
	}
	start := time.Now()
	if old := s.cur; old != nil {
		s.cur = nil
		s.retire(old)
	}
	m, err := engine.Load(src.Path, engine.LoadOptions{Threads: s.cfg.Threads, PromptCache: s.cfg.PromptCache})
	if err != nil {
		return nil, 0, fmt.Errorf("loading %s: %w", src.Name, err)
	}
	s.cur = &loaded{name: src.Name, src: src, m: m}
	s.hold(s.cur)
	load := time.Since(start)
	s.cfg.Logf("loaded %s in %s (%s)", src.Name, load.Round(time.Millisecond), m.Info().Kernels)
	return s.cur, load, nil
}

// hold marks l in use. Called with s.mu held.
func (s *Server) hold(l *loaded) {
	l.active++
	l.refs.Add(1)
	if l.timer != nil {
		l.timer.Stop()
		l.timer = nil
	}
	l.expires = time.Time{}
}

// release ends a use of l; once nothing uses it, it stays loaded for keepAlive (nil: the server's default).
func (s *Server) release(l *loaded, keepAlive *time.Duration) {
	ka := s.cfg.KeepAlive
	if keepAlive != nil {
		ka = *keepAlive
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	l.active--
	l.refs.Done()
	if l.active > 0 || s.cur != l {
		return
	}
	if l.drop {
		ka = 0
	}
	switch {
	case ka < 0:
		l.expires = time.Time{}
	case ka == 0:
		s.cur = nil
		s.retire(l)
	default:
		l.expires = time.Now().Add(ka)
		l.timer = time.AfterFunc(ka, func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			if s.cur == l && l.active == 0 {
				s.cur = nil
				s.retire(l)
			}
		})
	}
}

// retire closes l once its answers finish. Called with s.mu held, after l stopped being s.cur.
func (s *Server) retire(l *loaded) {
	if l.timer != nil {
		l.timer.Stop()
	}
	go func() {
		l.refs.Wait()
		_ = l.m.Close()
		s.cfg.Logf("unloaded %s", l.name)
	}()
}

// sampling is what a request asks of the sampler, in either API.
type sampling struct {
	Temperature *float64
	TopP        *float64
	TopK        *int
	Seed        *int64
	MaxTokens   int // 0: fill the context window
	NumCtx      int // 0: the server's default
	Stop        []string
	Schema      json.RawMessage // constrain the output to this JSON schema
}

// genRequest is one generation: the model, its prompt as tokens (built once the model is loaded) and the sampling.
type genRequest struct {
	model     string
	tokens    func(m *engine.Model) ([]int, error)
	sampling  sampling
	keepAlive *time.Duration
	onText    func(piece string) bool // nil: no streaming
}

type genOutput struct {
	model  string
	ids    []int // the prompt's tokens
	res    engine.Result
	load   time.Duration
	total  time.Duration
	reason string // "stop" or "length"
}

func (s *Server) generate(ctx context.Context, g genRequest) (genOutput, error) {
	start := time.Now()
	var out genOutput
	var constraint func(string) engine.Fit
	if len(g.sampling.Schema) > 0 {
		sc, err := schema.Compile(g.sampling.Schema)
		if err != nil {
			return out, badRequest{fmt.Errorf("format: %w", err)}
		}
		constraint = sc.Fit
	}
	l, load, err := s.acquire(g.model)
	if err != nil {
		return out, err
	}
	defer s.release(l, g.keepAlive)
	out.model, out.load = l.name, load
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return out, ctx.Err()
	}
	ids, err := g.tokens(l.m)
	if err != nil {
		return out, err
	}
	out.ids = ids
	if len(ids) == 0 {
		return out, badRequest{errors.New("the prompt is empty")}
	}
	modelCtx := l.m.Info().ContextLength
	numCtx := g.sampling.NumCtx
	if numCtx <= 0 {
		numCtx = s.cfg.NumCtx
	}
	if modelCtx > 0 && numCtx > modelCtx {
		numCtx = modelCtx
	}
	room := numCtx - len(ids)
	if room <= 0 {
		return out, badRequest{fmt.Errorf("the prompt is %d tokens, more than the context window (%d); set a larger num_ctx", len(ids), numCtx)}
	}
	maxTokens := g.sampling.MaxTokens
	if maxTokens <= 0 || maxTokens > room {
		maxTokens = room
	}
	o := engine.GenerateOptions{MaxTokens: maxTokens, Temperature: 0.8, TopP: 0.9, TopK: 40, Stop: g.sampling.Stop, Constraint: constraint}
	if t := g.sampling.Temperature; t != nil {
		o.Temperature = *t
	}
	if p := g.sampling.TopP; p != nil {
		o.TopP = *p
	}
	if k := g.sampling.TopK; k != nil {
		o.TopK = *k
	}
	if sd := g.sampling.Seed; sd != nil && *sd >= 0 {
		o.Seed = uint64(*sd)
	} else {
		o.Seed = rand.Uint64()
	}
	on := g.onText
	if on == nil {
		on = func(string) bool { return true }
	}
	res, err := l.m.GenerateTokens(ctx, ids, o, on)
	out.res = res
	out.total = time.Since(start)
	out.reason = "stop"
	if res.StopReason == "length" {
		out.reason = "length"
	}
	return out, err
}

// badRequest is an error in what the client sent.
type badRequest struct{ error }

func statusOf(err error) int {
	var nf errNotFound
	var br badRequest
	switch {
	case errors.As(err, &nf):
		return http.StatusNotFound
	case errors.As(err, &br):
		return http.StatusBadRequest
	case errors.Is(err, context.Canceled):
		return 499
	}
	return http.StatusInternalServerError
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func decode(r *http.Request, v any) error {
	dec := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 32<<20))
	if err := dec.Decode(v); err != nil {
		return badRequest{fmt.Errorf("reading the request: %w", err)}
	}
	return nil
}

// parseKeepAlive reads Ollama's keep_alive: seconds as a number, or a duration string ("5m", "-1"); nil when absent.
func parseKeepAlive(raw json.RawMessage) (*time.Duration, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var d time.Duration
	var n float64
	if err := json.Unmarshal(raw, &n); err == nil {
		d = time.Duration(n * float64(time.Second))
		return &d, nil
	}
	var str string
	if err := json.Unmarshal(raw, &str); err != nil {
		return nil, badRequest{fmt.Errorf("keep_alive: %w", err)}
	}
	if f, err := strconv.ParseFloat(str, 64); err == nil {
		d = time.Duration(f * float64(time.Second))
		return &d, nil
	}
	d, err := time.ParseDuration(str)
	if err != nil {
		return nil, badRequest{fmt.Errorf("keep_alive: %w", err)}
	}
	return &d, nil
}

// family guesses a model's family from its name, for listings ("qwen3-1.7b" is qwen3).
func family(name string) string {
	if i := strings.IndexAny(name, "-:"); i > 0 {
		return name[:i]
	}
	return name
}
