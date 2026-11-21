package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/benallfree/nollama/engine"
)

// embedInput is the texts to embed: a string, a list of strings, or (OpenAI) a list of token ids or of such lists.
type embedInput struct {
	texts  []string
	tokens [][]int
}

func (in *embedInput) UnmarshalJSON(b []byte) error {
	var one string
	if json.Unmarshal(b, &one) == nil {
		in.texts = []string{one}
		return nil
	}
	var many []string
	if json.Unmarshal(b, &many) == nil {
		in.texts = many
		return nil
	}
	var ids []int
	if json.Unmarshal(b, &ids) == nil {
		in.tokens = [][]int{ids}
		return nil
	}
	var lists [][]int
	if json.Unmarshal(b, &lists) == nil {
		in.tokens = lists
		return nil
	}
	return errors.New("input: want a string, a list of strings, or token ids")
}

func (in embedInput) len() int { return len(in.texts) + len(in.tokens) }

type embedRequest struct {
	model      string
	input      embedInput
	dimensions int
	numCtx     int  // 0: the server's default
	truncate   bool // cut texts longer than the context instead of failing
	keepAlive  *time.Duration
}

type embedOutput struct {
	model   string
	vectors [][]float32
	tokens  int
	load    time.Duration
	total   time.Duration
}

// embed loads the model and embeds the request's texts, one after another, taking a turn like an answer does.
func (s *Server) embed(ctx context.Context, e embedRequest) (embedOutput, error) {
	start := time.Now()
	var out embedOutput
	if e.dimensions < 0 {
		return out, badRequest{errors.New("dimensions must be positive")}
	}
	l, load, err := s.acquire(e.model)
	if err != nil {
		return out, err
	}
	defer s.release(l, e.keepAlive)
	out.model, out.load = l.name, load
	if dim := l.m.Info().Embedding; e.dimensions > dim {
		return out, badRequest{fmt.Errorf("dimensions %d: the model's vectors have %d", e.dimensions, dim)}
	}
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return out, ctx.Err()
	}
	numCtx := e.numCtx
	if numCtx <= 0 {
		numCtx = s.cfg.NumCtx
	}
	if numCtx > s.cfg.MaxNumCtx {
		numCtx = s.cfg.MaxNumCtx
	}
	ids := e.input.tokens
	if ids == nil {
		ids = make([][]int, len(e.input.texts))
		for i, t := range e.input.texts {
			ids[i] = l.m.Tokenize(t)
		}
	}
	vocab := l.m.VocabSize()
	for i, seq := range ids {
		for _, id := range seq {
			if id < 0 || id >= vocab {
				return out, badRequest{fmt.Errorf("input %d: token %d is outside the vocabulary", i, id)}
			}
		}
	}
	es, err := l.m.EmbedTokens(ctx, ids, engine.EmbedOptions{Dimensions: e.dimensions, MaxTokens: numCtx, Truncate: e.truncate})
	if err != nil {
		if ctx.Err() != nil {
			return out, err
		}
		return out, badRequest{err}
	}
	out.vectors = make([][]float32, len(es))
	for i, v := range es {
		out.vectors[i] = v.Vector
		out.tokens += v.Tokens
	}
	out.total = time.Since(start)
	return out, nil
}

// POST /api/embed: {"model", "input": string or [string], "truncate" (default true), "dimensions", "options",
// "keep_alive"} → {"model", "embeddings": [[...]], "total_duration", "load_duration", "prompt_eval_count"}.
func (s *Server) ollamaEmbed(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model      string          `json:"model"`
		Input      embedInput      `json:"input"`
		Truncate   *bool           `json:"truncate"`
		Dimensions int             `json:"dimensions"`
		Options    ollamaOptions   `json:"options"`
		KeepAlive  json.RawMessage `json:"keep_alive"`
	}
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	ka, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		ollamaError(w, err)
		return
	}
	if req.Input.len() == 0 { // Ollama's preload: load the model, embed nothing
		s.loadOnly(w, req.Model, ka, func(name string, _ bool) any {
			return map[string]any{"model": name, "embeddings": [][]float32{}}
		})
		return
	}
	out, err := s.embed(r.Context(), embedRequest{model: req.Model, input: req.Input, dimensions: req.Dimensions,
		numCtx: req.Options.NumCtx, truncate: req.Truncate == nil || *req.Truncate, keepAlive: ka})
	if err != nil {
		ollamaError(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"model": out.model, "embeddings": out.vectors, "total_duration": int64(out.total),
		"load_duration": int64(out.load), "prompt_eval_count": out.tokens})
}

// POST /api/embeddings, Ollama's older form: {"model", "prompt", "options", "keep_alive"} → {"embedding": [...]}.
// The vector is unit length, as /api/embed's are.
func (s *Server) ollamaEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model     string          `json:"model"`
		Prompt    string          `json:"prompt"`
		Options   ollamaOptions   `json:"options"`
		KeepAlive json.RawMessage `json:"keep_alive"`
	}
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	ka, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		ollamaError(w, err)
		return
	}
	if req.Prompt == "" {
		s.loadOnly(w, req.Model, ka, func(string, bool) any { return map[string]any{"embedding": []float32{}} })
		return
	}
	out, err := s.embed(r.Context(), embedRequest{model: req.Model, input: embedInput{texts: []string{req.Prompt}},
		numCtx: req.Options.NumCtx, truncate: true, keepAlive: ka})
	if err != nil {
		ollamaError(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"embedding": out.vectors[0]})
}

// POST /v1/embeddings: {"model", "input": string, [string], [int] or [[int]], "dimensions", "encoding_format"
// ("float" or "base64")} → {"object": "list", "data": [{"object": "embedding", "embedding", "index"}], "model",
// "usage"}. A text longer than the context is an error, as it is at OpenAI.
func (s *Server) openaiEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model          string     `json:"model"`
		Input          embedInput `json:"input"`
		Dimensions     int        `json:"dimensions"`
		EncodingFormat string     `json:"encoding_format"`
	}
	if err := decode(r, &req); err != nil {
		openaiError(w, err)
		return
	}
	if req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64" {
		openaiError(w, badRequest{fmt.Errorf("encoding_format %q: want float or base64", req.EncodingFormat)})
		return
	}
	if req.Input.len() == 0 {
		openaiError(w, badRequest{errors.New("input is required")})
		return
	}
	out, err := s.embed(r.Context(), embedRequest{model: req.Model, input: req.Input, dimensions: req.Dimensions})
	if err != nil {
		openaiError(w, err)
		return
	}
	data := make([]map[string]any, len(out.vectors))
	for i, v := range out.vectors {
		var emb any = v
		if req.EncodingFormat == "base64" { // little-endian float32s, as OpenAI sends them
			b := make([]byte, 4*len(v))
			for j, x := range v {
				binary.LittleEndian.PutUint32(b[4*j:], math.Float32bits(x))
			}
			emb = base64.StdEncoding.EncodeToString(b)
		}
		data[i] = map[string]any{"object": "embedding", "embedding": emb, "index": i}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data, "model": out.model,
		"usage": map[string]int{"prompt_tokens": out.tokens, "total_tokens": out.tokens}})
}
