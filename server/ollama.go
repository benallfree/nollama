package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/models"
)

// Ollama's API: https://github.com/ollama/ollama/blob/main/docs/api.md

type ollamaOptions struct {
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`
	TopK        *int     `json:"top_k"`
	NumPredict  *int     `json:"num_predict"`
	NumCtx      int      `json:"num_ctx"`
	Seed        *int64   `json:"seed"`
	Stop        []string `json:"stop"`
}

func (o ollamaOptions) sampling(format json.RawMessage) (sampling, error) {
	sp := sampling{Temperature: o.Temperature, TopP: o.TopP, TopK: o.TopK, Seed: o.Seed, NumCtx: o.NumCtx, Stop: o.Stop}
	if o.NumPredict != nil && *o.NumPredict > 0 {
		sp.MaxTokens = *o.NumPredict
	}
	sc, err := ollamaFormat(format)
	sp.Schema = sc
	return sp, err
}

// ollamaFormat reads "format": "json" (any JSON object) or a JSON schema.
func ollamaFormat(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 || string(raw) == "null" || string(raw) == `""` {
		return nil, nil
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		if str == "json" {
			return json.RawMessage(`{"type":"object"}`), nil
		}
		return nil, badRequest{fmt.Errorf("format %q: want \"json\" or a JSON schema", str)}
	}
	return raw, nil
}

type ollamaMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Images    []string          `json:"images,omitempty"`
	ToolCalls []json.RawMessage `json:"tool_calls,omitempty"`
}

type ollamaGenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	Suffix    string          `json:"suffix"`
	System    string          `json:"system"`
	Format    json.RawMessage `json:"format"`
	Options   ollamaOptions   `json:"options"`
	Stream    *bool           `json:"stream"`
	Raw       bool            `json:"raw"`
	Context   []int           `json:"context"`
	Images    []string        `json:"images"`
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type ollamaChatRequest struct {
	Model     string            `json:"model"`
	Messages  []ollamaMessage   `json:"messages"`
	Format    json.RawMessage   `json:"format"`
	Options   ollamaOptions     `json:"options"`
	Stream    *bool             `json:"stream"`
	Tools     []json.RawMessage `json:"tools"`
	KeepAlive json.RawMessage   `json:"keep_alive"`
	Truncate  *bool             `json:"truncate"` // drop the oldest messages if the chat doesn't fit (nollama's; nil: the server's default)
}

// ollamaDone is the closing fields of a generate or chat answer.
type ollamaDone struct {
	DoneReason         string     `json:"done_reason,omitempty"`
	TotalDuration      int64      `json:"total_duration,omitempty"`
	LoadDuration       int64      `json:"load_duration,omitempty"`
	PromptEvalCount    int        `json:"prompt_eval_count"`
	PromptEvalDuration int64      `json:"prompt_eval_duration,omitempty"`
	EvalCount          int        `json:"eval_count"`
	EvalDuration       int64      `json:"eval_duration,omitempty"`
	Truncated          *truncated `json:"truncated,omitempty"`
}

func doneOf(out genOutput) ollamaDone {
	return ollamaDone{DoneReason: out.reason, TotalDuration: int64(out.total), LoadDuration: int64(out.load),
		PromptEvalCount: out.res.PromptTokens, PromptEvalDuration: int64(out.res.Prefill),
		EvalCount: out.res.OutputTokens, EvalDuration: int64(out.res.Decode), Truncated: truncatedOf(out)}
}

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

func ollamaError(w http.ResponseWriter, err error) {
	writeJSON(w, statusOf(err), map[string]string{"error": err.Error()})
}

// ndjson streams one JSON object per line.
type ndjson struct {
	w       http.ResponseWriter
	started bool
	failed  bool
}

func (n *ndjson) send(v any) bool {
	if n.failed {
		return false
	}
	if !n.started {
		n.w.Header().Set("Content-Type", "application/x-ndjson")
		n.w.WriteHeader(200)
		n.started = true
	}
	b, _ := json.Marshal(v)
	if _, err := n.w.Write(append(b, '\n')); err != nil {
		n.failed = true
		return false
	}
	if f, ok := n.w.(http.Flusher); ok {
		f.Flush()
	}
	return true
}

func (s *Server) ollamaGenerate(w http.ResponseWriter, r *http.Request) {
	var req ollamaGenerateRequest
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	ka, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		ollamaError(w, err)
		return
	}
	if len(req.Images) > 0 {
		ollamaError(w, badRequest{errors.New("images aren't supported: nollama runs text models")})
		return
	}
	if req.Prompt == "" && len(req.Context) == 0 {
		// an empty prompt loads (or, with keep_alive 0, unloads) the model
		s.loadOnly(w, req.Model, ka, func(name string, unloading bool) any {
			m := map[string]any{"model": name, "created_at": now(), "response": "", "done": true, "done_reason": "load"}
			if unloading {
				m["done_reason"] = "unload"
			}
			return m
		})
		return
	}
	sp, err := req.Options.sampling(req.Format)
	if err != nil {
		ollamaError(w, err)
		return
	}
	g := genRequest{model: req.Model, sampling: sp, keepAlive: ka,
		tokens: func(m *engine.Model) ([]int, error) {
			prompt := req.Prompt
			if !req.Raw {
				var msgs []engine.Message
				if req.System != "" {
					msgs = append(msgs, engine.Message{Role: "system", Content: req.System})
				}
				msgs = append(msgs, engine.Message{Role: "user", Content: req.Prompt})
				prompt = m.ChatPrompt(msgs)
			}
			return append(append([]int(nil), req.Context...), m.Tokenize(prompt)...), nil
		}}
	stream := req.Stream == nil || *req.Stream
	nd := &ndjson{w: w}
	model := req.Model
	if stream {
		g.onText = func(piece string) bool {
			return nd.send(map[string]any{"model": model, "created_at": now(), "response": piece, "done": false})
		}
	}
	out, err := s.generate(r.Context(), g)
	if err != nil {
		if nd.started {
			nd.send(map[string]string{"error": err.Error()})
		} else {
			ollamaError(w, err)
		}
		return
	}
	final := struct {
		Model     string `json:"model"`
		CreatedAt string `json:"created_at"`
		Response  string `json:"response"`
		Done      bool   `json:"done"`
		Context   []int  `json:"context,omitempty"`
		ollamaDone
	}{Model: model, CreatedAt: now(), Done: true, ollamaDone: doneOf(out)}
	final.Context = append(append([]int(nil), out.ids...), out.res.Tokens...)
	if !stream {
		final.Response = out.res.Text
		writeJSON(w, 200, final)
		return
	}
	nd.send(final)
}

func (s *Server) ollamaChat(w http.ResponseWriter, r *http.Request) {
	var req ollamaChatRequest
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	ka, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		ollamaError(w, err)
		return
	}
	if len(req.Tools) > 0 {
		ollamaError(w, badRequest{errors.New("tools aren't supported yet")})
		return
	}
	msgs := make([]engine.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		if len(m.Images) > 0 {
			ollamaError(w, badRequest{errors.New("images aren't supported: nollama runs text models")})
			return
		}
		msgs = append(msgs, engine.Message{Role: m.Role, Content: m.Content})
	}
	if len(msgs) == 0 {
		s.loadOnly(w, req.Model, ka, func(name string, unloading bool) any {
			m := map[string]any{"model": name, "created_at": now(), "message": ollamaMessage{Role: "assistant"}, "done": true, "done_reason": "load"}
			if unloading {
				m["done_reason"] = "unload"
			}
			return m
		})
		return
	}
	sp, err := req.Options.sampling(req.Format)
	if err != nil {
		ollamaError(w, err)
		return
	}
	sp.Truncate = req.Truncate
	g := genRequest{model: req.Model, sampling: sp, keepAlive: ka, trim: chatTrim(msgs),
		tokens: func(m *engine.Model) ([]int, error) { return m.Tokenize(m.ChatPrompt(msgs)), nil }}
	stream := req.Stream == nil || *req.Stream
	nd := &ndjson{w: w}
	model := req.Model
	if stream {
		g.onText = func(piece string) bool {
			return nd.send(map[string]any{"model": model, "created_at": now(),
				"message": ollamaMessage{Role: "assistant", Content: piece}, "done": false})
		}
	}
	out, err := s.generate(r.Context(), g)
	if err != nil {
		if nd.started {
			nd.send(map[string]string{"error": err.Error()})
		} else {
			ollamaError(w, err)
		}
		return
	}
	final := struct {
		Model     string        `json:"model"`
		CreatedAt string        `json:"created_at"`
		Message   ollamaMessage `json:"message"`
		Done      bool          `json:"done"`
		ollamaDone
	}{Model: model, CreatedAt: now(), Message: ollamaMessage{Role: "assistant"}, Done: true, ollamaDone: doneOf(out)}
	if !stream {
		final.Message.Content = out.res.Text
		writeJSON(w, 200, final)
		return
	}
	nd.send(final)
}

// loadOnly loads a model without generating (Ollama's empty-prompt preload); keep_alive 0 unloads it instead.
func (s *Server) loadOnly(w http.ResponseWriter, name string, ka *time.Duration, reply func(name string, unloading bool) any) {
	unloading := ka != nil && *ka == 0
	if unloading {
		s.mu.Lock()
		if s.cur != nil {
			if src, err := s.resolve(name); err == nil && src.Key() == s.cur.src.Key() {
				if s.cur.active > 0 {
					s.cur.drop = true // after the answers it is writing
				} else {
					old := s.cur
					s.cur = nil
					s.retire(old)
				}
			}
		}
		s.mu.Unlock()
		writeJSON(w, 200, reply(name, true))
		return
	}
	l, _, err := s.acquire(name)
	if err != nil {
		ollamaError(w, err)
		return
	}
	s.release(l, ka)
	writeJSON(w, 200, reply(name, false))
}

type ollamaDetails struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size,omitempty"`
	QuantizationLevel string   `json:"quantization_level,omitempty"`
}

func (s *Server) tags(w http.ResponseWriter, r *http.Request) {
	local, err := models.Available(s.cfg.ModelsDir)
	if err != nil {
		ollamaError(w, err)
		return
	}
	type tag struct {
		Name       string        `json:"name"`
		Model      string        `json:"model"`
		ModifiedAt string        `json:"modified_at"`
		Size       int64         `json:"size"`
		Digest     string        `json:"digest"`
		Details    ollamaDetails `json:"details"`
	}
	out := []tag{}
	for _, m := range local {
		if !s.modelAllowed(m.Name) {
			continue
		}
		d := ollamaDetails{Format: "gguf", Family: family(m.Name), Families: []string{family(m.Name)}}
		out = append(out, tag{Name: m.Name, Model: m.Name, ModifiedAt: time.Unix(m.ModTime, 0).UTC().Format(time.RFC3339),
			Size: m.Size, Digest: m.SHA256, Details: d})
	}
	writeJSON(w, 200, map[string]any{"models": out})
}

func (s *Server) ps(w http.ResponseWriter, r *http.Request) {
	type running struct {
		Name      string        `json:"name"`
		Model     string        `json:"model"`
		Size      int64         `json:"size"`
		Digest    string        `json:"digest"`
		Details   ollamaDetails `json:"details"`
		ExpiresAt string        `json:"expires_at"`
		SizeVRAM  int64         `json:"size_vram"`
	}
	out := []running{}
	s.mu.Lock()
	if l := s.cur; l != nil {
		info := l.m.Info()
		exp := ""
		if !l.expires.IsZero() {
			exp = l.expires.UTC().Format(time.RFC3339Nano)
		}
		out = append(out, running{Name: l.name, Model: l.name, Size: info.FileSize, Digest: l.src.SHA256, ExpiresAt: exp,
			Details: detailsOf(l.name, info)})
	}
	s.mu.Unlock()
	writeJSON(w, 200, map[string]any{"models": out})
}

func detailsOf(name string, info engine.Info) ollamaDetails {
	d := ollamaDetails{Format: "gguf", Family: info.Arch, Families: []string{info.Arch}, QuantizationLevel: info.Quant}
	if d.Family == "" {
		d.Family = family(name)
		d.Families = []string{d.Family}
	}
	if info.Params > 0 {
		d.ParameterSize = fmt.Sprintf("%.1fB", float64(info.Params)/1e9)
	}
	return d
}

func (s *Server) show(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"` // older clients
	}
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	if req.Model == "" {
		req.Model = req.Name
	}
	l, _, err := s.acquire(req.Model)
	if err != nil {
		ollamaError(w, err)
		return
	}
	defer s.release(l, nil)
	info := l.m.Info()
	caps := []string{"completion"}
	if info.Pooling != "" { // an embedding model, as Ollama reports one
		caps = []string{"embedding"}
	}
	writeJSON(w, 200, map[string]any{
		"modelfile":    "",
		"parameters":   "",
		"template":     "",
		"details":      detailsOf(l.name, info),
		"capabilities": caps,
		"model_info": map[string]any{
			"general.architecture":          info.Arch,
			"general.basename":              info.Name,
			"general.parameter_count":       info.Params,
			"general.file_type":             info.Quant,
			info.Arch + ".context_length":   info.ContextLength,
			info.Arch + ".block_count":      info.Layers,
			info.Arch + ".embedding_length": info.Embedding,
			"nollama.threads":               info.Threads,
			"nollama.kernels":               info.Kernels,
		},
		"modified_at": now(),
	})
}

func (s *Server) pullModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		Name   string `json:"name"`
		Stream *bool  `json:"stream"`
	}
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	if req.Model == "" {
		req.Model = req.Name
	}
	src, err := s.resolve(req.Model)
	if err != nil {
		ollamaError(w, err)
		return
	}
	stream := req.Stream == nil || *req.Stream
	nd := &ndjson{w: w}
	if stream {
		nd.send(map[string]string{"status": "pulling manifest"})
	}
	last := time.Time{}
	progress := func(done, total int64) {
		if !stream || time.Since(last) < 250*time.Millisecond {
			return
		}
		last = time.Now()
		nd.send(map[string]any{"status": "pulling " + src.Name, "digest": src.SHA256, "total": total, "completed": done})
	}
	s.pull.Lock()
	err = models.Fetch(src, s.cfg.Client, progress)
	s.pull.Unlock()
	if err != nil {
		if stream {
			nd.send(map[string]string{"error": err.Error()})
		} else {
			ollamaError(w, err)
		}
		return
	}
	if stream {
		nd.send(map[string]string{"status": "verifying sha256 digest"})
		nd.send(map[string]string{"status": "success"})
		return
	}
	writeJSON(w, 200, map[string]string{"status": "success"})
}

func (s *Server) deleteModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	if req.Model == "" {
		req.Model = req.Name
	}
	src, err := s.resolve(req.Model)
	if err != nil {
		ollamaError(w, err)
		return
	}
	if src.Inside {
		ollamaError(w, badRequest{fmt.Errorf("%s is built into this program, not a download; it can't be deleted", src.Name)})
		return
	}
	if src.URL == "" {
		ollamaError(w, badRequest{fmt.Errorf("%s is a file you named, not a download; nollama leaves it alone", src.Name)})
		return
	}
	if !src.Present() {
		ollamaError(w, errNotFound{req.Model})
		return
	}
	s.mu.Lock()
	if s.cur != nil && s.cur.src.Key() == src.Key() {
		old := s.cur
		s.cur = nil
		s.retire(old)
	}
	s.mu.Unlock()
	if err := os.Remove(src.Path); err != nil {
		ollamaError(w, err)
		return
	}
	w.WriteHeader(200)
}

func (s *Server) notSupported(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "nollama doesn't support " + r.URL.Path + " yet"})
}
