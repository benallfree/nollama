package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/models"
	"github.com/benallfree/nollama/schema"
)

// OpenAI's API, the parts a local model answers: models, chat completions and (legacy) completions.

// openaiContent is a message's content: a string, or a list of parts of which only text is read.
type openaiContent string

func (c *openaiContent) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*c = openaiContent(s)
		return nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(b, &parts); err != nil {
		return err
	}
	var sb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text", "input_text":
			sb.WriteString(p.Text)
		default:
			return fmt.Errorf("content part %q isn't supported: nollama runs text models", p.Type)
		}
	}
	*c = openaiContent(sb.String())
	return nil
}

// openaiStop is "stop": a string or a list.
type openaiStop []string

func (s *openaiStop) UnmarshalJSON(b []byte) error {
	var one string
	if err := json.Unmarshal(b, &one); err == nil {
		*s = openaiStop{one}
		return nil
	}
	var many []string
	err := json.Unmarshal(b, &many)
	*s = many
	return err
}

type openaiCommon struct {
	Model               string     `json:"model"`
	MaxTokens           int        `json:"max_tokens"`
	MaxCompletionTokens int        `json:"max_completion_tokens"`
	Temperature         *float64   `json:"temperature"`
	TopP                *float64   `json:"top_p"`
	Seed                *int64     `json:"seed"`
	Stop                openaiStop `json:"stop"`
	Stream              bool       `json:"stream"`
	StreamOptions       struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	ResponseFormat *struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
	N          int               `json:"n"`
	Tools      []json.RawMessage `json:"tools"`
	ToolChoice json.RawMessage   `json:"tool_choice"`
	Truncate   *bool             `json:"truncate"` // nollama's: drop the oldest messages if the chat doesn't fit
}

func (c openaiCommon) sampling() (sampling, error) {
	sp := sampling{Temperature: c.Temperature, TopP: c.TopP, Seed: c.Seed, Stop: c.Stop, MaxTokens: c.MaxTokens, Truncate: c.Truncate}
	if c.MaxCompletionTokens > 0 {
		sp.MaxTokens = c.MaxCompletionTokens
	}
	if c.Temperature == nil {
		t := 1.0 // OpenAI's default
		sp.Temperature = &t
	}
	if c.N > 1 {
		return sp, badRequest{errors.New("n > 1 isn't supported")}
	}
	ts, err := parseTools(c.Tools, c.ToolChoice)
	if err != nil {
		return sp, err
	}
	sp.Tools = ts
	if rf := c.ResponseFormat; rf != nil {
		switch rf.Type {
		case "", "text":
		case "json_object":
			sp.Schema = json.RawMessage(`{"type":"object"}`)
		case "json_schema":
			if rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
				return sp, badRequest{errors.New("response_format json_schema needs json_schema.schema")}
			}
			sp.Schema = rf.JSONSchema.Schema
		default:
			return sp, badRequest{fmt.Errorf("response_format %q isn't supported", rf.Type)}
		}
	}
	return sp, nil
}

type openaiUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func usageOf(out genOutput) *openaiUsage {
	u := &openaiUsage{PromptTokens: out.res.PromptTokens, CompletionTokens: out.res.OutputTokens,
		TotalTokens: out.res.PromptTokens + out.res.OutputTokens}
	u.PromptTokensDetails.CachedTokens = out.res.CachedTokens
	return u
}

func openaiError(w http.ResponseWriter, err error) {
	status := statusOf(err)
	typ := "server_error"
	switch status {
	case http.StatusBadRequest:
		typ = "invalid_request_error"
	case http.StatusNotFound:
		typ = "not_found_error"
	case http.StatusUnauthorized:
		typ = "authentication_error"
	case http.StatusForbidden:
		typ = "permission_error"
	case http.StatusTooManyRequests:
		typ = "rate_limit_error"
	case http.StatusRequestEntityTooLarge:
		typ = "invalid_request_error"
	}
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": err.Error(), "type": typ, "code": nil}})
}

func (s *Server) openaiNotSupported(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": map[string]any{
		"message": "nollama doesn't support " + r.URL.Path + " yet", "type": "invalid_request_error", "code": nil}})
}

// sse streams server-sent events, ending with [DONE].
type sse struct {
	w       http.ResponseWriter
	started bool
	failed  bool
}

func (e *sse) send(v any) bool {
	if e.failed {
		return false
	}
	if !e.started {
		e.w.Header().Set("Content-Type", "text/event-stream")
		e.w.Header().Set("Cache-Control", "no-cache")
		e.w.WriteHeader(200)
		e.started = true
	}
	var data []byte
	if s, ok := v.(string); ok {
		data = []byte(s)
	} else {
		data, _ = json.Marshal(v)
	}
	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		e.failed = true
		return false
	}
	if f, ok := e.w.(http.Flusher); ok {
		f.Flush()
	}
	return true
}

func completionID(prefix string) string {
	return fmt.Sprintf("%s-%x", prefix, time.Now().UnixNano())
}

// openaiToolCall is a call in an assistant message: arguments are a string holding JSON.
type openaiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

func openaiCalls(calls []schema.ToolCall, from int) []any {
	out := make([]any, len(calls))
	for i, c := range calls {
		out[i] = map[string]any{"index": from + i, "id": callID(), "type": "function",
			"function": map[string]any{"name": c.Name, "arguments": string(c.Arguments)}}
	}
	return out
}

func (s *Server) openaiChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		openaiCommon
		Messages []struct {
			Role       string           `json:"role"`
			Content    openaiContent    `json:"content"`
			ToolCalls  []openaiToolCall `json:"tool_calls"`
			ToolCallID string           `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := decode(r, &req); err != nil {
		openaiError(w, err)
		return
	}
	sp, err := req.sampling()
	if err != nil {
		openaiError(w, err)
		return
	}
	if len(req.Messages) == 0 {
		openaiError(w, badRequest{errors.New("messages is required")})
		return
	}
	msgs := make([]engine.Message, len(req.Messages))
	for i, m := range req.Messages {
		role := m.Role
		if role == "developer" {
			role = "system"
		}
		msgs[i] = engine.Message{Role: role, Content: string(m.Content)}
		for _, c := range m.ToolCalls {
			msgs[i].ToolCalls = append(msgs[i].ToolCalls, engine.ToolCall{Name: c.Function.Name, Arguments: argsText(c.Function.Arguments)})
		}
	}
	id, created, model := completionID("chatcmpl"), time.Now().Unix(), req.Model
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	tools := sp.Tools.prompt()
	g := genRequest{model: req.Model, sampling: sp, trim: chatTrim(msgs, tools),
		tokens: func(m *engine.Model) ([]int, error) { return m.Tokenize(m.ChatPromptTools(msgs, tools)), nil }}
	ev := &sse{w: w}
	first := true
	send := func(d map[string]any) bool {
		if first {
			d["role"] = "assistant"
			first = false
		}
		return ev.send(chunk(d, nil))
	}
	var split *toolStream
	ncalls := 0
	if req.Stream {
		if sp.Tools.calling() {
			split = &toolStream{}
			g.onText = func(piece string) bool {
				content, calls := split.feed(piece)
				if content != "" && !send(map[string]any{"content": content}) {
					return false
				}
				if len(calls) > 0 {
					if !send(map[string]any{"tool_calls": openaiCalls(calls, ncalls)}) {
						return false
					}
					ncalls += len(calls)
				}
				return true
			}
		} else {
			g.onText = func(piece string) bool { return send(map[string]any{"content": piece}) }
		}
	}
	out, err := s.generate(r.Context(), g)
	if err != nil {
		if ev.started {
			ev.send(map[string]any{"error": map[string]any{"message": err.Error(), "type": "server_error"}})
			ev.send("[DONE]")
		} else {
			openaiError(w, err)
		}
		return
	}
	finish := out.reason
	if !req.Stream {
		msg := map[string]any{"role": "assistant", "content": out.res.Text}
		if sp.Tools.calling() {
			content, calls := schema.SplitToolCalls(out.res.Text)
			msg["content"] = content
			if len(calls) > 0 {
				calls := openaiCalls(calls, 0)
				for _, c := range calls {
					delete(c.(map[string]any), "index")
				}
				msg["tool_calls"] = calls
				if content == "" {
					msg["content"] = nil
				}
				if finish == "stop" {
					finish = "tool_calls"
				}
			}
		}
		res := map[string]any{"id": id, "object": "chat.completion", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}},
			"usage":   usageOf(out)}
		if t := truncatedOf(out); t != nil {
			res["truncated"] = t
		}
		writeJSON(w, 200, res)
		return
	}
	if split != nil {
		if rest := split.rest(); rest != "" {
			send(map[string]any{"content": rest})
		}
		if ncalls > 0 && finish == "stop" {
			finish = "tool_calls"
		}
	}
	last := chunk(map[string]any{}, finish)
	if t := truncatedOf(out); t != nil {
		last["truncated"] = t
	}
	ev.send(last)
	if req.StreamOptions.IncludeUsage {
		ev.send(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{}, "usage": usageOf(out)})
	}
	ev.send("[DONE]")
}

func (s *Server) openaiCompletions(w http.ResponseWriter, r *http.Request) {
	var req struct {
		openaiCommon
		Prompt json.RawMessage `json:"prompt"`
	}
	if err := decode(r, &req); err != nil {
		openaiError(w, err)
		return
	}
	sp, err := req.sampling()
	if err != nil {
		openaiError(w, err)
		return
	}
	var prompt string
	if err := json.Unmarshal(req.Prompt, &prompt); err != nil {
		var list []string
		if json.Unmarshal(req.Prompt, &list) != nil || len(list) != 1 {
			openaiError(w, badRequest{errors.New("prompt: want a string (or a list of one)")})
			return
		}
		prompt = list[0]
	}
	if sp.Tools != nil {
		openaiError(w, badRequest{errors.New("tools need /v1/chat/completions")})
		return
	}
	if sp.MaxTokens == 0 {
		sp.MaxTokens = 16 // OpenAI's default for completions
	}
	id, created, model := completionID("cmpl"), time.Now().Unix(), req.Model
	chunk := func(text string, finish any) map[string]any {
		return map[string]any{"id": id, "object": "text_completion", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "text": text, "finish_reason": finish, "logprobs": nil}}}
	}
	g := genRequest{model: req.Model, sampling: sp,
		tokens: func(m *engine.Model) ([]int, error) { return m.Tokenize(prompt), nil }}
	ev := &sse{w: w}
	if req.Stream {
		g.onText = func(piece string) bool { return ev.send(chunk(piece, nil)) }
	}
	out, err := s.generate(r.Context(), g)
	if err != nil {
		if ev.started {
			ev.send(map[string]any{"error": map[string]any{"message": err.Error(), "type": "server_error"}})
			ev.send("[DONE]")
		} else {
			openaiError(w, err)
		}
		return
	}
	if !req.Stream {
		c := chunk(out.res.Text, out.reason)
		c["usage"] = usageOf(out)
		writeJSON(w, 200, c)
		return
	}
	ev.send(chunk("", out.reason))
	if req.StreamOptions.IncludeUsage {
		ev.send(map[string]any{"id": id, "object": "text_completion", "created": created, "model": model,
			"choices": []any{}, "usage": usageOf(out)})
	}
	ev.send("[DONE]")
}

func (s *Server) openaiModels(w http.ResponseWriter, r *http.Request) {
	local, err := models.Available(s.cfg.ModelsDir)
	if err != nil {
		openaiError(w, err)
		return
	}
	data := []any{}
	for _, m := range local {
		if !s.modelAllowed(m.Name) {
			continue
		}
		data = append(data, map[string]any{"id": m.Name, "object": "model", "created": m.ModTime, "owned_by": "nollama"})
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func (s *Server) openaiModel(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("model")
	src, err := s.resolve(name)
	if err != nil || !src.Present() {
		openaiError(w, errNotFound{name})
		return
	}
	writeJSON(w, 200, map[string]any{"id": src.Name, "object": "model", "created": time.Now().Unix(), "owned_by": "nollama"})
}
