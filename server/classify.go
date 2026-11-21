package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/benallfree/nollama/engine"
)

// POST /api/classify is nollama's own endpoint (neither Ollama nor OpenAI has one): it scores each candidate label as
// the model's answer to the prompt, on one read of the prompt, and returns a probability per label. No text is
// generated, so the answer is always one of the labels.
//
//	{"model": "qwen3-0.6b", "system": "Label the support ticket.", "prompt": "The app crashes on save.",
//	 "labels": ["bug", "feature request", "billing question"], "complete": true}
//
// The prompt is the user's turn, in the chat template, unless raw is set; messages may replace system and prompt.
// complete scores each label as the whole answer (followed by the end of the turn), normalize divides each label's
// log-probability by its token count.

type classifyRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	System    string          `json:"system"`
	Messages  []ollamaMessage `json:"messages"`
	Raw       bool            `json:"raw"`
	Labels    []string        `json:"labels"`
	Normalize bool            `json:"normalize"`
	Complete  bool            `json:"complete"`
	Options   struct {
		NumCtx int `json:"num_ctx"`
	} `json:"options"`
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type classifyLabel struct {
	Label       string  `json:"label"`
	Probability float64 `json:"probability"`
	Logprob     float64 `json:"logprob"` // natural log, summed over the label's tokens
	Tokens      int     `json:"tokens"`
}

type classifyResponse struct {
	Model           string          `json:"model"`
	CreatedAt       string          `json:"created_at"`
	Label           string          `json:"label"`  // the most likely label
	Labels          []classifyLabel `json:"labels"` // in request order
	TotalDuration   int64           `json:"total_duration"`
	LoadDuration    int64           `json:"load_duration"`
	PromptEvalCount int             `json:"prompt_eval_count"`
	CachedTokens    int             `json:"prompt_cache_count"`
}

func (s *Server) classify(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req classifyRequest
	if err := decode(r, &req); err != nil {
		ollamaError(w, err)
		return
	}
	ka, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		ollamaError(w, err)
		return
	}
	if len(req.Labels) == 0 {
		ollamaError(w, badRequest{errors.New("labels is required")})
		return
	}
	for _, l := range req.Labels {
		if l == "" {
			ollamaError(w, badRequest{errors.New("labels: a label is empty")})
			return
		}
	}
	if req.Prompt == "" && len(req.Messages) == 0 {
		ollamaError(w, badRequest{errors.New("prompt or messages is required")})
		return
	}
	if req.Raw && len(req.Messages) > 0 {
		ollamaError(w, badRequest{errors.New("raw takes a prompt, not messages")})
		return
	}

	l, load, err := s.acquire(req.Model)
	if err != nil {
		ollamaError(w, err)
		return
	}
	defer s.release(l, ka)
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-r.Context().Done():
		ollamaError(w, r.Context().Err())
		return
	}

	prompt := req.Prompt
	if !req.Raw {
		var msgs []engine.Message
		if req.System != "" {
			msgs = append(msgs, engine.Message{Role: "system", Content: req.System})
		}
		for _, m := range req.Messages {
			msgs = append(msgs, engine.Message{Role: m.Role, Content: m.Content})
		}
		if req.Prompt != "" {
			msgs = append(msgs, engine.Message{Role: "user", Content: req.Prompt})
		}
		prompt = l.m.ChatPrompt(msgs)
	}
	ids := l.m.Tokenize(prompt)
	numCtx := req.Options.NumCtx
	if numCtx <= 0 {
		numCtx = s.cfg.NumCtx
	}
	numCtx = min(numCtx, s.cfg.MaxNumCtx)
	if mc := l.m.Info().ContextLength; mc > 0 && numCtx > mc {
		numCtx = mc
	}
	if len(ids) >= numCtx { // never cut the prompt: a 400 with the counts
		ollamaError(w, badRequest{fmt.Errorf("%w (num_ctx %d); raise num_ctx or shorten the prompt",
			&engine.ContextError{PromptTokens: len(ids), Limit: numCtx - 1}, numCtx)})
		return
	}

	res, err := l.m.ClassifyTokens(r.Context(), ids, req.Labels, engine.ClassifyOptions{Normalize: req.Normalize, Complete: req.Complete})
	if errors.Is(err, engine.ErrContextExceeded) {
		err = badRequest{err}
	}
	if err != nil {
		ollamaError(w, err)
		return
	}
	out := classifyResponse{Model: l.name, CreatedAt: now(), Label: req.Labels[res.Best()], Labels: make([]classifyLabel, len(req.Labels)),
		LoadDuration: int64(load), PromptEvalCount: res.PromptTokens, CachedTokens: res.CachedTokens}
	for i, lb := range req.Labels {
		out.Labels[i] = classifyLabel{Label: lb, Probability: res.Probs[i], Logprob: res.Logprobs[i], Tokens: res.Tokens[i]}
	}
	out.TotalDuration = int64(time.Since(start))
	writeJSON(w, 200, out)
}
