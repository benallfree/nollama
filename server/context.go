package server

import (
	"fmt"

	"github.com/benallfree/nollama/engine"
)

// truncated is the "truncated" field of an answer whose prompt was cut to fit (a request's "truncate": true, or the
// server's Config.Truncate): what was dropped, so a client can tell the model never saw it.
type truncated struct {
	Messages     int `json:"messages"`      // the oldest messages after the system prompt
	Tokens       int `json:"tokens"`        // tokens they took
	PromptTokens int `json:"prompt_tokens"` // the prompt before trimming
}

func truncatedOf(out genOutput) *truncated {
	if out.trimmed.Messages == 0 {
		return nil
	}
	return &truncated{Messages: out.trimmed.Messages, Tokens: out.trimmed.Tokens, PromptTokens: out.trimmed.PromptTokens}
}

// chatTrim is genRequest.trim for a chat: drop the oldest messages until the prompt is within budget tokens.
func chatTrim(msgs []engine.Message) func(m *engine.Model, budget int) ([]int, engine.Trimmed, error) {
	return func(m *engine.Model, budget int) ([]int, engine.Trimmed, error) {
		kept, t, err := m.TrimMessages(msgs, budget)
		if err != nil {
			return nil, t, err
		}
		return m.Tokenize(m.ChatPrompt(kept)), t, nil
	}
}

// answerReserve is how many tokens a truncated prompt leaves for the answer when the request doesn't set a maximum:
// a quarter of the window, at most 1024.
func answerReserve(numCtx, maxTokens int) int {
	if maxTokens > 0 {
		return maxTokens
	}
	return max(1, min(numCtx/4, 1024))
}

// fitContext makes the prompt fit the window: as it is (the default), or, when g.truncates, by dropping old messages
// so that reserve tokens are left for the answer. It never cuts a prompt silently: a prompt that doesn't fit is a
// *engine.ContextError (a 400), and a cut one is logged and reported in the answer.
func (s *Server) fitContext(m *engine.Model, g genRequest, ids []int, numCtx, modelCtx int) ([]int, engine.Trimmed, error) {
	reserve := 1
	truncate := s.cfg.Truncate
	if p := g.sampling.Truncate; p != nil {
		truncate = *p
	}
	truncate = truncate && g.trim != nil
	if truncate {
		reserve = answerReserve(numCtx, s.maxTokens(g))
	}
	if len(ids)+reserve <= numCtx {
		return ids, engine.Trimmed{}, nil
	}
	hint := func(err error) error {
		h := fmt.Sprintf(" (num_ctx %d, less %d kept for the answer)", numCtx, reserve)
		if allowed := min(modelCtx, s.cfg.MaxNumCtx); numCtx < allowed {
			h += fmt.Sprintf("; raise num_ctx (up to %d here)", allowed)
		}
		h += " or shorten the prompt"
		if g.trim != nil && !truncate {
			h += `, or set "truncate": true to drop the oldest messages`
		}
		return badRequest{fmt.Errorf("%w%s", err, h)}
	}
	if !truncate || numCtx-reserve < 1 {
		return nil, engine.Trimmed{}, hint(&engine.ContextError{PromptTokens: len(ids), Limit: numCtx - reserve})
	}
	trimmed, t, err := g.trim(m, numCtx-reserve)
	if err != nil {
		return nil, t, hint(err)
	}
	s.cfg.Logf("truncated a prompt for %s: dropped the %d oldest messages (%d of %d tokens) to fit num_ctx %d less %d for the answer",
		g.model, t.Messages, t.Tokens, t.PromptTokens, numCtx, reserve)
	return trimmed, t, nil
}

// maxTokens is the answer length a request asks for, within the server's Config.MaxTokens (0: fill the window).
func (s *Server) maxTokens(g genRequest) int {
	mt := g.sampling.MaxTokens
	if s.cfg.MaxTokens > 0 && (mt <= 0 || mt > s.cfg.MaxTokens) {
		mt = s.cfg.MaxTokens
	}
	return mt
}
