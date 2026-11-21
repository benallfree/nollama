package engine

import (
	"errors"
	"fmt"
)

// ErrContextExceeded is what every context-window error matches with errors.Is: the prompt, plus the room asked for
// the answer, doesn't fit. nollama never cuts a prompt to make it fit unless asked to (TrimMessages).
var ErrContextExceeded = errors.New("context window exceeded")

// ContextError says by how much a request missed the context window. Get it with errors.As; it also matches
// ErrContextExceeded.
type ContextError struct {
	PromptTokens int // tokens in the prompt
	MaxTokens    int // tokens asked for the answer (0: none counted)
	Limit        int // tokens that fit: the context window, or less what's kept for the answer when MaxTokens is 0
}

func (e *ContextError) Error() string {
	if e.MaxTokens > 0 {
		return fmt.Sprintf("context window exceeded: the prompt is %d tokens and %d more were asked for the answer, %d in all, but the context window is %d tokens",
			e.PromptTokens, e.MaxTokens, e.PromptTokens+e.MaxTokens, e.Limit)
	}
	return fmt.Sprintf("context window exceeded: the prompt is %d tokens, more than the %d that fit", e.PromptTokens, e.Limit)
}

// Is makes errors.Is(err, ErrContextExceeded) true.
func (e *ContextError) Is(target error) bool { return target == ErrContextExceeded }

// checkContext is nil when prompt tokens plus maxTokens fit in limit.
func checkContext(prompt, maxTokens, limit int) error {
	if prompt+maxTokens > limit {
		return &ContextError{PromptTokens: prompt, MaxTokens: maxTokens, Limit: limit}
	}
	return nil
}

// Trimmed says what TrimMessages dropped to make a conversation fit.
type Trimmed struct {
	Messages     int // messages dropped (the oldest ones after the system prompt)
	PromptTokens int // tokens before trimming
	Tokens       int // tokens dropped: PromptTokens minus the trimmed prompt's
}

// TrimMessages makes a conversation fit in budget tokens by dropping its oldest messages. It keeps the leading system
// messages and the last message, drops whole messages only, and never leaves an assistant (or tool) turn first after
// the system prompt. count is the prompt's length in tokens for a list of messages (the chat template applied).
// It returns the messages unchanged when they already fit, and a *ContextError when even the system prompt plus the
// last message don't.
func TrimMessages(msgs []Message, budget int, count func([]Message) int) ([]Message, Trimmed, error) {
	full := count(msgs)
	t := Trimmed{PromptTokens: full}
	if full <= budget {
		return msgs, t, nil
	}
	sys := 0
	for sys < len(msgs) && msgs[sys].Role == "system" {
		sys++
	}
	keep := func(drop int) []Message { // msgs without drop messages after the system prompt
		out := append([]Message(nil), msgs[:sys]...)
		rest := msgs[sys+drop:]
		for len(rest) > 1 && rest[0].Role != "user" { // start the kept turns with the user
			rest = rest[1:]
		}
		return append(out, rest...)
	}
	// the prompt shrinks as more is dropped: find the fewest drops that fit
	lo, hi := 1, len(msgs)-sys-1 // hi: everything but the last message
	if hi < 1 {
		return nil, t, &ContextError{PromptTokens: full, Limit: budget}
	}
	if n := count(keep(hi)); n > budget {
		return nil, t, &ContextError{PromptTokens: n, Limit: budget}
	}
	for lo < hi {
		mid := (lo + hi) / 2
		if count(keep(mid)) <= budget {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	out := keep(lo)
	t.Messages = len(msgs) - len(out)
	t.Tokens = full - count(out)
	return out, t, nil
}

// TrimMessages is the package function with this model's chat template (ChatPrompt) and tokenizer.
func (m *Model) TrimMessages(msgs []Message, budget int) ([]Message, Trimmed, error) {
	return TrimMessages(msgs, budget, func(ms []Message) int { return len(m.Tokenize(ChatPrompt(ms))) })
}
