package engine

import (
	"context"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"
)

// generateScripted answers with LoadOptions.Respond instead of running the weights. It keeps Generate's contract:
// the reply is tokenized with the model's tokenizer and ends at an end-of-turn token, a stop string or MaxTokens, the
// Constraint is checked, and onToken gets the text in pieces. The prompt cache and threads are not used.
func (m *Model) generateScripted(ctx context.Context, ids []int, o GenerateOptions, onToken func(piece string) bool) (Result, error) {
	start := time.Now()
	res := Result{PromptTokens: len(ids), StopReason: "eos"}
	reply := m.respond(m.Detokenize(ids))

	var raw []byte
	for _, tok := range m.Tokenize(reply) {
		if m.tok.eos[tok] && !o.IgnoreEOS {
			break
		}
		if res.OutputTokens >= o.MaxTokens {
			res.StopReason = "length"
			break
		}
		next := append(raw, m.tok.tokenBytes(tok)...)
		if o.Constraint != nil {
			switch o.Constraint(string(next)) {
			case FitNo:
				return res, fmt.Errorf("engine: the scripted reply %q does not fit the constraint after %q", reply, raw)
			case FitDone:
				raw = next
				res.Tokens = append(res.Tokens, tok)
				res.OutputTokens++
				res.StopReason = "stop"
			}
			if res.StopReason == "stop" {
				break
			}
		}
		raw = next
		res.Tokens = append(res.Tokens, tok)
		res.OutputTokens++
	}
	if o.Constraint != nil && res.StopReason == "eos" && len(raw) > 0 {
		if f := o.Constraint(string(raw)); f != FitComplete && f != FitDone {
			return res, fmt.Errorf("engine: the scripted reply %q is not complete for the constraint", reply)
		}
	}
	text := string(raw)
	for _, st := range o.Stop {
		if st == "" {
			continue
		}
		if i := strings.Index(text, st); i >= 0 {
			text = text[:i]
			res.StopReason = "stop"
		}
	}
	res.Prefill = time.Since(start)
	res.Text = text

	// stream it a token's worth at a time, never splitting a character
	if onToken != nil {
		emitted := 0
		for _, tok := range res.Tokens {
			if emitted >= len(text) {
				break
			}
			if err := ctx.Err(); err != nil {
				res.StopReason = "canceled"
				return res, err
			}
			end := min(len(text), emitted+len(m.tok.tokenBytes(tok)))
			for end < len(text) && !utf8.RuneStart(text[end]) {
				end++
			}
			if !onToken(text[emitted:end]) {
				res.StopReason = "callback"
				return res, nil
			}
			emitted = end
		}
		if emitted < len(text) {
			onToken(text[emitted:])
		}
	}
	return res, nil
}
