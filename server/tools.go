package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/engine/schema"
)

// Tool calling, the same in both APIs: the tools go into the prompt, and the answer is held to calls that name a listed
// tool with arguments matching its parameter schema (schema.CompileTools), so a call always parses.

// toolSet is a request's tools and tool_choice.
type toolSet struct {
	defs   []json.RawMessage // {"type": "function", "function": {...}}, as sent, for the prompt
	tools  []schema.Tool
	choice string // "auto", "none" or "required"
	name   string // with "required": the one function to call ("": any)
}

// parseTools reads "tools" and "tool_choice" (OpenAI's: "auto", "none", "required" or
// {"type": "function", "function": {"name": ...}}). nil when there are no tools.
func parseTools(raw []json.RawMessage, choice json.RawMessage) (*toolSet, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	ts := &toolSet{choice: "auto"}
	for i, r := range raw {
		var d struct {
			Type     string `json:"type"`
			Function *struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
		}
		if err := json.Unmarshal(r, &d); err != nil {
			return nil, badRequest{fmt.Errorf("tools[%d]: %w", i, err)}
		}
		if d.Function == nil {
			return nil, badRequest{fmt.Errorf("tools[%d]: want {\"type\": \"function\", \"function\": {\"name\": ..., \"parameters\": ...}}", i)}
		}
		if d.Type != "" && d.Type != "function" {
			return nil, badRequest{fmt.Errorf("tools[%d]: type %q isn't supported; only functions are", i, d.Type)}
		}
		if d.Function.Name == "" {
			return nil, badRequest{fmt.Errorf("tools[%d]: function.name is required", i)}
		}
		ts.defs = append(ts.defs, r)
		ts.tools = append(ts.tools, schema.Tool{Name: d.Function.Name, Parameters: d.Function.Parameters})
	}
	if len(choice) > 0 && string(choice) != "null" {
		var s string
		if json.Unmarshal(choice, &s) == nil {
			switch s {
			case "auto", "none", "required":
				ts.choice = s
			default:
				return nil, badRequest{fmt.Errorf("tool_choice %q: want auto, none, required or a function", s)}
			}
		} else {
			var c struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if err := json.Unmarshal(choice, &c); err != nil || c.Function.Name == "" {
				return nil, badRequest{errors.New(`tool_choice: want auto, none, required or {"type": "function", "function": {"name": ...}}`)}
			}
			found := false
			for _, t := range ts.tools {
				found = found || t.Name == c.Function.Name
			}
			if !found {
				return nil, badRequest{fmt.Errorf("tool_choice names %q, which isn't in tools", c.Function.Name)}
			}
			ts.choice, ts.name = "required", c.Function.Name
		}
	}
	// compile now, so a bad parameter schema is the client's error before any model loads
	if _, err := ts.constraint(); err != nil {
		return nil, badRequest{err}
	}
	return ts, nil
}

// calling says the model may call tools in this answer.
func (ts *toolSet) calling() bool { return ts != nil && ts.choice != "none" }

// prompt is the tool definitions for the chat template (none when tool_choice is "none").
func (ts *toolSet) prompt() []json.RawMessage {
	if !ts.calling() {
		return nil
	}
	return ts.defs
}

// constraint holds the answer to well-formed calls.
func (ts *toolSet) constraint() (func(string) engine.Fit, error) {
	tools := ts.tools
	if ts.name != "" {
		for _, t := range ts.tools {
			if t.Name == ts.name {
				tools = []schema.Tool{t}
			}
		}
	}
	c, err := schema.CompileTools(tools, ts.choice == "required")
	if err != nil {
		return nil, err
	}
	return c.Fit, nil
}

// callID names a tool call, for clients that match results to calls by id.
func callID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}

// argsText is a call's arguments as the chat template writes them back: OpenAI sends a JSON string holding them,
// Ollama an object.
func argsText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	if len(raw) == 0 || string(raw) == "null" {
		return "{}"
	}
	return engine.PyJSON(raw)
}

// toolStream splits a streamed answer into text to pass on now and tool calls as each one closes. It holds back what
// could be the start of a call (a partial <tool_call>, whitespace before one).
type toolStream struct {
	text  strings.Builder
	sent  int // bytes of content passed on
	calls int // calls passed on
}

func (t *toolStream) feed(piece string) (content string, calls []schema.ToolCall) {
	t.text.WriteString(piece)
	full := t.text.String()
	if !strings.Contains(full, schema.CallOpen) {
		safe := len(full)
		for k := min(len(schema.CallOpen)-1, len(full)); k > 0; k-- {
			if strings.HasSuffix(full, schema.CallOpen[:k]) {
				safe -= k
				break
			}
		}
		safe = len(strings.TrimRight(full[:safe], " \t\r\n"))
		if safe > t.sent {
			content = full[t.sent:safe]
			t.sent = safe
		}
		return content, nil
	}
	c, all := schema.SplitToolCalls(full)
	if len(c) > t.sent {
		content = c[t.sent:]
		t.sent = len(c)
	}
	calls = all[t.calls:]
	t.calls = len(all)
	return content, calls
}

// rest is the content still held back once the answer is over.
func (t *toolStream) rest() string {
	full := t.text.String()
	if strings.Contains(full, schema.CallOpen) || t.sent >= len(full) {
		return ""
	}
	s := full[t.sent:]
	t.sent = len(full)
	return s
}
