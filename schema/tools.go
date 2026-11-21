package schema

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/benallfree/nollama/engine"
)

// Tool calls in Qwen3's format: the model writes
//
//	<tool_call>
//	{"name": "get_weather", "arguments": {"city": "Paris"}}
//	</tool_call>
//
// once per call, after any text. Tools compiles a constraint that keeps every call to a tool that exists, with
// arguments that match that tool's parameter schema.

// The tags around each call.
const (
	CallOpen  = "<tool_call>"
	CallClose = "</tool_call>"
)

// Tool is a function the model may call: its name and its parameters' JSON schema (nil: any object).
type Tool struct {
	Name       string
	Parameters json.RawMessage
}

// ToolCall is one call the model wrote.
type ToolCall struct {
	Name      string
	Arguments json.RawMessage // a JSON object
}

// Tools is a compiled set of tools; pass its Fit as engine.GenerateOptions.Constraint.
type Tools struct {
	call     *schemaNode // {"name": <a tool's name>, "arguments": <that tool's parameters>}
	required bool
}

// CompileTools builds the constraint for tools. Unless required, the model writes freely (Fit says engine.FitFree)
// until it opens a call; from there each call must match a tool, and only more calls or the end may follow. With
// required, the answer is calls from the first token. To force one tool, pass only that tool, with required.
func CompileTools(tools []Tool, required bool) (*Tools, error) {
	if len(tools) == 0 {
		return nil, errors.New("schema: no tools")
	}
	t := &Tools{call: &schemaNode{maxItems: -1}, required: required}
	seen := map[string]bool{}
	for _, tool := range tools {
		if tool.Name == "" {
			return nil, errors.New("schema: a tool has no name")
		}
		if seen[tool.Name] {
			return nil, fmt.Errorf("schema: tool %q is listed twice", tool.Name)
		}
		seen[tool.Name] = true
		params := tool.Parameters
		if len(params) == 0 || string(params) == "null" {
			params = json.RawMessage(`{"type":"object"}`)
		}
		args, err := Compile(params)
		if err != nil {
			return nil, fmt.Errorf("tool %s: %w", tool.Name, err)
		}
		if len(args.types) == 0 && len(args.anyOf) == 0 && args.literals == nil {
			args.types = map[string]bool{"object": true} // arguments are always an object
		}
		name, _ := json.Marshal(tool.Name)
		t.call.anyOf = append(t.call.anyOf, &schemaNode{
			types:    map[string]bool{"object": true},
			props:    map[string]*schemaNode{"name": {literals: []string{string(name)}, maxItems: -1}, "arguments": args},
			order:    []string{"name", "arguments"},
			ordered:  true, // the name first, so the arguments follow the tool it picked
			required: map[string]bool{"name": true, "arguments": true},
			closed:   true,
			maxItems: -1,
		})
	}
	return t, nil
}

// Fit judges the answer so far.
func (t *Tools) Fit(text string) engine.Fit {
	i := strings.Index(text, CallOpen)
	if !t.required {
		if i < 0 {
			return engine.FitFree
		}
		return t.calls(text[i+len(CallOpen):])
	}
	// required: a call from the start (a little whitespace allowed)
	p := &jparser{s: text}
	if !p.ws() {
		return engine.FitNo
	}
	rest := text[p.i:]
	if len(rest) < len(CallOpen) {
		if strings.HasPrefix(CallOpen, rest) {
			return engine.FitPrefix
		}
		return engine.FitNo
	}
	if !strings.HasPrefix(rest, CallOpen) {
		return engine.FitNo
	}
	return t.calls(rest[len(CallOpen):])
}

// calls judges what follows the first open tag: a call, its close tag, then more calls or the end.
func (t *Tools) calls(s string) engine.Fit {
	for {
		p := &jparser{s: s}
		if !p.ws() {
			return engine.FitNo
		}
		switch p.value(t.call) {
		case resMore:
			return engine.FitPrefix
		case resBad:
			return engine.FitNo
		}
		if !p.ws() {
			return engine.FitNo
		}
		f, n := tag(s[p.i:], CallClose)
		if f != engine.FitDone {
			return f
		}
		p.i += n
		if !p.ws() {
			return engine.FitNo
		}
		if p.eof() {
			return engine.FitComplete // the end, or another call
		}
		f, n = tag(s[p.i:], CallOpen)
		if f != engine.FitDone {
			return f
		}
		s = s[p.i+n:]
	}
}

// tag matches want at the start of s: FitDone (and its length) when it's all there, FitPrefix when s ends inside it.
func tag(s, want string) (engine.Fit, int) {
	if strings.HasPrefix(s, want) {
		return engine.FitDone, len(want)
	}
	if strings.HasPrefix(want, s) {
		return engine.FitPrefix, 0
	}
	return engine.FitNo, 0
}

// SplitToolCalls reads an answer written under Tools: the text before the first call (trailing whitespace trimmed)
// and every complete call after it. A call cut off before its close tag is left out.
func SplitToolCalls(text string) (content string, calls []ToolCall) {
	i := strings.Index(text, CallOpen)
	if i < 0 {
		return text, nil
	}
	content = strings.TrimRight(text[:i], " \t\r\n")
	rest := text[i:]
	for {
		j := strings.Index(rest, CallOpen)
		if j < 0 {
			break
		}
		rest = rest[j+len(CallOpen):]
		k := strings.Index(rest, CallClose)
		if k < 0 {
			break
		}
		var c struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if json.Unmarshal([]byte(rest[:k]), &c) == nil && c.Name != "" {
			if len(c.Arguments) == 0 || string(c.Arguments) == "null" {
				c.Arguments = json.RawMessage(`{}`)
			}
			calls = append(calls, ToolCall{Name: c.Name, Arguments: c.Arguments})
		}
		rest = rest[k+len(CallClose):]
	}
	return content, calls
}
