package server

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testWeather = `{"type":"function","function":{"name":"get_weather","description":"Current weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}`
const testTime = `{"type":"function","function":{"name":"get_time","description":"Current time in an IANA time zone","parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}}}`

// openaiCallsOf reads the tool calls of a non-streamed OpenAI answer, checking each one's shape.
func openaiCallsOf(t *testing.T, out map[string]any) (calls []map[string]any, finish string, content any) {
	t.Helper()
	ch := out["choices"].([]any)[0].(map[string]any)
	msg := ch["message"].(map[string]any)
	list, _ := msg["tool_calls"].([]any)
	for _, c := range list {
		c := c.(map[string]any)
		fn := c["function"].(map[string]any)
		var args map[string]any
		if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil {
			t.Errorf("arguments aren't a JSON object: %v %v", fn["arguments"], err)
		}
		if !strings.HasPrefix(c["id"].(string), "call_") || c["type"] != "function" {
			t.Errorf("call shape: %v", c)
		}
		calls = append(calls, map[string]any{"name": fn["name"], "args": args, "raw": c})
	}
	return calls, ch["finish_reason"].(string), msg["content"]
}

// TestTools runs tool calls against a real model: GGUF=/path/to/qwen3-1.7b.gguf go test ./server -run Tools
func TestTools(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	dir := t.TempDir()
	if err := os.Symlink(p, filepath.Join(dir, "test.gguf")); err != nil {
		t.Fatal(err)
	}
	ts := newTest(t, dir)
	model := filepath.Join(dir, "test.gguf")
	tools := `[` + testWeather + `,` + testTime + `]`
	chat := func(extra, messages string) map[string]any {
		t.Helper()
		code, out := call(t, ts, "POST", "/v1/chat/completions", `{"model":`+jsonQ(model)+`,"temperature":0,"max_tokens":120,"tools":`+tools+extra+`,"messages":`+messages+`}`)
		if code != 200 {
			t.Fatalf("%s: %d %v", extra, code, out)
		}
		return out
	}
	ask := `[{"role":"user","content":"What's the weather in Paris right now?"}]`

	t.Run("auto", func(t *testing.T) {
		calls, finish, _ := openaiCallsOf(t, chat("", ask))
		if len(calls) == 0 || calls[0]["name"] != "get_weather" || finish != "tool_calls" {
			t.Fatalf("got %v, finish %s", calls, finish)
		}
		if city, _ := calls[0]["args"].(map[string]any)["city"].(string); !strings.Contains(city, "Paris") {
			t.Errorf("city: %v", calls[0]["args"])
		}
	})
	t.Run("auto, no tool needed", func(t *testing.T) {
		calls, finish, content := openaiCallsOf(t, chat("", `[{"role":"user","content":"Say hello in French, nothing else."}]`))
		if len(calls) != 0 || finish != "stop" || content == nil || content == "" {
			t.Errorf("got %v %s %v", calls, finish, content)
		}
	})
	t.Run("required", func(t *testing.T) {
		calls, finish, _ := openaiCallsOf(t, chat(`,"tool_choice":"required"`, `[{"role":"user","content":"Say hello in French."}]`))
		if len(calls) == 0 || finish != "tool_calls" {
			t.Errorf("got %v %s", calls, finish)
		}
	})
	t.Run("named", func(t *testing.T) {
		calls, _, _ := openaiCallsOf(t, chat(`,"tool_choice":{"type":"function","function":{"name":"get_time"}}`, ask))
		if len(calls) == 0 || calls[0]["name"] != "get_time" {
			t.Fatalf("got %v", calls)
		}
		if _, ok := calls[0]["args"].(map[string]any)["zone"].(string); !ok {
			t.Errorf("zone: %v", calls[0]["args"])
		}
	})
	t.Run("none", func(t *testing.T) {
		calls, finish, content := openaiCallsOf(t, chat(`,"tool_choice":"none"`, ask))
		if len(calls) != 0 || finish == "tool_calls" || strings.Contains(fmt.Sprint(content), "<tool_call>") {
			t.Errorf("got %v %s %v", calls, finish, content)
		}
	})
	t.Run("sampled arguments match the schema", func(t *testing.T) {
		for seed := 1; seed <= 4; seed++ {
			code, out := call(t, ts, "POST", "/v1/chat/completions", fmt.Sprintf(`{"model":%s,"temperature":1.2,"seed":%d,"max_tokens":120,
				"tool_choice":"required","tools":%s,"messages":%s}`, jsonQ(model), seed, tools, ask))
			if code != 200 {
				t.Fatalf("%d %v", code, out)
			}
			calls, _, _ := openaiCallsOf(t, out)
			for _, c := range calls {
				args := c["args"].(map[string]any)
				switch c["name"] {
				case "get_weather":
					if _, ok := args["city"].(string); !ok {
						t.Errorf("seed %d: %v", seed, args)
					}
					if u, ok := args["unit"]; ok && u != "celsius" && u != "fahrenheit" {
						t.Errorf("seed %d: unit %v", seed, u)
					}
				case "get_time":
					if _, ok := args["zone"].(string); !ok {
						t.Errorf("seed %d: %v", seed, args)
					}
				default:
					t.Errorf("seed %d: tool %v", seed, c["name"])
				}
			}
			if len(calls) == 0 {
				t.Errorf("seed %d: no call", seed)
			}
		}
	})
	t.Run("round trip", func(t *testing.T) {
		calls, _, _ := openaiCallsOf(t, chat("", ask))
		if len(calls) == 0 {
			t.Fatal("no call")
		}
		raw := calls[0]["raw"].(map[string]any)
		asst, _ := json.Marshal(map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{raw}})
		msgs := `[{"role":"user","content":"What's the weather in Paris right now?"},` + string(asst) +
			`,{"role":"tool","tool_call_id":"` + raw["id"].(string) + `","content":"{\"temp_c\": 18, \"sky\": \"sunny\"}"}]`
		calls, finish, content := openaiCallsOf(t, chat("", msgs))
		if len(calls) != 0 || finish != "stop" || !strings.Contains(fmt.Sprint(content), "18") {
			t.Errorf("got %v %s %q", calls, finish, content)
		}
	})
	t.Run("several calls", func(t *testing.T) {
		calls, finish, _ := openaiCallsOf(t, chat("", `[{"role":"user","content":"What's the weather in Paris, and in Tokyo?"}]`))
		t.Logf("%d calls: %v", len(calls), calls)
		if len(calls) == 0 || finish != "tool_calls" {
			t.Errorf("got %v %s", calls, finish)
		}
	})
	t.Run("openai stream", func(t *testing.T) {
		res, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":`+jsonQ(model)+`,"stream":true,
			"temperature":0,"max_tokens":120,"tools":`+tools+`,"messages":`+ask+`}`))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		var names []string
		finish := ""
		sc := bufio.NewScanner(res.Body)
		for sc.Scan() {
			line, ok := strings.CutPrefix(sc.Text(), "data: ")
			if !ok || line == "[DONE]" {
				continue
			}
			var ev struct {
				Choices []struct {
					Delta struct {
						Content   string
						ToolCalls []struct {
							Index    int
							ID       string
							Function struct{ Name, Arguments string }
						} `json:"tool_calls"`
					}
					FinishReason *string `json:"finish_reason"`
				}
			}
			if err := json.Unmarshal([]byte(line), &ev); err != nil || len(ev.Choices) == 0 {
				continue
			}
			c := ev.Choices[0]
			if strings.Contains(c.Delta.Content, "<tool_call") {
				t.Errorf("tag in content: %q", c.Delta.Content)
			}
			for _, tc := range c.Delta.ToolCalls {
				if !json.Valid([]byte(tc.Function.Arguments)) || tc.ID == "" {
					t.Errorf("streamed call: %+v", tc)
				}
				names = append(names, tc.Function.Name)
			}
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
		}
		if len(names) == 0 || names[0] != "get_weather" || finish != "tool_calls" {
			t.Errorf("got %v, finish %q", names, finish)
		}
	})
	t.Run("ollama", func(t *testing.T) {
		code, out := call(t, ts, "POST", "/api/chat", `{"model":`+jsonQ(model)+`,"stream":false,"options":{"temperature":0,"num_predict":120},
			"tools":`+tools+`,"messages":`+ask+`}`)
		if code != 200 {
			t.Fatalf("%d %v", code, out)
		}
		msg := out["message"].(map[string]any)
		list, _ := msg["tool_calls"].([]any)
		if len(list) == 0 {
			t.Fatalf("no calls: %v", msg)
		}
		fn := list[0].(map[string]any)["function"].(map[string]any)
		if fn["name"] != "get_weather" {
			t.Errorf("call: %v", fn)
		}
		args, ok := fn["arguments"].(map[string]any)
		if !ok || args["city"] == nil {
			t.Errorf("arguments: %v", fn["arguments"])
		}
		// round trip, Ollama style: the call back with object arguments, then the tool's answer
		asst, _ := json.Marshal(map[string]any{"role": "assistant", "content": "", "tool_calls": list})
		code, out = call(t, ts, "POST", "/api/chat", `{"model":`+jsonQ(model)+`,"stream":false,"options":{"temperature":0,"num_predict":120},
			"tools":`+tools+`,"messages":[{"role":"user","content":"What's the weather in Paris right now?"},`+string(asst)+
			`,{"role":"tool","tool_name":"get_weather","content":"{\"temp_c\": 18, \"sky\": \"sunny\"}"}]}`)
		msg = out["message"].(map[string]any)
		if code != 200 || msg["tool_calls"] != nil || !strings.Contains(msg["content"].(string), "18") {
			t.Errorf("round trip: %d %v", code, out)
		}
	})
	t.Run("ollama stream", func(t *testing.T) {
		res, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"model":`+jsonQ(model)+`,
			"options":{"temperature":0,"num_predict":120},"tools":`+tools+`,"messages":`+ask+`}`))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		calls, done := 0, false
		sc := bufio.NewScanner(res.Body)
		for sc.Scan() {
			var ev struct {
				Message struct {
					Content   string
					ToolCalls []struct {
						Function struct {
							Name      string
							Arguments map[string]any
						}
					} `json:"tool_calls"`
				}
				Done bool
			}
			if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
				t.Fatalf("%s: %v", sc.Text(), err)
			}
			if strings.Contains(ev.Message.Content, "<tool_call") {
				t.Errorf("tag in content: %q", ev.Message.Content)
			}
			for _, c := range ev.Message.ToolCalls {
				if c.Function.Name != "get_weather" || c.Function.Arguments["city"] == nil {
					t.Errorf("call: %+v", c)
				}
				calls++
			}
			done = done || ev.Done
		}
		if calls == 0 || !done {
			t.Errorf("calls %d, done %v", calls, done)
		}
	})
}
