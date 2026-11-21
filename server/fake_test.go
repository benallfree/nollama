package server_test

import (
	"bufio"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/benallfree/nollama/nollamatest"
)

func post(t *testing.T, url, body string) (int, map[string]any) {
	t.Helper()
	res, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(res.Body).Decode(&out)
	return res.StatusCode, out
}

func lines(t *testing.T, url, body string) []string {
	t.Helper()
	res, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out []string
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		if l := strings.TrimPrefix(sc.Text(), "data: "); l != "" {
			out = append(out, l)
		}
	}
	return out
}

// TestTinyModel runs both APIs end to end on the tiny model: real loading, tokenizing, sampling and constraints.
func TestTinyModel(t *testing.T) {
	ts := nollamatest.NewServer(t)

	code, out := post(t, ts.URL+"/api/chat", `{"model":"qwen3:1.7b","stream":false,"options":{"temperature":0,"num_predict":40},
		"format":{"enum":["yes","no"]},"messages":[{"role":"user","content":"Is the sky blue?"}]}`)
	if code != 200 || out["done"] != true {
		t.Fatalf("chat: %d %v", code, out)
	}
	if c := strings.TrimSpace(out["message"].(map[string]any)["content"].(string)); c != `"yes"` && c != `"no"` {
		t.Errorf("structured answer: %q", c)
	}

	events := lines(t, ts.URL+"/v1/chat/completions", `{"model":"`+ts.Model+`","stream":true,"max_tokens":5,
		"messages":[{"role":"user","content":"Hi"}]}`)
	if len(events) < 3 || events[len(events)-1] != "[DONE]" || !strings.Contains(events[len(events)-2], `"finish_reason":"length"`) {
		t.Errorf("stream: %v", events)
	}

	code, out = post(t, ts.URL+"/api/generate", `{"model":"qwen3-0.6b","prompt":"Hi","stream":false,"options":{"num_predict":4}}`)
	if code != 200 || out["eval_count"] != float64(4) || out["done_reason"] != "length" {
		t.Errorf("generate: %d %v", code, out)
	}
	code, out = post(t, ts.URL+"/api/generate", `{"model":"qwen3-0.6b","prompt":"`+strings.Repeat("word ", 5000)+`","stream":false}`)
	if code != 400 {
		t.Errorf("prompt past the context window: %d %v", code, out)
	}
}

// TestScriptedServer checks what the server does with an answer, given a model whose answers are known.
func TestScriptedServer(t *testing.T) {
	ts := nollamatest.NewServer(t, nollamatest.WithResponder(func(p nollamatest.Prompt) string {
		if p.Last("system") != "" {
			return "system: " + p.Last("system")
		}
		return "Bonjour, ça va?"
	}))

	code, out := post(t, ts.URL+"/v1/chat/completions", `{"model":"qwen3-4b","messages":[{"role":"user","content":"Say hi in French."}]}`)
	choice := out["choices"].([]any)[0].(map[string]any)
	if code != 200 || choice["message"].(map[string]any)["content"] != "Bonjour, ça va?" || choice["finish_reason"] != "stop" {
		t.Errorf("v1 chat: %d %v", code, out)
	}
	usage := out["usage"].(map[string]any)
	if usage["completion_tokens"].(float64) < 2 || usage["prompt_tokens"].(float64) < 5 {
		t.Errorf("usage: %v", usage)
	}

	var got strings.Builder
	events := lines(t, ts.URL+"/api/chat", `{"model":"qwen3:4b","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}`)
	for _, e := range events {
		var ev struct {
			Message struct{ Content string }
			Done    bool
		}
		if err := json.Unmarshal([]byte(e), &ev); err != nil {
			t.Fatal(err)
		}
		got.WriteString(ev.Message.Content)
	}
	if got.String() != "system: be brief" || len(events) < 3 {
		t.Errorf("ollama stream: %q from %v", got.String(), events)
	}

	code, out = post(t, ts.URL+"/v1/chat/completions", `{"model":"qwen3-4b","stop":[","],"messages":[{"role":"user","content":"hi"}]}`)
	if c := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]; code != 200 || c != "Bonjour" {
		t.Errorf("stop: %d %v", code, out)
	}
}
