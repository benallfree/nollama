package server

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/benallfree/nollama/engine"
)

// fakeChat is a chat whose prompt is one token per word plus 2 per message, with a trim like chatTrim's.
func fakeChat(msgs []engine.Message) ([]int, func(*engine.Model, int) ([]int, engine.Trimmed, error)) {
	count := func(ms []engine.Message) int {
		n := 0
		for _, m := range ms {
			n += 2 + len(strings.Fields(m.Content))
		}
		return n
	}
	return make([]int, count(msgs)), func(_ *engine.Model, budget int) ([]int, engine.Trimmed, error) {
		kept, t, err := engine.TrimMessages(msgs, budget, count)
		return make([]int, count(kept)), t, err
	}
}

func TestFitContext(t *testing.T) {
	var logs []string
	s := New(Config{Logf: func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }})
	var msgs []engine.Message
	for range 10 {
		msgs = append(msgs, engine.Message{Role: "user", Content: strings.Repeat("w ", 18)}, engine.Message{Role: "assistant", Content: "ok"})
	}
	msgs = append(msgs, engine.Message{Role: "user", Content: "last"})
	ids, trim := fakeChat(msgs) // 10*(20+3) + 3 = 233 tokens
	yes, no := true, false

	// the default: a 400 with the counts, matching engine.ErrContextExceeded
	_, _, err := s.fitContext(nil, genRequest{model: "m", trim: trim}, ids, 100, 4096)
	var ce *engine.ContextError
	if statusOf(err) != 400 || !errors.As(err, &ce) || !errors.Is(err, engine.ErrContextExceeded) || ce.PromptTokens != 233 || ce.Limit != 99 {
		t.Fatalf("default: %v", err)
	}
	for _, want := range []string{"233", "99", "num_ctx 100", "up to 4096 here", `"truncate": true`} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("message %q lacks %q", err, want)
		}
	}
	// a prompt that fits, with room for one token, is left alone
	if got, tr, err := s.fitContext(nil, genRequest{trim: trim}, ids, 234, 4096); err != nil || len(got) != 233 || tr.Messages != 0 {
		t.Errorf("fits: %d %+v %v", len(got), tr, err)
	}
	// no trim (a raw prompt): no truncate hint, even when asked
	_, _, err = s.fitContext(nil, genRequest{sampling: sampling{Truncate: &yes}}, ids, 100, 100)
	if statusOf(err) != 400 || strings.Contains(err.Error(), "truncate") || strings.Contains(err.Error(), "up to") {
		t.Errorf("raw: %v", err)
	}

	// opt in per request: max_tokens 30 kept for the answer, so the prompt gets 70
	got, tr, err := s.fitContext(nil, genRequest{model: "m", trim: trim, sampling: sampling{Truncate: &yes, MaxTokens: 30}}, ids, 100, 4096)
	if err != nil || len(got) > 70 || tr.Messages == 0 || tr.Tokens != 233-len(got) || tr.PromptTokens != 233 {
		t.Fatalf("truncate: %d %+v %v", len(got), tr, err)
	}
	if len(logs) != 1 || !strings.Contains(logs[0], fmt.Sprintf("dropped the %d oldest messages", tr.Messages)) {
		t.Errorf("log: %q", logs)
	}
	// without max_tokens, a quarter of the window is kept for the answer
	if got, _, err := s.fitContext(nil, genRequest{trim: trim, sampling: sampling{Truncate: &yes}}, ids, 100, 4096); err != nil || len(got) > 75 {
		t.Errorf("reserve: %d %v", len(got), err)
	}

	// the server default, and a request turning it off
	st := New(Config{Truncate: true})
	if _, tr, err := st.fitContext(nil, genRequest{trim: trim}, ids, 100, 4096); err != nil || tr.Messages == 0 {
		t.Errorf("server default: %+v %v", tr, err)
	}
	if _, _, err := st.fitContext(nil, genRequest{trim: trim, sampling: sampling{Truncate: &no}}, ids, 100, 4096); !errors.Is(err, engine.ErrContextExceeded) {
		t.Errorf("truncate false: %v", err)
	}
	// even truncated, the system prompt and last message must fit
	if _, _, err := s.fitContext(nil, genRequest{trim: trim, sampling: sampling{Truncate: &yes, MaxTokens: 98}}, ids, 100, 4096); statusOf(err) != 400 || !errors.Is(err, engine.ErrContextExceeded) {
		t.Errorf("too small to trim: %v", err)
	}
}

func TestContextErrorShapes(t *testing.T) {
	err := badRequest{fmt.Errorf("%w; shorten it", &engine.ContextError{PromptTokens: 500, Limit: 99})}
	rec := httptest.NewRecorder()
	ollamaError(rec, err)
	var o map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &o)
	if rec.Code != 400 || !strings.Contains(o["error"].(string), "500 tokens") {
		t.Errorf("ollama: %d %s", rec.Code, rec.Body)
	}
	rec = httptest.NewRecorder()
	openaiError(rec, err)
	var a struct {
		Error struct{ Message, Type string }
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &a)
	if rec.Code != 400 || a.Error.Type != "invalid_request_error" || !strings.Contains(a.Error.Message, "500 tokens") {
		t.Errorf("openai: %d %s", rec.Code, rec.Body)
	}
}

func TestUsageFields(t *testing.T) {
	out := genOutput{reason: "length", res: engine.Result{PromptTokens: 12, CachedTokens: 4}, trimmed: engine.Trimmed{Messages: 2, Tokens: 30, PromptTokens: 42}}
	b, _ := json.Marshal(doneOf(out))
	for _, want := range []string{`"done_reason":"length"`, `"prompt_eval_count":12`, `"eval_count":0`, `"truncated":{"messages":2,"tokens":30,"prompt_tokens":42}`} {
		if !strings.Contains(string(b), want) {
			t.Errorf("ollama done %s lacks %s", b, want)
		}
	}
	u := usageOf(out)
	if u.PromptTokens != 12 || u.CompletionTokens != 0 || u.TotalTokens != 12 || u.PromptTokensDetails.CachedTokens != 4 {
		t.Errorf("usage: %+v", u)
	}
	if truncatedOf(genOutput{}) != nil {
		t.Error("untrimmed: want no truncated field")
	}
}

// TestContextLimits runs the context rules against a real model: GGUF=/path/to/qwen3-0.6b.gguf go test ./server
func TestContextLimits(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	dir := t.TempDir()
	if err := os.Symlink(p, filepath.Join(dir, "test.gguf")); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var logs []string
	s := New(Config{ModelsDir: dir, NumCtx: 256, Logf: func(f string, a ...any) {
		mu.Lock()
		logs = append(logs, fmt.Sprintf(f, a...))
		mu.Unlock()
	}})
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); _ = s.Close() })
	model := filepath.Join(dir, "test.gguf")
	var history []string
	for i := range 12 {
		history = append(history, fmt.Sprintf(`{"role":"user","content":"Remember fact %d: %s"},{"role":"assistant","content":"ok"}`, i, strings.Repeat("blue ", 15)))
	}
	long := `[{"role":"system","content":"Answer in one word."},` + strings.Join(history, ",") + `,{"role":"user","content":"Say hi."}]`

	// too long: 400 in each API's shape, with the counts
	code, out := call(t, ts, "POST", "/api/chat", `{"model":"`+model+`","stream":false,"messages":`+long+`}`)
	if code != 400 || !strings.Contains(out["error"].(string), "context window exceeded") || !strings.Contains(out["error"].(string), "num_ctx 256") {
		t.Errorf("ollama too long: %d %v", code, out)
	}
	code, out = call(t, ts, "POST", "/v1/chat/completions", `{"model":"`+model+`","messages":`+long+`}`)
	if e, _ := out["error"].(map[string]any); code != 400 || e["type"] != "invalid_request_error" || !strings.Contains(e["message"].(string), "context window exceeded") {
		t.Errorf("openai too long: %d %v", code, out)
	}
	code, out = call(t, ts, "POST", "/api/generate", `{"model":"`+model+`","stream":false,"prompt":"`+strings.Repeat("blue ", 400)+`"}`)
	if code != 400 || !strings.Contains(out["error"].(string), "context window exceeded") {
		t.Errorf("generate too long: %d %v", code, out)
	}

	// opt-in truncation: answered, with what was dropped in the answer and the log
	code, out = call(t, ts, "POST", "/api/chat", `{"model":"`+model+`","stream":false,"truncate":true,"options":{"temperature":0,"num_predict":4},"messages":`+long+`}`)
	tr, _ := out["truncated"].(map[string]any)
	if code != 200 || tr == nil || tr["messages"].(float64) < 2 || out["prompt_eval_count"].(float64) > 256-4 || out["done_reason"] != "length" || out["eval_count"] != 4.0 {
		t.Errorf("ollama truncate: %d %v", code, out)
	}
	code, out = call(t, ts, "POST", "/v1/chat/completions", `{"model":"`+model+`","truncate":true,"max_tokens":4,"temperature":0,"messages":`+long+`}`)
	u, _ := out["usage"].(map[string]any)
	if code != 200 || out["truncated"] == nil || u == nil || u["completion_tokens"] != 4.0 || u["total_tokens"].(float64) != u["prompt_tokens"].(float64)+4 {
		t.Errorf("openai truncate: %d %v", code, out)
	}
	mu.Lock()
	if n := len(logs); n == 0 || !strings.Contains(strings.Join(logs, "\n"), "truncated a prompt") {
		t.Errorf("no truncation log: %q", logs)
	}
	mu.Unlock()

	// length and usage when streaming
	lines := stream(t, ts.URL+"/api/chat", `{"model":"`+model+`","options":{"num_predict":3},"messages":[{"role":"user","content":"Count to ten."}]}`)
	var last map[string]any
	_ = json.Unmarshal([]byte(lines[len(lines)-1]), &last)
	if last["done"] != true || last["done_reason"] != "length" || last["eval_count"] != 3.0 || last["prompt_eval_count"].(float64) < 5 {
		t.Errorf("ollama stream: %v", last)
	}
	events := stream(t, ts.URL+"/v1/chat/completions", `{"model":"`+model+`","stream":true,"stream_options":{"include_usage":true},"max_tokens":3,"messages":[{"role":"user","content":"Count to ten."}]}`)
	if len(events) < 3 || events[len(events)-1] != "[DONE]" || !strings.Contains(events[len(events)-3], `"finish_reason":"length"`) ||
		!strings.Contains(events[len(events)-2], `"completion_tokens":3`) {
		t.Errorf("openai stream: %v", events)
	}
	events = stream(t, ts.URL+"/v1/completions", `{"model":"`+model+`","stream":true,"stream_options":{"include_usage":true},"max_tokens":3,"prompt":"One, two,"}`)
	if len(events) < 3 || !strings.Contains(events[len(events)-2], `"completion_tokens":3`) {
		t.Errorf("completions stream: %v", events)
	}
	// the window's end, not max_tokens: still "length"
	code, out = call(t, ts, "POST", "/api/generate", `{"model":"`+model+`","stream":false,"raw":true,"options":{"num_ctx":24},"prompt":"One, two, three, four,"}`)
	if code != 200 || out["done_reason"] != "length" || out["prompt_eval_count"].(float64)+out["eval_count"].(float64) != 24 {
		t.Errorf("window end: %d %v", code, out)
	}
}

// stream posts body and returns the response's lines (SSE data, or NDJSON).
func stream(t *testing.T, url, body string) []string {
	t.Helper()
	res, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var lines []string
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		line := sc.Text()
		if d, ok := strings.CutPrefix(line, "data: "); ok {
			line = d
		}
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}
