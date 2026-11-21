package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTest(t *testing.T, dir string) *httptest.Server {
	t.Helper()
	s := New(Config{ModelsDir: dir})
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); _ = s.Close() })
	return ts
}

func call(t *testing.T, ts *httptest.Server, method, path, body string) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(method, ts.URL+path, strings.NewReader(body))
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(res.Body).Decode(&out)
	return res.StatusCode, out
}

func TestNoModels(t *testing.T) {
	ts := newTest(t, t.TempDir())
	res, err := http.Get(ts.URL + "/")
	if err != nil || res.StatusCode != 200 {
		t.Fatalf("GET /: %v %v", res, err)
	}
	if code, out := call(t, ts, "GET", "/api/version", ""); code != 200 || out["version"] != OllamaVersion {
		t.Errorf("version: %d %v", code, out)
	}
	if code, out := call(t, ts, "GET", "/api/tags", ""); code != 200 || len(out["models"].([]any)) != 0 {
		t.Errorf("tags: %d %v", code, out)
	}
	if code, out := call(t, ts, "GET", "/v1/models", ""); code != 200 || len(out["data"].([]any)) != 0 {
		t.Errorf("v1/models: %d %v", code, out)
	}
	// a model that isn't downloaded: Ollama's 404, so clients offer to pull it
	if code, out := call(t, ts, "POST", "/api/chat", `{"model":"qwen3:1.7b","messages":[{"role":"user","content":"hi"}]}`); code != 404 ||
		!strings.Contains(out["error"].(string), "pulling") {
		t.Errorf("chat, missing model: %d %v", code, out)
	}
	if code, out := call(t, ts, "POST", "/v1/chat/completions", `{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi"}]}`); code != 404 ||
		out["error"].(map[string]any)["type"] != "not_found_error" {
		t.Errorf("v1 chat, missing model: %d %v", code, out)
	}
	if code, _ := call(t, ts, "POST", "/api/chat", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[{}]}`); code != 400 {
		t.Errorf("tools: %d", code)
	}
	if code, _ := call(t, ts, "POST", "/api/generate", `{"model":"x","prompt":"hi","format":"yaml"}`); code != 400 && code != 404 {
		t.Errorf("bad format: %d", code)
	}
	if code, _ := call(t, ts, "POST", "/api/embed", `{}`); code != 501 {
		t.Errorf("embed: %d", code)
	}
}

func TestParseKeepAlive(t *testing.T) {
	for in, want := range map[string]time.Duration{`300`: 300 * time.Second, `"5m"`: 5 * time.Minute, `"-1"`: -time.Second, `0`: 0, `"10"`: 10 * time.Second} {
		d, err := parseKeepAlive(json.RawMessage(in))
		if err != nil || d == nil || *d != want {
			t.Errorf("%s: %v %v, want %s", in, d, err, want)
		}
	}
	if d, err := parseKeepAlive(nil); d != nil || err != nil {
		t.Errorf("absent: %v %v", d, err)
	}
	if _, err := parseKeepAlive(json.RawMessage(`"soon"`)); err == nil {
		t.Error("soon: want an error")
	}
}

// TestChat runs both APIs against a real model: GGUF=/path/to/qwen3-0.6b.gguf go test ./server
func TestChat(t *testing.T) {
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

	code, out := call(t, ts, "POST", "/api/chat", `{"model":"`+model+`","stream":false,"options":{"temperature":0,"num_predict":20},
		"format":{"type":"object","properties":{"answer":{"enum":["yes","no"]}},"required":["answer"]},
		"messages":[{"role":"user","content":"Is the sky blue? Answer yes or no."}]}`)
	if code != 200 || out["done"] != true {
		t.Fatalf("chat: %d %v", code, out)
	}
	var data struct{ Answer string }
	if err := json.Unmarshal([]byte(out["message"].(map[string]any)["content"].(string)), &data); err != nil || (data.Answer != "yes" && data.Answer != "no") {
		t.Errorf("structured answer: %v %v", out["message"], err)
	}

	res, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"`+model+`","stream":true,
		"max_tokens":5,"messages":[{"role":"user","content":"Hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var events []string
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		if line, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			events = append(events, line)
		}
	}
	if len(events) < 3 || events[len(events)-1] != "[DONE]" || !strings.Contains(events[len(events)-2], `"finish_reason":"length"`) {
		t.Errorf("stream: %v", events)
	}
}
