package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newServer(t *testing.T, cfg Config) *httptest.Server {
	t.Helper()
	if cfg.ModelsDir == "" {
		cfg.ModelsDir = t.TempDir()
	}
	s := New(cfg)
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); _ = s.Close() })
	return ts
}

func do(t *testing.T, ts *httptest.Server, method, path, body string, hdr map[string]string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(method, ts.URL+path, strings.NewReader(body))
	for k, v := range hdr {
		if k == "Host" {
			req.Host = v
		} else {
			req.Header.Set(k, v)
		}
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	return res
}

func TestAPIKey(t *testing.T) {
	ts := newServer(t, Config{APIKey: "s3cret"})
	if res := do(t, ts, "GET", "/", "", nil); res.StatusCode != 200 {
		t.Errorf("GET / without a key: %d, want 200 (health checks)", res.StatusCode)
	}
	for _, p := range []string{"/api/tags", "/api/version", "/v1/models"} {
		if res := do(t, ts, "GET", p, "", nil); res.StatusCode != 401 {
			t.Errorf("%s without a key: %d, want 401", p, res.StatusCode)
		}
		if res := do(t, ts, "GET", p, "", map[string]string{"Authorization": "Bearer wrong"}); res.StatusCode != 401 {
			t.Errorf("%s with a wrong key: %d, want 401", p, res.StatusCode)
		}
		if res := do(t, ts, "GET", p, "", map[string]string{"Authorization": "Bearer s3cret"}); res.StatusCode != 200 {
			t.Errorf("%s with the key: %d, want 200", p, res.StatusCode)
		}
		if res := do(t, ts, "GET", p, "", map[string]string{"X-API-Key": "s3cret"}); res.StatusCode != 200 {
			t.Errorf("%s with X-API-Key: %d, want 200", p, res.StatusCode)
		}
	}
	if res := do(t, ts, "POST", "/api/pull", `{"model":"qwen3-0.6b"}`, nil); res.StatusCode != 401 {
		t.Errorf("pull without a key: %d, want 401", res.StatusCode)
	}
}

func TestHostHeader(t *testing.T) {
	ts := newServer(t, Config{})
	for host, want := range map[string]int{"evil.example": 403, "evil.example:11434": 403, "localhost:11434": 200,
		"app.localhost": 200, "127.0.0.1:11434": 200, "[::1]:11434": 200, "192.168.1.5:11434": 200} {
		if res := do(t, ts, "GET", "/api/tags", "", map[string]string{"Host": host}); res.StatusCode != want {
			t.Errorf("Host %s: %d, want %d", host, res.StatusCode, want)
		}
	}
	ts = newServer(t, Config{AllowedHosts: []string{"llm.internal", ".corp.example"}})
	for host, want := range map[string]int{"llm.internal": 200, "a.corp.example": 200, "corp.example.evil": 403} {
		if res := do(t, ts, "GET", "/api/tags", "", map[string]string{"Host": host}); res.StatusCode != want {
			t.Errorf("allowed hosts, Host %s: %d, want %d", host, res.StatusCode, want)
		}
	}
	ts = newServer(t, Config{APIKey: "k"})
	if res := do(t, ts, "GET", "/api/tags", "", map[string]string{"Host": "evil.example", "Authorization": "Bearer k"}); res.StatusCode != 200 {
		t.Errorf("any Host with an API key: %d, want 200", res.StatusCode)
	}
}

func TestOrigin(t *testing.T) {
	ts := newServer(t, Config{})
	// a web page posting text/plain needs no preflight, so the Origin check is what stops it
	if res := do(t, ts, "POST", "/api/pull", `{"model":"qwen3-0.6b"}`, map[string]string{"Origin": "https://evil.example", "Content-Type": "text/plain"}); res.StatusCode != 403 {
		t.Errorf("pull from a web page: %d, want 403", res.StatusCode)
	}
	res := do(t, ts, "GET", "/api/tags", "", map[string]string{"Origin": "http://localhost:5173"})
	if res.StatusCode != 200 || res.Header.Get("Access-Control-Allow-Origin") != "http://localhost:5173" {
		t.Errorf("local page: %d %q", res.StatusCode, res.Header.Get("Access-Control-Allow-Origin"))
	}
	if res := do(t, ts, "OPTIONS", "/api/chat", "", map[string]string{"Origin": "http://127.0.0.1:8080"}); res.StatusCode != 204 {
		t.Errorf("preflight from a local page: %d, want 204", res.StatusCode)
	}
	if res := do(t, ts, "GET", "/api/tags", "", map[string]string{"Origin": "null"}); res.StatusCode != 403 {
		t.Errorf("Origin null: %d, want 403", res.StatusCode)
	}
	ts = newServer(t, Config{AllowedOrigins: []string{"https://app.example"}})
	if res := do(t, ts, "GET", "/api/tags", "", map[string]string{"Origin": "https://app.example"}); res.StatusCode != 200 {
		t.Errorf("allowed origin: %d, want 200", res.StatusCode)
	}
}

func TestManagementSwitches(t *testing.T) {
	ts := newServer(t, Config{ReadOnly: true})
	for _, c := range [][2]string{{"POST", "/api/pull"}, {"DELETE", "/api/delete"}, {"POST", "/api/create"}, {"POST", "/api/copy"}, {"POST", "/api/push"}} {
		if res := do(t, ts, c[0], c[1], `{"model":"qwen3-0.6b"}`, nil); res.StatusCode != 403 {
			t.Errorf("read-only %s %s: %d, want 403", c[0], c[1], res.StatusCode)
		}
	}
	ts = newServer(t, Config{NoPull: true})
	if res := do(t, ts, "POST", "/api/pull", `{"model":"qwen3-0.6b"}`, nil); res.StatusCode != 403 {
		t.Errorf("no-pull: %d, want 403", res.StatusCode)
	}
	if res := do(t, ts, "DELETE", "/api/delete", `{"model":"qwen3-0.6b"}`, nil); res.StatusCode != 404 {
		t.Errorf("no-pull, delete of a missing model: %d, want 404", res.StatusCode)
	}
}

func TestPaths(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "mine.gguf"), []byte("not really"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := New(Config{ModelsDir: dir})
	for _, name := range []string{"/etc/passwd", "../../etc/passwd.gguf", "~/x.gguf", `..\..\x.gguf`, filepath.Join(dir, "..", "x.gguf")} {
		if _, err := s.resolve(name); statusOf(err) != 403 {
			t.Errorf("resolve %q: %v, want refused", name, err)
		}
	}
	for _, name := range []string{"mine", filepath.Join(dir, "mine.gguf"), "qwen3:0.6b", "hf:a/b/c.gguf"} {
		if _, err := s.resolve(name); err != nil {
			t.Errorf("resolve %q: %v", name, err)
		}
	}
	if _, err := New(Config{ModelsDir: dir, AllowPaths: true}).resolve("/tmp/x.gguf"); err != nil {
		t.Errorf("AllowPaths: %v", err)
	}
	ts := newServer(t, Config{ModelsDir: dir})
	if res := do(t, ts, "POST", "/api/chat", `{"model":"/etc/passwd","messages":[{"role":"user","content":"hi"}]}`, nil); res.StatusCode != 403 {
		t.Errorf("chat with a path: %d, want 403", res.StatusCode)
	}
	if res := do(t, ts, "POST", "/api/show", `{"model":"/etc/hosts"}`, nil); res.StatusCode != 403 {
		t.Errorf("show a path: %d, want 403", res.StatusCode)
	}
	if res := do(t, ts, "GET", "/v1/models//etc/passwd", "", nil); res.StatusCode != 404 {
		t.Errorf("v1 model by path: %d, want 404", res.StatusCode)
	}
}

func TestAllowlist(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"qwen3-0.6b.gguf", "other.gguf"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ts := newServer(t, Config{ModelsDir: dir, Models: []string{"qwen3:0.6b"}})
	if res := do(t, ts, "POST", "/api/chat", `{"model":"other","messages":[{"role":"user","content":"hi"}]}`, nil); res.StatusCode != 403 {
		t.Errorf("chat with a model not allowed: %d, want 403", res.StatusCode)
	}
	if res := do(t, ts, "POST", "/api/pull", `{"model":"qwen3-4b"}`, nil); res.StatusCode != 403 {
		t.Errorf("pull a model not allowed: %d, want 403", res.StatusCode)
	}
	_, out := call(t, ts, "GET", "/api/tags", "")
	if ms := out["models"].([]any); len(ms) != 1 || ms[0].(map[string]any)["name"] != "qwen3-0.6b" {
		t.Errorf("tags: %v, want only qwen3-0.6b", ms)
	}
	_, out = call(t, ts, "GET", "/v1/models", "")
	if ms := out["data"].([]any); len(ms) != 1 {
		t.Errorf("v1/models: %v, want one", ms)
	}
}

func TestBodyLimit(t *testing.T) {
	ts := newServer(t, Config{MaxBodyBytes: 100})
	big := `{"model":"x","prompt":"` + strings.Repeat("a", 200) + `"}`
	if res := do(t, ts, "POST", "/api/generate", big, nil); res.StatusCode != 413 {
		t.Errorf("big body: %d, want 413", res.StatusCode)
	}
	if res := do(t, ts, "POST", "/v1/chat/completions", big, nil); res.StatusCode != 413 {
		t.Errorf("big body, OpenAI: %d, want 413", res.StatusCode)
	}
}

func TestPerClient(t *testing.T) {
	var c clientCounts
	if !c.enter("a", 2) || !c.enter("a", 2) {
		t.Fatal("first two should enter")
	}
	if c.enter("a", 2) {
		t.Error("third should wait")
	}
	if !c.enter("b", 2) {
		t.Error("another client should enter")
	}
	c.leave("a")
	if !c.enter("a", 2) {
		t.Error("after one leaves, a should enter")
	}
	c.leave("a")
	c.leave("a")
	c.leave("b")
	if len(c.n) != 0 {
		t.Errorf("counts left over: %v", c.n)
	}
}
