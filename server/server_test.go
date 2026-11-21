package server

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/internal/tinymodel"
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
	if code, _ := call(t, ts, "POST", "/api/generate", `{"model":"x","prompt":"hi","format":"yaml"}`); code != 400 && code != 404 {
		t.Errorf("bad format: %d", code)
	}
	if code, _ := call(t, ts, "POST", "/api/create", `{}`); code != 501 {
		t.Errorf("create: %d", code)
	}
	if code, _ := call(t, ts, "POST", "/api/embed", `{"model":"qwen3-embedding:0.6b","input":["a","b"]}`); code != 404 {
		t.Errorf("embed, missing model: %d", code)
	}
	if code, _ := call(t, ts, "POST", "/v1/embeddings", `{"model":"x","input":{"text":"a"}}`); code != 400 {
		t.Errorf("v1 embeddings, bad input: %d", code)
	}
	if code, _ := call(t, ts, "POST", "/v1/embeddings", `{"model":"x","input":"a","encoding_format":"int8"}`); code != 400 {
		t.Errorf("v1 embeddings, bad encoding: %d", code)
	}
}

func TestEmbedInput(t *testing.T) {
	for in, want := range map[string]string{`"a"`: "[a] []", `["a","b"]`: "[a b] []", `[1,2]`: "[] [[1 2]]", `[[1],[2,3]]`: "[] [[1] [2 3]]"} {
		var e embedInput
		if err := json.Unmarshal([]byte(in), &e); err != nil {
			t.Errorf("%s: %v", in, err)
			continue
		}
		if got := fmt.Sprint(e.texts, " ", e.tokens); got != want {
			t.Errorf("%s: %s, want %s", in, got, want)
		}
	}
}

// The embedding routes on the tiny random-weight model, in every test run.
func TestEmbedTiny(t *testing.T) {
	dir := t.TempDir()
	if err := tinymodel.Write(filepath.Join(dir, "tiny.gguf"), tinymodel.Config{}); err != nil {
		t.Fatal(err)
	}
	ts := newTest(t, dir)
	code, out := call(t, ts, "POST", "/api/embed", `{"model":"tiny","input":["one","two","three"]}`)
	if code != 200 || len(out["embeddings"].([]any)) != 3 {
		t.Fatalf("embed: %d %v", code, out)
	}
	code, out = call(t, ts, "POST", "/v1/embeddings", `{"model":"tiny","input":[[1,2,3],[4,5]],"dimensions":16}`)
	if code != 200 || len(out["data"].([]any)) != 2 || len(out["data"].([]any)[1].(map[string]any)["embedding"].([]any)) != 16 {
		t.Fatalf("v1 embeddings, token ids: %d %v", code, out)
	}
	if code, _ := call(t, ts, "POST", "/v1/embeddings", `{"model":"tiny","input":[[99999999]]}`); code != 400 {
		t.Errorf("token outside the vocabulary: %d", code)
	}
	if code, out := call(t, ts, "POST", "/api/embeddings", `{"model":"tiny","prompt":"one"}`); code != 200 || len(out["embedding"].([]any)) == 0 {
		t.Errorf("embeddings: %d %v", code, out)
	}
	if code, out := call(t, ts, "POST", "/api/embed", `{"model":"tiny","input":"`+strings.Repeat("word ", 50)+`","options":{"num_ctx":8}}`); code != 200 || out["prompt_eval_count"].(float64) != 8 {
		t.Errorf("truncated: %d %v", code, out)
	}
}

// TestEmbed runs the three embedding routes against a real model:
// EMBED_GGUF=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf go test ./server
func TestEmbed(t *testing.T) {
	p := os.Getenv("EMBED_GGUF")
	if p == "" {
		t.Skip("EMBED_GGUF not set")
	}
	dir := t.TempDir()
	if err := os.Symlink(p, filepath.Join(dir, "emb.gguf")); err != nil {
		t.Fatal(err)
	}
	ts := newTest(t, dir)
	model := filepath.Join(dir, "emb.gguf")
	vec := func(v any) []float64 {
		var out []float64
		for _, x := range v.([]any) {
			out = append(out, x.(float64))
		}
		return out
	}
	dot := func(a, b []float64) (s float64) {
		for i := range a {
			s += a[i] * b[i]
		}
		return s
	}

	code, out := call(t, ts, "POST", "/api/embed", `{"model":"`+model+`","input":["The cat sat on the mat.","A cat was sitting on a rug.","Quarterly revenue grew by ten percent."]}`)
	if code != 200 {
		t.Fatalf("embed: %d %v", code, out)
	}
	es := out["embeddings"].([]any)
	if len(es) != 3 || out["prompt_eval_count"].(float64) < 3 {
		t.Fatalf("embed: %v embeddings, prompt_eval_count %v", len(es), out["prompt_eval_count"])
	}
	a, b, c := vec(es[0]), vec(es[1]), vec(es[2])
	if len(a) != 1024 || math.Abs(dot(a, a)-1) > 1e-4 {
		t.Errorf("embed: %d values, |v|² %.5f", len(a), dot(a, a))
	}
	if dot(a, b) <= dot(a, c) {
		t.Errorf("similar %.3f <= unrelated %.3f", dot(a, b), dot(a, c))
	}

	if code, out := call(t, ts, "POST", "/api/show", `{"model":"`+model+`"}`); code != 200 || fmt.Sprint(out["capabilities"]) != "[embedding]" {
		t.Errorf("show: %d %v", code, out["capabilities"])
	}

	code, out = call(t, ts, "POST", "/api/embeddings", `{"model":"`+model+`","prompt":"The cat sat on the mat."}`)
	if code != 200 || dot(vec(out["embedding"]), a) < 0.9999 {
		t.Errorf("embeddings: %d", code)
	}

	code, out = call(t, ts, "POST", "/v1/embeddings", `{"model":"`+model+`","input":["The cat sat on the mat."],"dimensions":64}`)
	if code != 200 {
		t.Fatalf("v1 embeddings: %d %v", code, out)
	}
	d := out["data"].([]any)[0].(map[string]any)
	if v := vec(d["embedding"]); len(v) != 64 || math.Abs(dot(v, v)-1) > 1e-4 || d["index"].(float64) != 0 {
		t.Errorf("v1 embeddings, 64 dimensions: %d values", len(v))
	}
	if out["usage"].(map[string]any)["prompt_tokens"].(float64) < 2 {
		t.Errorf("usage: %v", out["usage"])
	}

	code, out = call(t, ts, "POST", "/v1/embeddings", `{"model":"`+model+`","input":"The cat sat on the mat.","encoding_format":"base64"}`)
	if code != 200 {
		t.Fatalf("v1 embeddings base64: %d %v", code, out)
	}
	raw, err := base64.StdEncoding.DecodeString(out["data"].([]any)[0].(map[string]any)["embedding"].(string))
	if err != nil || len(raw) != 4*1024 {
		t.Fatalf("base64: %d bytes, %v", len(raw), err)
	}
	b64 := make([]float64, 1024)
	for i := range b64 {
		b64[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
	}
	if dot(b64, a) < 0.9999 {
		t.Errorf("base64 vector differs: %.5f", dot(b64, a))
	}

	// too long: /api/embed truncates by default, /v1/embeddings refuses
	long := strings.Repeat("word ", 300)
	if code, out := call(t, ts, "POST", "/api/embed", `{"model":"`+model+`","input":"`+long+`","options":{"num_ctx":64}}`); code != 200 || out["prompt_eval_count"].(float64) != 64 {
		t.Errorf("truncated embed: %d %v", code, out["prompt_eval_count"])
	}
	if code, _ := call(t, ts, "POST", "/api/embed", `{"model":"`+model+`","input":"`+long+`","truncate":false,"options":{"num_ctx":64}}`); code != 400 {
		t.Errorf("embed, truncate false: %d", code)
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

func TestToolsRequests(t *testing.T) {
	ts := newTest(t, t.TempDir())
	weather := `{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}`
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[{}]}`},
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[` + weather + `],"tool_choice":"sometimes"}`},
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[` + weather + `],"tool_choice":{"type":"function","function":{"name":"nope"}}}`},
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[` + weather + `],"response_format":{"type":"json_object"}}`},
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"a","parameters":"object"}}]}`},
		{"/v1/completions", `{"model":"x","prompt":"hi","tools":[` + weather + `]}`},
		{"/api/chat", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{}}]}`},
		{"/api/chat", `{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[` + weather + `],"format":"json"}`},
	} {
		if code, out := call(t, ts, "POST", c.path, c.body); code != 400 {
			t.Errorf("%s %s: %d %v, want 400", c.path, c.body, code, out)
		}
	}
	// well-formed tools get as far as loading the model
	if code, _ := call(t, ts, "POST", "/api/chat", `{"model":"qwen3:1.7b","messages":[{"role":"user","content":"hi"}],"tools":[`+weather+`]}`); code != 404 {
		t.Errorf("good tools: %d, want 404 (no model)", code)
	}
}

func TestToolStream(t *testing.T) {
	var ts toolStream
	var content strings.Builder
	var names []string
	answer := "Let me check.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>"
	for i := 0; i < len(answer); i += 3 { // pieces that cut the tags apart
		c, calls := ts.feed(answer[i:min(i+3, len(answer))])
		if strings.Contains(c, "<") {
			t.Errorf("passed on part of a tag: %q", c)
		}
		content.WriteString(c)
		for _, call := range calls {
			names = append(names, call.Name)
		}
	}
	content.WriteString(ts.rest())
	if content.String() != "Let me check." || strings.Join(names, ",") != "a,b" {
		t.Errorf("got %q %v", content.String(), names)
	}
	var plain toolStream
	got, _ := plain.feed("1 < 2 and ")
	got2, _ := plain.feed("<b> ")
	if got+got2+plain.rest() != "1 < 2 and <b> " {
		t.Errorf("plain text: %q %q", got, got2)
	}
}

// A model that doesn't fit the memory budget answers 503 with the reason, in both APIs; a shared budget is left open
// by Close. GGUF=/path/to/qwen3-0.6b.gguf go test ./server
func TestMemoryBudget(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	dir := t.TempDir()
	if err := os.Symlink(p, filepath.Join(dir, "test.gguf")); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "test.gguf")
	shared := engine.NewResources(engine.ResourceOptions{Threads: 2, Memory: 1 << 20})
	defer shared.Close()
	s := New(Config{ModelsDir: dir, Resources: shared})
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	code, out := call(t, ts, "POST", "/api/chat", `{"model":"`+model+`","stream":false,"messages":[{"role":"user","content":"Hi"}]}`)
	if code != http.StatusServiceUnavailable || !strings.Contains(fmt.Sprint(out["error"]), "memory limit") {
		t.Errorf("ollama: %d %v", code, out)
	}
	code, out = call(t, ts, "POST", "/v1/chat/completions", `{"model":"`+model+`","messages":[{"role":"user","content":"Hi"}]}`)
	if code != http.StatusServiceUnavailable || !strings.Contains(fmt.Sprint(out["error"]), "memory limit") {
		t.Errorf("openai: %d %v", code, out)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if s.Resources() != shared || shared.Usage().MemoryUsed != 0 {
		t.Errorf("shared resources: %+v", shared.Usage())
	}
	// still open: a model loads with it
	if _, err := engine.Load(model, engine.LoadOptions{Resources: shared}); !errors.Is(err, engine.ErrMemoryLimit) {
		t.Errorf("after Close: want the budget still in force, got %v", err)
	}
}

// /api/classify on the tiny random-weight model, in every test run, and against a real model when
// GGUF=/path/to/qwen3-0.6b.gguf is set.
func TestClassify(t *testing.T) {
	dir := t.TempDir()
	if err := tinymodel.Write(filepath.Join(dir, "tiny.gguf"), tinymodel.Config{}); err != nil {
		t.Fatal(err)
	}
	ts := newTest(t, dir)
	labels := func(out map[string]any) (sum float64, names []string) {
		ls, _ := out["labels"].([]any)
		for _, l := range ls {
			sum += l.(map[string]any)["probability"].(float64)
			names = append(names, l.(map[string]any)["label"].(string))
		}
		return sum, names
	}
	code, out := call(t, ts, "POST", "/api/classify", `{"model":"tiny","prompt":"the fox","labels":["dog","fox jumps","lazy dog"],"complete":true}`)
	if sum, names := labels(out); code != 200 || sum < 0.999 || sum > 1.001 || strings.Join(names, ",") != "dog,fox jumps,lazy dog" {
		t.Fatalf("tiny: %d %v", code, out)
	}
	for _, body := range []string{
		`{"model":"tiny","prompt":"hi","labels":[]}`,
		`{"model":"tiny","prompt":"hi","labels":["a",""]}`,
		`{"model":"tiny","labels":["a"]}`,
		`{"model":"tiny","prompt":"` + strings.Repeat("word ", 50) + `","labels":["a"],"options":{"num_ctx":8}}`,
	} {
		if code, out := call(t, ts, "POST", "/api/classify", body); code != 400 {
			t.Errorf("%s: %d %v, want 400", body, code, out)
		}
	}

	p := os.Getenv("GGUF")
	if p == "" {
		return
	}
	if err := os.Symlink(p, filepath.Join(dir, "test.gguf")); err != nil {
		t.Fatal(err)
	}
	code, out = call(t, ts, "POST", "/api/classify", `{"model":"test","complete":true,
		"system":"You label customer support tickets. A bug is something that is broken or crashes. Answer with one label: bug, feature request or billing question.",
		"prompt":"Bug report: the app crashes and closes every time I press the Save button. It is broken.",
		"labels":["bug","feature request","billing question"]}`)
	if sum, _ := labels(out); code != 200 || out["label"] != "bug" || sum < 0.999 || sum > 1.001 {
		t.Errorf("classify: %d %v", code, out)
	}
}
