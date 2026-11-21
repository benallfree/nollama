package server

import (
	"bufio"
	"encoding/json"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

func tok(text string, start int) lpToken {
	return lpToken{bytes: []byte(text), start: start, end: start + len(text)}
}

// A token goes out with the piece that completes it; at the end, tokens that put any byte into the text go too.
func TestLPStream(t *testing.T) {
	s := &lpStream{q: []lpToken{tok("Hel", 0), tok("lo", 3), tok(" wor", 5), tok("ld", 9)}}
	if got := s.take("He"); len(got) != 0 {
		t.Errorf("He: %v", got)
	}
	if got := s.take("llo w"); len(got) != 2 || string(got[1].bytes) != "lo" {
		t.Errorf("llo w: %v", got)
	}
	// the text ends inside " wor" (say a stop string "orld" cut it): " wor" counts, "ld" doesn't
	if got := s.rest(7); len(got) != 1 || string(got[0].bytes) != " wor" {
		t.Errorf("rest: %v", got)
	}
}

func TestLogprobsValidation(t *testing.T) {
	ts := newTest(t, t.TempDir())
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"top_logprobs":3}`},
		{"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":21}`},
		{"/v1/completions", `{"model":"x","prompt":"hi","logprobs":6}`},
		{"/v1/completions", `{"model":"x","prompt":"hi","echo":true}`},
		{"/api/chat", `{"model":"x","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":-1}`},
	} {
		if code, out := call(t, ts, "POST", c.path, c.body); code != 400 {
			t.Errorf("%s %s: %d %v, want 400", c.path, c.body, code, out)
		}
	}
}

type entry struct {
	Token       string  `json:"token"`
	Logprob     float64 `json:"logprob"`
	RawBytes    []int   `json:"bytes"`
	TopLogprobs []entry `json:"top_logprobs"`
}

func joined(es []entry) string {
	var b []byte
	for _, e := range es {
		for _, c := range e.RawBytes {
			b = append(b, byte(c))
		}
	}
	return string(b)
}

// checkEntries: greedy decoding picks the likeliest token, so each entry is its own top alternative.
func checkEntries(t *testing.T, what string, es []entry, top int) {
	t.Helper()
	if len(es) == 0 {
		t.Fatalf("%s: no logprobs", what)
	}
	for i, e := range es {
		if e.Logprob > 0 || len(e.TopLogprobs) != top {
			t.Fatalf("%s %d: %+v", what, i, e)
		}
		if top > 0 && (e.TopLogprobs[0].Token != e.Token || math.Abs(e.TopLogprobs[0].Logprob-e.Logprob) > 1e-9) {
			t.Fatalf("%s %d: chosen %q %v, top %q %v", what, i, e.Token, e.Logprob, e.TopLogprobs[0].Token, e.TopLogprobs[0].Logprob)
		}
		for j := 1; j < len(e.TopLogprobs); j++ {
			if e.TopLogprobs[j].Logprob > e.TopLogprobs[j-1].Logprob {
				t.Fatalf("%s %d: top not sorted: %+v", what, i, e.TopLogprobs)
			}
		}
	}
}

func post(t *testing.T, url, body string) []string {
	t.Helper()
	res, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var lines []string
	sc := bufio.NewScanner(res.Body)
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		if l := strings.TrimPrefix(sc.Text(), "data: "); l != "" {
			lines = append(lines, l)
		}
	}
	if res.StatusCode != 200 {
		t.Fatalf("%s: %d %v", url, res.StatusCode, lines)
	}
	return lines
}

// TestLogprobs runs every API's logprobs on the tiny random model, or on a real one:
// GGUF=/path/to/qwen3-0.6b.gguf go test -run Logprobs ./server
func TestLogprobs(t *testing.T) {
	dir := t.TempDir()
	if p := os.Getenv("GGUF"); p != "" {
		if err := os.Symlink(p, filepath.Join(dir, "test.gguf")); err != nil {
			t.Fatal(err)
		}
	} else if err := tinymodel.Write(filepath.Join(dir, "test.gguf"), tinymodel.Config{}); err != nil {
		t.Fatal(err)
	}
	ts := newTest(t, dir)
	model := filepath.Join(dir, "test.gguf")
	msgs := `"messages":[{"role":"user","content":"/no_think Count from one to ten in words, separated by spaces."}]`

	// OpenAI chat, whole answer: the tokens spell the content
	var chat struct {
		Choices []struct {
			Message  struct{ Content string }
			Logprobs struct{ Content []entry }
		}
	}
	lines := post(t, ts.URL+"/v1/chat/completions", `{"model":"`+model+`","temperature":0,"max_tokens":30,"logprobs":true,"top_logprobs":3,`+msgs+`}`)
	if err := json.Unmarshal([]byte(lines[0]), &chat); err != nil {
		t.Fatal(err)
	}
	c := chat.Choices[0]
	checkEntries(t, "chat", c.Logprobs.Content, 3)
	if joined(c.Logprobs.Content) != c.Message.Content {
		t.Errorf("chat: tokens %q, content %q", joined(c.Logprobs.Content), c.Message.Content)
	}

	// the same, streamed with a stop string: each chunk's tokens, then the leftovers, cover the text exactly or run
	// past it only inside the token the stop string cut
	lines = post(t, ts.URL+"/v1/chat/completions", `{"model":"`+model+`","temperature":0,"max_tokens":30,"stream":true,"stop":["Five","five"],"logprobs":true,"top_logprobs":2,`+msgs+`}`)
	var text string
	var all []entry
	for _, l := range lines {
		if l == "[DONE]" {
			continue
		}
		var ch struct {
			Choices []struct {
				Delta    struct{ Content string }
				Logprobs *struct{ Content []entry }
			}
		}
		if err := json.Unmarshal([]byte(l), &ch); err != nil {
			t.Fatal(err)
		}
		if len(ch.Choices) == 0 {
			continue
		}
		text += ch.Choices[0].Delta.Content
		if ch.Choices[0].Logprobs == nil {
			t.Fatalf("chunk without logprobs: %s", l)
		}
		all = append(all, ch.Choices[0].Logprobs.Content...)
	}
	t.Logf("streamed %q", text)
	checkEntries(t, "stream", all, 2)
	if strings.Contains(strings.ToLower(text), "five") {
		t.Errorf("stream: stop string in %q", text)
	}
	if j := joined(all); !strings.HasPrefix(j, text) || (len(j) > len(text) && len(j)-len(all[len(all)-1].RawBytes) >= len(text)) {
		t.Errorf("stream: tokens %q, text %q", j, text)
	}

	// legacy completions
	var cmpl struct {
		Choices []struct {
			Text     string
			Logprobs legacyLogprobs
		}
	}
	lines = post(t, ts.URL+"/v1/completions", `{"model":"`+model+`","prompt":"one two three","temperature":0,"max_tokens":8,"logprobs":2}`)
	if err := json.Unmarshal([]byte(lines[0]), &cmpl); err != nil {
		t.Fatal(err)
	}
	lp := cmpl.Choices[0].Logprobs
	if len(lp.Tokens) == 0 || len(lp.Tokens) != len(lp.TokenLogprobs) || len(lp.Tokens) != len(lp.TextOffset) || strings.Join(lp.Tokens, "") != cmpl.Choices[0].Text ||
		lp.TextOffset[0] != 0 || len(lp.TopLogprobs[0]) == 0 {
		t.Errorf("completions: %q %+v", cmpl.Choices[0].Text, lp)
	}

	// Ollama chat, streamed
	lines = post(t, ts.URL+"/api/chat", `{"model":"`+model+`","options":{"temperature":0,"num_predict":20},"logprobs":true,"top_logprobs":1,`+msgs+`}`)
	text, all = "", nil
	for _, l := range lines {
		var ch struct {
			Message  struct{ Content string }
			Logprobs []entry
		}
		if err := json.Unmarshal([]byte(l), &ch); err != nil {
			t.Fatal(err)
		}
		text += ch.Message.Content
		all = append(all, ch.Logprobs...)
	}
	checkEntries(t, "ollama chat", all, 1)
	if joined(all) != text {
		t.Errorf("ollama chat: tokens %q, text %q", joined(all), text)
	}

	// Ollama generate, whole answer; no logprobs unless asked
	var gen struct {
		Response string
		Logprobs []entry
	}
	lines = post(t, ts.URL+"/api/generate", `{"model":"`+model+`","stream":false,"raw":true,"prompt":"one two three","options":{"temperature":0,"num_predict":8},"logprobs":true}`)
	if err := json.Unmarshal([]byte(lines[0]), &gen); err != nil {
		t.Fatal(err)
	}
	checkEntries(t, "ollama generate", gen.Logprobs, 0)
	if joined(gen.Logprobs) != gen.Response {
		t.Errorf("ollama generate: tokens %q, response %q", joined(gen.Logprobs), gen.Response)
	}
	lines = post(t, ts.URL+"/api/generate", `{"model":"`+model+`","stream":false,"raw":true,"prompt":"one two three","options":{"num_predict":2}}`)
	if strings.Contains(lines[0], "logprobs") {
		t.Errorf("logprobs not asked for: %s", lines[0])
	}

	// streamed tool calls hold text back until a call is complete: every generated token still comes out once,
	// in order, with a chunk that is sent
	tools := `"tools":[` + testWeather + `],"tool_choice":"required"`
	ask := `"messages":[{"role":"user","content":"/no_think What's the weather in Paris?"}]`
	lines = post(t, ts.URL+"/v1/chat/completions", `{"model":"`+model+`","temperature":0,"max_tokens":80,"stream":true,"stream_options":{"include_usage":true},"logprobs":true,`+tools+`,`+ask+`}`)
	all, calls, usage := nil, 0, 0
	for _, l := range lines {
		if l == "[DONE]" {
			continue
		}
		var ch struct {
			Choices []struct {
				Delta struct {
					ToolCalls []any `json:"tool_calls"`
				}
				Logprobs struct{ Content []entry }
			}
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
			}
		}
		if err := json.Unmarshal([]byte(l), &ch); err != nil {
			t.Fatal(err)
		}
		if ch.Usage != nil {
			usage = ch.Usage.CompletionTokens
		}
		for _, c := range ch.Choices {
			calls += len(c.Delta.ToolCalls)
			all = append(all, c.Logprobs.Content...)
		}
	}
	real := os.Getenv("GGUF") != "" // the tiny model's vocabulary can't write a tool call, so only count tokens
	if (real && calls == 0) || len(all) != usage {
		t.Errorf("tool call stream: %d calls, %d logprobs for %d tokens", calls, len(all), usage)
	}
	lines = post(t, ts.URL+"/api/chat", `{"model":"`+model+`","options":{"temperature":0,"num_predict":80},"logprobs":true,`+tools+`,`+ask+`}`)
	all, calls, usage = nil, 0, 0
	for _, l := range lines {
		var ch struct {
			Message struct {
				ToolCalls []any `json:"tool_calls"`
			}
			Logprobs  []entry
			EvalCount int `json:"eval_count"`
		}
		if err := json.Unmarshal([]byte(l), &ch); err != nil {
			t.Fatal(err)
		}
		calls += len(ch.Message.ToolCalls)
		all = append(all, ch.Logprobs...)
		usage = max(usage, ch.EvalCount)
	}
	if (real && calls == 0) || len(all) != usage {
		t.Errorf("ollama tool call stream: %d calls, %d logprobs for %d tokens", calls, len(all), usage)
	}
}
