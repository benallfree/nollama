package nollamatest_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/nollamatest"
	"github.com/benallfree/nollama/schema"
)

var ctx = context.Background()

func TestTinyDeterministic(t *testing.T) {
	m := nollamatest.Load(t, nollamatest.TinyConfig{})
	prompt := m.ChatPrompt([]engine.Message{{Role: "user", Content: "hello"}})
	a, err := m.Generate(ctx, prompt, engine.GenerateOptions{MaxTokens: 12}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Generate(ctx, prompt, engine.GenerateOptions{MaxTokens: 12}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text == "" || a.Text != b.Text || a.StopReason != "length" || a.OutputTokens != 12 {
		t.Fatalf("%+v\n%+v", a, b)
	}
	for _, r := range a.Text {
		if r < ' ' || r > '~' {
			t.Errorf("not printable ASCII: %q", a.Text)
		}
	}
	other := nollamatest.Load(t, nollamatest.TinyConfig{Seed: 7})
	c, _ := other.Generate(ctx, prompt, engine.GenerateOptions{MaxTokens: 12}, nil)
	if c.Text == a.Text {
		t.Errorf("seeds 0 and 7 wrote the same: %q", c.Text)
	}
}

func TestTinyConstraint(t *testing.T) {
	m := nollamatest.Load(t, nollamatest.TinyConfig{})
	prompt := m.ChatPrompt([]engine.Message{{Role: "user", Content: "I was charged twice."}})
	for sch, ok := range map[string]func(string) bool{
		`{"enum":["bug","billing","question"]}`: func(s string) bool { return s == `"bug"` || s == `"billing"` || s == `"question"` },
		`{"type":"boolean"}`:                    func(s string) bool { return s == "true" || s == "false" },
		`{"type":"array","items":{"enum":[1,2,3]},"minItems":2,"maxItems":2}`: func(s string) bool {
			var v []int
			return json.Unmarshal([]byte(s), &v) == nil && len(v) == 2
		},
	} {
		sc, err := schema.Compile([]byte(sch))
		if err != nil {
			t.Fatal(err)
		}
		res, err := m.Generate(ctx, prompt, engine.GenerateOptions{MaxTokens: 60, Constraint: sc.Fit}, nil)
		if err != nil || !ok(strings.TrimSpace(res.Text)) || res.StopReason != "stop" {
			t.Errorf("%s: %+v %v", sch, res, err)
		}
	}
}

func TestEndTurns(t *testing.T) {
	m := nollamatest.Load(t, nollamatest.TinyConfig{EndTurns: true})
	if _, err := m.Generate(ctx, "hi", engine.GenerateOptions{MaxTokens: 5}, nil); err != nil {
		t.Fatal(err)
	}
}

func TestScripted(t *testing.T) {
	m := nollamatest.Scripted(t, func(p nollamatest.Prompt) string {
		if strings.Contains(p.Last("user"), "charged") {
			return `{"label": "billing"}`
		}
		return "Bonjour, ça va?"
	})
	var pieces []string
	res, err := m.Generate(ctx, m.ChatPrompt([]engine.Message{{Role: "system", Content: "Be brief."}, {Role: "user", Content: "Say hi in French."}}),
		engine.GenerateOptions{}, func(s string) bool { pieces = append(pieces, s); return true })
	if err != nil || res.Text != "Bonjour, ça va?" || res.StopReason != "eos" || strings.Join(pieces, "") != res.Text || len(pieces) < 2 {
		t.Fatalf("%+v %v %q", res, err, pieces)
	}

	res, _ = m.Generate(ctx, "Say hi", engine.GenerateOptions{MaxTokens: 3}, nil)
	if res.StopReason != "length" || res.OutputTokens != 3 {
		t.Errorf("max tokens: %+v", res)
	}
	res, _ = m.Generate(ctx, "Say hi", engine.GenerateOptions{Stop: []string{","}}, nil)
	if res.Text != "Bonjour" || res.StopReason != "stop" {
		t.Errorf("stop: %+v", res)
	}

	label, _ := schema.Compile([]byte(`{"type":"object","properties":{"label":{"enum":["bug","billing"]}},"required":["label"]}`))
	charged := m.ChatPrompt([]engine.Message{{Role: "user", Content: "I was charged twice."}})
	res, err = m.Generate(ctx, charged, engine.GenerateOptions{Constraint: label.Fit}, nil)
	if err != nil || res.Text != `{"label": "billing"}` {
		t.Errorf("constrained: %+v %v", res, err)
	}
	strict, _ := schema.Compile([]byte(`{"type":"object","properties":{"label":{"enum":["bug","question"]}},"required":["label"]}`))
	if _, err := m.Generate(ctx, charged, engine.GenerateOptions{Constraint: strict.Fit}, nil); err == nil {
		t.Error("a scripted answer outside the schema: want an error")
	}
}

func TestParsePrompt(t *testing.T) {
	msgs := []engine.Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "q1\nmore"}, {Role: "assistant", Content: "a1"}, {Role: "user", Content: "q2"}}
	p := nollamatest.ParsePrompt(engine.ChatPrompt(msgs))
	if len(p.Messages) != 4 || p.Messages[1].Content != "q1\nmore" || p.Messages[2].Content != "a1" || p.Last("user") != "q2" {
		t.Errorf("%+v", p.Messages)
	}
	if p := nollamatest.ParsePrompt("plain"); p.Messages != nil || p.Raw != "plain" {
		t.Errorf("%+v", p)
	}
}
