package engine

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestContextError(t *testing.T) {
	err := checkContext(100, 50, 120)
	var ce *ContextError
	if !errors.Is(err, ErrContextExceeded) || !errors.As(err, &ce) {
		t.Fatalf("checkContext: %v", err)
	}
	if ce.PromptTokens != 100 || ce.MaxTokens != 50 || ce.Limit != 120 {
		t.Errorf("counts: %+v", ce)
	}
	for _, want := range []string{"100", "50", "150", "120"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("message %q lacks %s", err, want)
		}
	}
	if err := checkContext(100, 20, 120); err != nil {
		t.Errorf("exact fit: %v", err)
	}
	wrapped := errors.Join(errors.New("x"), &ContextError{PromptTokens: 9, Limit: 8})
	if !errors.Is(wrapped, ErrContextExceeded) {
		t.Error("wrapped: not ErrContextExceeded")
	}
}

// words counts a conversation as its words plus 2 per message: a stand-in for the template and tokenizer.
func words(ms []Message) int {
	n := 0
	for _, m := range ms {
		n += 2 + len(strings.Fields(m.Content))
	}
	return n
}

func TestTrimMessages(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "be brief"},           // 4
		{Role: "user", Content: "one two three"},        // 5
		{Role: "assistant", Content: "four five six"},   // 5
		{Role: "user", Content: "seven eight nine ten"}, // 6
		{Role: "assistant", Content: "eleven"},          // 3
		{Role: "user", Content: "last question here"},   // 5
	} // 28 in all
	got, tr, err := TrimMessages(msgs, 28, words)
	if err != nil || len(got) != 6 || tr.Messages != 0 {
		t.Errorf("fits: %v %+v %v", got, tr, err)
	}
	// 20 fits after dropping the first exchange (28-10=18)
	got, tr, err = TrimMessages(msgs, 20, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 || got[0].Role != "system" || got[1].Content != "seven eight nine ten" || got[3].Content != "last question here" {
		t.Errorf("trimmed: %v", got)
	}
	if tr.Messages != 2 || tr.Tokens != 10 || tr.PromptTokens != 28 {
		t.Errorf("report: %+v", tr)
	}
	// 15: dropping 3 would leave assistant first, so it goes too
	got, tr, err = TrimMessages(msgs, 15, words)
	if err != nil || len(got) != 2 || got[1].Content != "last question here" || tr.Messages != 4 {
		t.Errorf("no assistant first: %v %+v %v", got, tr, err)
	}
	// system + last = 9: 8 can't fit
	_, _, err = TrimMessages(msgs, 8, words)
	var ce *ContextError
	if !errors.As(err, &ce) || ce.PromptTokens != 9 || ce.Limit != 8 {
		t.Errorf("too small: %v", err)
	}
	// one message that doesn't fit
	if _, _, err := TrimMessages([]Message{{Role: "user", Content: "a b c"}}, 2, words); !errors.Is(err, ErrContextExceeded) {
		t.Errorf("single: %v", err)
	}
}

// GGUF=/path/to/qwen3-0.6b.gguf go test -run TestContextReal ./engine
func TestContextReal(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m, err := Load(p, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := m.ChatPrompt([]Message{{Role: "user", Content: "Count from one to fifty in words."}})
	n := m.CountTokens(prompt)
	limit := m.Info().ContextLength
	_, err = m.Generate(context.Background(), prompt, GenerateOptions{MaxTokens: limit}, nil)
	var ce *ContextError
	if !errors.As(err, &ce) || ce.PromptTokens != n || ce.MaxTokens != limit || ce.Limit != limit {
		t.Errorf("too long: %v", err)
	}
	res, err := m.Generate(context.Background(), prompt, GenerateOptions{MaxTokens: 3}, nil)
	if err != nil || res.StopReason != "length" || res.OutputTokens != 3 || res.PromptTokens != n {
		t.Errorf("max tokens: %+v %v", res, err)
	}
	// TrimMessages with the real template and tokenizer
	var long []Message
	for range 20 {
		long = append(long, Message{Role: "user", Content: strings.Repeat("lorem ipsum ", 20)}, Message{Role: "assistant", Content: "ok"})
	}
	long = append(long, Message{Role: "user", Content: "What is two plus two?"})
	kept, tr, err := m.TrimMessages(long, 200)
	if err != nil || tr.Messages == 0 || m.CountTokens(m.ChatPrompt(kept)) > 200 || kept[len(kept)-1].Content != "What is two plus two?" {
		t.Errorf("trim: %d kept, %+v %v", len(kept), tr, err)
	}
}
