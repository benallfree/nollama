package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/engine/schema"
)

// GGUF=<file> go test -run XXX -bench Classify ./engine
//
// Labeling a ticket two ways: Classify scores the labels on one read of the prompt; the usual way generates a JSON
// answer constrained to a schema enum and parses it. Neither uses the prompt cache, so both read the whole prompt.
// ns/op is both together; classify-ms and generate-ms split it.
func BenchmarkClassify(b *testing.B) {
	p := os.Getenv("GGUF")
	if p == "" {
		b.Skip("GGUF not set")
	}
	m, err := engine.Load(p, engine.LoadOptions{Threads: 8})
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	labels := []string{"bug", "feature request", "billing question", "account access", "other"}
	prompt := engine.ChatPrompt([]engine.Message{
		{Role: "system", Content: "You label customer support tickets for a software company. A bug is something that is broken or crashes. " +
			"A feature request asks for something new. Answer with exactly one label: bug, feature request, billing question, account access or other."},
		{Role: "user", Content: "Bug report: the app crashes and closes every time I press the Save button. It is broken."},
	})
	ctx := context.Background()
	enum, _ := json.Marshal(labels)
	sc, err := schema.Compile(json.RawMessage(`{"type":"object","properties":{"label":{"enum":` + string(enum) + `}},"required":["label"]}`))
	if err != nil {
		b.Fatal(err)
	}
	gopt := engine.GenerateOptions{MaxTokens: 32, Temperature: 0, Constraint: sc.Fit}

	// the two ways alternate, so a busy machine slows both alike
	var tc, tg time.Duration
	var tokens int
	for b.Loop() {
		t0 := time.Now()
		res, err := m.Classify(ctx, prompt, labels, engine.ClassifyOptions{Complete: true})
		tc += time.Since(t0)
		if err != nil {
			b.Fatal(err)
		}
		t0 = time.Now()
		gen, err := m.Generate(ctx, prompt, gopt, nil)
		tg += time.Since(t0)
		var out struct{ Label string }
		if err != nil || json.Unmarshal([]byte(gen.Text), &out) != nil {
			b.Fatal(gen.Text, err)
		}
		if out.Label != labels[res.Best()] {
			b.Logf("they disagree: classify %q %v, generate %q", labels[res.Best()], res.Probs, out.Label)
		}
		tokens = gen.OutputTokens
	}
	n := float64(b.N)
	b.ReportMetric(float64(tc.Milliseconds())/n, "classify-ms")
	b.ReportMetric(float64(tg.Milliseconds())/n, "generate-ms")
	b.ReportMetric(float64(tg)/float64(tc), "speedup")
	b.ReportMetric(float64(tokens), "gen-tokens")
}
