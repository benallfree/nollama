package compose

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/benallfree/nollama/engine"
)

// TestCompareSpeed times the rebuilt Classify and Score against the engine's, interleaved round by round so machine
// load hits both alike. COMPOSE_BENCH=1 (and GGUF=<model> for real numbers) to run it.
func TestCompareSpeed(t *testing.T) {
	if os.Getenv("COMPOSE_BENCH") == "" {
		t.Skip("COMPOSE_BENCH=1 to run")
	}
	ctx := context.Background()
	for _, det := range []bool{false, true} {
		m := load(t, engine.LoadOptions{Deterministic: det})
		ids := m.Tokenize(prompt)
		text := m.Tokenize(prompt + "The dog is lazy, and the fox is quick. " + prompt)
		ls := labels()
		cases := []struct {
			name string
			f    func() error
		}{
			{"classify engine", func() error { _, err := m.ClassifyTokens(ctx, ids, ls, engine.ClassifyOptions{}); return err }},
			{"classify compose", func() error { _, err := Classify(ctx, m, ids, ls, engine.ClassifyOptions{}); return err }},
			{"classify rewind", func() error { _, err := ClassifyRewind(ctx, m, ids, ls); return err }},
			{"score engine", func() error { _, err := m.Score(ctx, text, engine.ScoreOptions{}); return err }},
			{"score compose", func() error { _, err := Score(ctx, m, text, engine.ScoreOptions{}); return err }},
		}
		times := make([][]time.Duration, len(cases))
		for round := range 12 {
			for i, c := range cases {
				start := time.Now()
				if err := c.f(); err != nil {
					t.Fatal(err)
				}
				if round > 1 { // warm-up
					times[i] = append(times[i], time.Since(start))
				}
			}
		}
		t.Logf("GPU %q, deterministic=%v, prompt %d tokens, %d labels, score text %d tokens", m.Info().GPU, det, len(ids), len(ls), len(text))
		for i, c := range cases {
			slices.Sort(times[i])
			t.Logf("  %-17s min %8.2f ms  median %8.2f ms", c.name, ms(times[i][0]), ms(times[i][len(times[i])/2]))
		}
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
