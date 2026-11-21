package engine

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

// testModelPath is $GGUF, or, when that's unset, a tiny Qwen3 model with random weights: the engine's plumbing (prompt
// cache, threads, concurrency) runs in every `go test`, and GGUF=<a real Qwen3 .gguf> checks it on a real model too.
func testModelPath(t testing.TB) string {
	t.Helper()
	if p := os.Getenv("GGUF"); p != "" {
		return p
	}
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := tinymodel.Write(p, tinymodel.Config{}); err != nil {
		t.Fatal(err)
	}
	return p
}
