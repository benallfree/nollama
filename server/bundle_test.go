package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/benallfree/nollama/catalog"
	"github.com/benallfree/nollama/internal/tinymodel"
)

// TestBundledModels lists a model bundled in the executable next to the downloads, and won't delete it.
func TestBundledModels(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "prog")
	if err := os.WriteFile(exe, []byte("a program"), 0o755); err != nil {
		t.Fatal(err)
	}
	bundled := filepath.Join(dir, "bundled")
	if _, err := catalog.Bundle(bundled, exe, []catalog.Source{{Name: "built-in", Data: []byte("GGUF not really")}}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.UseBundle(bundled); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { catalog.UseBundle("") })
	mdir := filepath.Join(dir, "models")
	os.MkdirAll(mdir, 0o755)
	os.WriteFile(filepath.Join(mdir, "qwen3-0.6b.gguf"), []byte("x"), 0o644)
	ts := newTest(t, mdir)

	code, out := call(t, ts, "GET", "/api/tags", "")
	tags, _ := out["models"].([]any)
	if code != 200 || len(tags) != 2 {
		t.Fatalf("tags: %d %v", code, out)
	}
	first := tags[0].(map[string]any)
	if first["name"] != "built-in" || first["digest"] == "" || first["size"].(float64) != float64(len("GGUF not really")) {
		t.Errorf("bundled tag: %v", first)
	}
	if code, out := call(t, ts, "GET", "/v1/models", ""); code != 200 || len(out["data"].([]any)) != 2 {
		t.Errorf("/v1/models: %d %v", code, out)
	}
	if code, _ := call(t, ts, "GET", "/v1/models/built-in", ""); code != 200 {
		t.Errorf("/v1/models/built-in: %d", code)
	}
	if code, out := call(t, ts, "DELETE", "/api/delete", `{"model":"built-in"}`); code != 400 {
		t.Errorf("delete a bundled model: %d %v", code, out)
	}
	if _, err := os.Stat(bundled); err != nil {
		t.Error("the bundle is gone")
	}
}

// TestBundledChat answers from a real (tiny, random-weight) model bundled in the executable.
func TestBundledChat(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "prog")
	if err := os.WriteFile(exe, []byte("a program"), 0o755); err != nil {
		t.Fatal(err)
	}
	bundled := filepath.Join(dir, "bundled")
	gguf := tinymodel.GGUF(tinymodel.Config{})
	if _, err := catalog.Bundle(bundled, exe, []catalog.Source{{Name: "tiny", Data: gguf}, {Name: "tiny2", Data: gguf}}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.UseBundle(bundled); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { catalog.UseBundle("") })
	ts := newTest(t, filepath.Join(dir, "models"))
	for _, name := range []string{"tiny", "tiny2", "tiny"} { // two bundled models share a path: each load must swap
		code, out := call(t, ts, "POST", "/api/generate", `{"model":"`+name+`","prompt":"hi","stream":false,"options":{"num_predict":3}}`)
		if code != 200 || out["done"] != true || out["model"] != name {
			t.Fatalf("%s: %d %v", name, code, out)
		}
	}
	code, out := call(t, ts, "GET", "/api/ps", "")
	if ms, _ := out["models"].([]any); code != 200 || len(ms) != 1 || ms[0].(map[string]any)["name"] != "tiny" {
		t.Errorf("ps: %d %v", code, out)
	}
}
