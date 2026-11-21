package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNormalize(t *testing.T) {
	for in, want := range map[string]string{
		"qwen3:1.7b":           "qwen3-1.7b",
		"qwen3-1.7b":           "qwen3-1.7b",
		"qwen3-1.7b:latest":    "qwen3-1.7b",
		"qwen3-embedding:0.6b": "qwen3-embedding-0.6b",
		" qwen3:0.6b ":         "qwen3-0.6b",
		"hf:a/b/c:d.gguf":      "hf:a/b/c:d.gguf",
		"/tmp/x:y.gguf":        "/tmp/x:y.gguf",
		"some-local-model":     "some-local-model",
		"some:local:model":     "some-local:model",
	} {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestResolve(t *testing.T) {
	dir := t.TempDir()
	src, err := Resolve("qwen3:1.7b", dir)
	if err != nil {
		t.Fatal(err)
	}
	if src.Name != "qwen3-1.7b" || src.Path != filepath.Join(dir, "qwen3-1.7b.gguf") || src.SHA256 == "" || src.URL == "" {
		t.Errorf("catalog model: %+v", src)
	}
	src, err = Resolve("hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", dir)
	if err != nil {
		t.Fatal(err)
	}
	if src.URL != "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf" || src.SHA256 != "" {
		t.Errorf("hf model: %+v", src)
	}
	if _, err := Resolve("hf:only/two", dir); err == nil {
		t.Error("hf:only/two: want an error")
	}
	src, err = Resolve("/models/mine.gguf", dir)
	if err != nil || src.Name != "mine" || src.URL != "" {
		t.Errorf("path: %+v %v", src, err)
	}
	if _, err := Resolve("llama9000", dir); err == nil {
		t.Error("unknown name: want an error")
	}
}

func TestList(t *testing.T) {
	dir := t.TempDir()
	if got, err := List(filepath.Join(dir, "missing")); err != nil || len(got) != 0 {
		t.Fatalf("missing dir: %v %v", got, err)
	}
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(os.WriteFile(filepath.Join(dir, "qwen3-0.6b.gguf"), []byte("x"), 0o644))
	must(os.WriteFile(filepath.Join(dir, "qwen3-0.6b.gguf.123.part"), []byte("x"), 0o644))
	must(os.MkdirAll(filepath.Join(dir, "hf"), 0o755))
	must(os.WriteFile(filepath.Join(dir, "hf", "a_b_c.gguf"), []byte("xy"), 0o644))
	got, err := List(dir)
	must(err)
	if len(got) != 2 {
		t.Fatalf("got %+v", got)
	}
	if got[1].Name != "qwen3-0.6b" || got[1].Size != 1 {
		t.Errorf("catalog file: %+v", got[1])
	}
}

func TestResolveSafety(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "mine.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	src, err := Resolve("mine", dir)
	if err != nil || src.Path != filepath.Join(dir, "mine.gguf") || src.URL != "" {
		t.Errorf("a file in the models folder by name: %+v %v", src, err)
	}
	for _, name := range []string{"missing", "..", ".hidden"} {
		if _, err := Resolve(name, dir); err == nil {
			t.Errorf("%q: want an error", name)
		}
	}
	// hf: files stay in dir/hf whatever the spec holds: no separators, no "..", no ':' (a Windows drive or NTFS stream)
	for _, spec := range []string{`hf:a/b/..\..\..\x.gguf`, "hf:a/b/c:evil.gguf", "hf:../../x/y.gguf"} {
		src, err := Resolve(spec, dir)
		if err != nil {
			continue
		}
		if filepath.Dir(src.Path) != filepath.Join(dir, "hf") || strings.ContainsAny(filepath.Base(src.Path), `:\/`) {
			t.Errorf("%s: path %s escapes %s/hf", spec, src.Path, dir)
		}
	}
	if !IsPath(`C:\models\x.gguf`) || !IsPath("~/x") || !IsPath("a/b") || IsPath("qwen3-0.6b") || IsPath("hf-model") {
		t.Error("IsPath")
	}
}
