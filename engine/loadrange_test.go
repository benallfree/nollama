package engine

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/benallfree/nollama/catalog"
)

// tinyGGUF is a valid GGUF with one key and no tensors: enough to test finding and mapping one.
func tinyGGUF() []byte {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("GGUF")
	w(uint32(3)) // version
	w(uint64(0)) // tensors
	w(uint64(1)) // keys
	key, val := "general.architecture", "qwen3"
	w(uint64(len(key)))
	b.WriteString(key)
	w(uint32(8)) // string
	w(uint64(len(val)))
	b.WriteString(val)
	return b.Bytes()
}

// TestOpenGGUFRange finds a GGUF inside a bigger file at an offset that isn't page-aligned, by mmap where the engine
// mmaps and by the read-into-memory fallback everywhere.
func TestOpenGGUFRange(t *testing.T) {
	g := tinyGGUF()
	prefix := bytes.Repeat([]byte{0xab}, 70001)
	path := filepath.Join(t.TempDir(), "prog")
	if err := os.WriteFile(path, slices.Concat(prefix, g, []byte("trailing bytes")), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := openGGUFRange(path, int64(len(prefix)), int64(len(g)))
	if err != nil {
		t.Fatal(err)
	}
	if f.getString("general.architecture") != "qwen3" || f.size != int64(len(g)) || !bytes.Equal(f.data, g) {
		t.Errorf("got %v, size %d", f.KV, f.size)
	}
	if err := f.close(); err != nil {
		t.Error(err)
	}
	if _, err := openGGUFRange(path, int64(len(prefix))+1, int64(len(g))); err == nil {
		t.Error("an offset off by one: want bad magic")
	}
	if _, err := openGGUFRange(path, int64(len(prefix)), 1<<30); err == nil {
		t.Error("a range past the end: want an error")
	}

	file, _ := os.Open(path)
	defer file.Close()
	b, err := readRange(file, int64(len(prefix)), int64(len(g)))
	if err != nil || !bytes.Equal(b, g) {
		t.Errorf("readRange: %v", err)
	}
	if _, err := readRange(file, int64(len(prefix)), 1<<20); err == nil {
		t.Error("readRange past the end: want an error")
	}
}

// TestLoadBundled loads a real model three ways (its file, bundled at the end of another file, and from memory) and
// checks they answer alike: GGUF=/path/to/qwen3-0.6b.gguf go test ./engine -run TestLoadBundled
func TestLoadBundled(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	dir := t.TempDir()
	exe := filepath.Join(dir, "prog")
	if err := os.WriteFile(exe, bytes.Repeat([]byte("x"), 12345), 0o755); err != nil {
		t.Fatal(err)
	}
	bundled := filepath.Join(dir, "bundled")
	if _, err := catalog.Bundle(bundled, exe, []catalog.Source{{Name: "m", Path: p}}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.UseBundle(bundled); err != nil {
		t.Fatal(err)
	}
	defer catalog.UseBundle("")
	src, err := catalog.Resolve("m", dir)
	if err != nil || !src.Inside {
		t.Fatalf("%+v %v", src, err)
	}
	f, _ := os.Open(p)
	data, _ := io.ReadAll(f)
	f.Close()

	prompt := ChatPrompt([]Message{{Role: "user", Content: "Name three primary colors, comma separated."}})
	answer := func(m *Model, err error) []int {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		res, err := m.Generate(context.Background(), prompt, GenerateOptions{MaxTokens: 12, IgnoreEOS: true}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return res.Tokens
	}
	want := answer(Load(p, LoadOptions{Threads: 4}))
	// by path, the way code written for downloaded files loads
	if got := answer(Load(src.Path, LoadOptions{Threads: 4})); !slices.Equal(got, want) {
		t.Errorf("bundled: %v, want %v", got, want)
	}
	// a bundled copy has the file's fingerprint, so a saved conversation loads into either
	fm, err := Load(p, LoadOptions{Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	bm, err := Load(src.Path, LoadOptions{Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	if fm.Fingerprint() != bm.Fingerprint() {
		t.Errorf("fingerprints differ: file %s, bundled %s", fm.Fingerprint(), bm.Fingerprint())
	}
	fm.Close()
	bm.Close()
	if _, err := Load("bundled:nope", LoadOptions{}); err == nil {
		t.Error("bundled:nope: want an error")
	}
	// from memory, deliberately misaligned, as go:embed data can be
	odd := make([]byte, len(data)+1)
	copy(odd[1:], data)
	catalog.Embed("odd", odd[1:])
	emb, err := catalog.Resolve("odd", dir)
	if err != nil || emb.Path != "embedded:odd" {
		t.Fatalf("%+v %v", emb, err)
	}
	if got := answer(Load(emb.Path, LoadOptions{Threads: 4})); !slices.Equal(got, want) {
		t.Errorf("embedded: %v, want %v", got, want)
	}
}
