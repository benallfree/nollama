package catalog

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func sum(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func write(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestBundle(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "prog")
	prog := bytes.Repeat([]byte("program!"), 10001) // not a multiple of the alignment
	write(t, exe, prog)
	a, b := []byte("GGUF first model"), bytes.Repeat([]byte("GGUF second"), 9000)
	write(t, filepath.Join(dir, "a.gguf"), a)
	write(t, filepath.Join(dir, "b.gguf"), b)

	out := filepath.Join(dir, "bundled")
	list, err := Bundle(out, exe, []Source{
		{Name: "alpha", Path: filepath.Join(dir, "a.gguf"), SHA256: sum(a)},
		{Name: "beta", Data: b},
	})
	if err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(out)
	if runtime.GOOS != "windows" && st.Mode().Perm()&0o100 == 0 { // Windows has no executable bit
		t.Errorf("bundle isn't executable: %v", st.Mode())
	}
	all, _ := os.ReadFile(out)
	if !bytes.Equal(all[:len(prog)], prog) {
		t.Fatal("the program isn't at the start, unchanged")
	}
	exeSize, got, err := ReadBundle(out)
	if err != nil || exeSize != int64(len(prog)) || len(got) != 2 {
		t.Fatalf("ReadBundle: %d %+v %v", exeSize, got, err)
	}
	for i, want := range [][]byte{a, b} {
		m := got[i]
		if m != list[i] || m.Offset%bundleAlign != 0 || m.SHA256 != sum(want) || !bytes.Equal(all[m.Offset:m.Offset+m.Size], want) {
			t.Errorf("model %d: %+v", i, m)
		}
	}

	// bundling a bundled program copies only the program
	again := filepath.Join(dir, "again")
	if _, err := Bundle(again, out, []Source{{Name: "alpha", Path: out, Offset: got[0].Offset, Size: got[0].Size}}); err != nil {
		t.Fatal(err)
	}
	exeSize, got2, err := ReadBundle(again)
	if err != nil || exeSize != int64(len(prog)) || len(got2) != 1 || got2[0].SHA256 != sum(a) {
		t.Fatalf("rebundle: %d %+v %v", exeSize, got2, err)
	}

	if _, err := Bundle(filepath.Join(dir, "bad"), exe, []Source{{Name: "x", Data: a, SHA256: sum(b)}}); err == nil ||
		!strings.Contains(err.Error(), "checksum") {
		t.Errorf("wrong SHA-256: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "bad")); err == nil {
		t.Error("a failed bundle left its file behind")
	}
	if _, err := Bundle(filepath.Join(dir, "dup"), exe, []Source{{Name: "x", Data: a}, {Name: "x", Data: b}}); err == nil {
		t.Error("two models with one name: want an error")
	}

	// a plain file has no bundle
	if n, l, err := ReadBundle(exe); err != nil || l != nil || n != int64(len(prog)) {
		t.Errorf("plain file: %d %v %v", n, l, err)
	}
	// a damaged trailer is an error, not a crash
	broken := append([]byte(nil), all...)
	copy(broken[len(broken)-16:], bytes.Repeat([]byte{0xff}, 16))
	write(t, filepath.Join(dir, "broken"), broken)
	if _, _, err := ReadBundle(filepath.Join(dir, "broken")); err == nil {
		t.Error("damaged trailer: want an error")
	}

	// signing a Windows executable appends a certificate after the trailer
	signed := append(append([]byte(nil), all...), bytes.Repeat([]byte("certificate"), 1000)...)
	write(t, filepath.Join(dir, "signed"), signed)
	if n, l, err := ReadBundle(filepath.Join(dir, "signed")); err != nil || n != int64(len(prog)) || len(l) != 2 || l[1] != got[1] {
		t.Errorf("signed after bundling: %d %+v %v", n, l, err)
	}
}

func TestBundleResolve(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "prog")
	write(t, exe, []byte("program"))
	m := []byte("GGUF a model")
	out := filepath.Join(dir, "bundled")
	if _, err := Bundle(out, exe, []Source{{Name: "qwen3-0.6b", Data: m}, {Name: "mine", Data: []byte("GGUF mine")}}); err != nil {
		t.Fatal(err)
	}
	if err := UseBundle(out); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { UseBundle("") })

	models := filepath.Join(dir, "models")
	os.MkdirAll(models, 0o755)
	write(t, filepath.Join(models, "qwen3-0.6b.gguf"), []byte("a download")) // shadowed by the bundled one
	write(t, filepath.Join(models, "qwen3-1.7b.gguf"), []byte("another"))

	src, err := Resolve("qwen3:0.6b", models)
	if err != nil || !src.Inside || src.Path != "bundled:qwen3-0.6b" || src.File != out || src.Offset == 0 || src.Size != int64(len(m)) || src.SHA256 != sum(m) || !src.Present() {
		t.Fatalf("bundled: %+v %v", src, err)
	}
	if err := Fetch(src, nil, nil); err != nil {
		t.Errorf("Fetch of a bundled model: %v", err)
	}
	if err := Verify(src); err != nil {
		t.Errorf("Verify: %v", err)
	}
	if mine, _ := Resolve("mine", models); mine.Path == src.Path {
		t.Errorf("two bundled models share a path: %s", src.Path)
	}
	if again, ok := Builtin(src.Path); !ok || again.File != src.File || again.Offset != src.Offset {
		t.Errorf("Builtin(%s): %+v %v", src.Path, again, ok)
	}
	if again, err := Resolve(src.Path, models); err != nil || again.Offset != src.Offset {
		t.Errorf("Resolve(%s): %+v %v", src.Path, again, err)
	}
	if _, ok := Builtin("bundled:nope"); ok {
		t.Error("bundled:nope: found")
	}
	if dl, _ := Resolve("qwen3-1.7b", models); dl.Inside || dl.Path != filepath.Join(models, "qwen3-1.7b.gguf") {
		t.Errorf("download: %+v", dl)
	}

	got, err := Available(models)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, l := range got {
		names = append(names, l.Name)
		if (l.Name == "qwen3-1.7b") == l.Inside {
			t.Errorf("%s: Inside = %v", l.Name, l.Inside)
		}
	}
	if strings.Join(names, ",") != "mine,qwen3-0.6b,qwen3-1.7b" {
		t.Errorf("Available: %v", names)
	}

	// a model whose bytes were changed after bundling fails Verify
	all, _ := os.ReadFile(out)
	all[src.Offset] ^= 1
	write(t, out, all)
	if err := Verify(src); err == nil {
		t.Error("Verify of a changed model: want an error")
	}

	UseBundle("")
	if src, _ := Resolve("qwen3-0.6b", models); src.Inside {
		t.Error("UseBundle(\"\") left bundled models on")
	}
}

func TestEmbed(t *testing.T) {
	data := []byte("GGUF embedded")
	Embed("qwen3:4b", data)
	t.Cleanup(func() {
		inside.mu.Lock()
		delete(inside.embedded, "qwen3-4b")
		inside.mu.Unlock()
	})
	src, err := Resolve("qwen3-4b", t.TempDir())
	if err != nil || !src.Inside || src.Path != "embedded:qwen3-4b" || !bytes.Equal(src.Data, data) || src.SHA256 == "" || !src.Present() {
		t.Fatalf("embedded: %+v %v", src, err)
	}
	if Verify(src) == nil {
		t.Error("these bytes aren't Qwen3 4B: Verify should fail")
	}
	got, _ := Available(t.TempDir())
	if len(got) != 1 || got[0].Name != "qwen3-4b" || !got[0].Inside {
		t.Errorf("Available: %+v", got)
	}
}
