// Package models finds model files: a name nollama knows (downloaded from its publisher and checked against its
// SHA-256), an hf:<org>/<repo>/<file>.gguf from Hugging Face, or a path to a .gguf file.
package models

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Known is a model nollama can fetch by name: a GGUF file from its publisher, checked against its SHA-256.
type Known struct {
	Name   string
	URL    string
	SHA256 string
	Size   int64
	About  string
}

// Catalog lists the models nollama knows by name. Each is one the engine supports, from its publisher's own
// repository.
var Catalog = []Known{
	{Name: "qwen3-0.6b", About: "Qwen3 0.6B, 8-bit: the smallest; tagging and short extraction",
		URL:    "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
		SHA256: "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031", Size: 639446688},
	{Name: "qwen3-1.7b", About: "Qwen3 1.7B, 8-bit: a good default for a small server",
		URL:    "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
		SHA256: "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a", Size: 1834426016},
	{Name: "qwen3-4b", About: "Qwen3 4B, 4-bit: better answers, needs about 3 GB of memory",
		URL:    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
		SHA256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", Size: 2497280256},
}

// Find returns the catalog entry for name.
func Find(name string) (Known, bool) {
	for _, m := range Catalog {
		if m.Name == name {
			return m, true
		}
	}
	return Known{}, false
}

// Names lists the catalog's names, comma-separated.
func Names() string {
	names := make([]string, len(Catalog))
	for i, m := range Catalog {
		names[i] = m.Name
	}
	return strings.Join(names, ", ")
}

// DefaultDir is where downloaded models go: nollama/models in the user's cache folder.
func DefaultDir() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		dir = os.TempDir()
	}
	return filepath.Join(dir, "nollama", "models")
}

// Source is where a model's file comes from: a local path (used as it is), or a download into the cache.
type Source struct {
	Name   string // the name asked for (an Ollama-style "qwen3:0.6b" comes back as "qwen3-0.6b")
	Path   string // the local file
	URL    string // "" for a local file
	SHA256 string // "" when unverified (an hf: model)
	Size   int64
}

// Normalize turns an Ollama-style name into nollama's: "qwen3:0.6b" is "qwen3-0.6b", and ":latest" is dropped.
// Anything else comes back as it is.
func Normalize(spec string) string {
	spec = strings.TrimSpace(spec)
	if strings.HasPrefix(spec, "hf:") || strings.HasSuffix(spec, ".gguf") || strings.ContainsRune(spec, os.PathSeparator) {
		return spec
	}
	spec = strings.TrimSuffix(spec, ":latest")
	return strings.Replace(spec, ":", "-", 1)
}

// Resolve turns a model spec into a source: a catalog name (or its Ollama spelling, "qwen3:0.6b"),
// "hf:<org>/<repo>/<file>.gguf", or a path to a .gguf file. Downloads go under dir.
func Resolve(spec, dir string) (Source, error) {
	spec = Normalize(spec)
	if m, ok := Find(spec); ok {
		return Source{Name: spec, Path: filepath.Join(dir, m.Name+".gguf"), URL: m.URL, SHA256: m.SHA256, Size: m.Size}, nil
	}
	if rest, ok := strings.CutPrefix(spec, "hf:"); ok {
		parts := strings.SplitN(rest, "/", 3)
		if len(parts) != 3 || !strings.HasSuffix(parts[2], ".gguf") {
			return Source{}, fmt.Errorf("model %q: want hf:<org>/<repo>/<file>.gguf", spec)
		}
		file := strings.NewReplacer("/", "_", "..", "_").Replace(rest)
		return Source{Name: spec, Path: filepath.Join(dir, "hf", file),
			URL: "https://huggingface.co/" + parts[0] + "/" + parts[1] + "/resolve/main/" + parts[2]}, nil
	}
	if strings.HasSuffix(spec, ".gguf") || strings.ContainsRune(spec, os.PathSeparator) {
		return Source{Name: strings.TrimSuffix(filepath.Base(spec), ".gguf"), Path: ExpandHome(spec)}, nil
	}
	return Source{}, fmt.Errorf("model %q: not a model nollama knows (%s), an hf:<org>/<repo>/<file>.gguf, or a path to a .gguf file", spec, Names())
}

// ExpandHome expands a leading ~/ to the user's home folder.
func ExpandHome(p string) string {
	if rest, ok := strings.CutPrefix(p, "~/"); ok {
		if h, err := os.UserHomeDir(); err == nil {
			return filepath.Join(h, rest)
		}
	}
	return p
}

// Present reports whether the source's file is on disk.
func (s Source) Present() bool {
	_, err := os.Stat(s.Path)
	return err == nil
}

// Fetch makes sure the source's file is on disk, downloading it (to a .part file, then renamed) when it isn't, and
// checking its SHA-256 when known. progress, when set, gets the bytes done and the total (0 when unknown).
func Fetch(src Source, client *http.Client, progress func(done, total int64)) error {
	if _, err := os.Stat(src.Path); err == nil {
		return nil
	} else if src.URL == "" {
		return fmt.Errorf("model %s: %w", src.Path, err)
	}
	if err := os.MkdirAll(filepath.Dir(src.Path), 0o755); err != nil {
		return err
	}
	if client == nil {
		client = http.DefaultClient
	}
	res, err := client.Get(src.URL)
	if err != nil {
		return fmt.Errorf("downloading %s: %w", src.Name, err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("downloading %s: %s said %s", src.Name, src.URL, res.Status)
	}
	total := src.Size
	if total == 0 && res.ContentLength > 0 {
		total = res.ContentLength
	}
	part := fmt.Sprintf("%s.%d.part", src.Path, os.Getpid()) // two processes fetching at once don't share a file
	f, err := os.Create(part)
	if err != nil {
		return err
	}
	h := sha256.New()
	pw := &progressWriter{total: total, progress: progress}
	_, err = io.Copy(io.MultiWriter(f, h, pw), res.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(part)
		return fmt.Errorf("downloading %s: %w", src.Name, err)
	}
	if src.SHA256 != "" {
		if got := hex.EncodeToString(h.Sum(nil)); got != src.SHA256 {
			_ = os.Remove(part)
			return fmt.Errorf("downloading %s: checksum %s, want %s", src.Name, got, src.SHA256)
		}
	}
	return os.Rename(part, src.Path)
}

type progressWriter struct {
	total, done int64
	progress    func(done, total int64)
}

func (p *progressWriter) Write(b []byte) (int, error) {
	p.done += int64(len(b))
	if p.progress != nil {
		p.progress(p.done, p.total)
	}
	return len(b), nil
}

// Local is a model file found in a models folder.
type Local struct {
	Name    string // catalog name, hf:<org>/<repo>/<file>.gguf, or the file name without .gguf
	Path    string
	Size    int64
	ModTime int64 // Unix seconds
}

// List returns the .gguf files under dir: catalog downloads by name, hf: downloads by their spec.
func List(dir string) ([]Local, error) {
	var out []Local
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) && p == dir {
				return filepath.SkipDir
			}
			return err
		}
		if d.IsDir() || !strings.HasSuffix(p, ".gguf") {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		name := strings.TrimSuffix(filepath.Base(p), ".gguf")
		if strings.HasPrefix(rel, "hf"+string(os.PathSeparator)) {
			// hf/<org>_<repo>_<file>.gguf: the org and repo can't be told apart again, so the file stands for itself
			name = p
		}
		out = append(out, Local{Name: name, Path: p, Size: fi.Size(), ModTime: fi.ModTime().Unix()})
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, err
}
