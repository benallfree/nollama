package models

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// A bundle is an executable with models added to its end, so it runs them with nothing to download:
//
//	the executable, as built
//	GGUF files, each starting on a 64 KiB boundary (so each can be mmapped where it lies)
//	a JSON manifest: the executable's own size, and each model's name, offset, size and SHA-256
//	a 32-byte trailer: bundleMagic, then the manifest's offset and length (uint64, little-endian)
//
// Operating systems load an executable by the headers at its start and ignore bytes past its end, so the program
// runs as before. At startup, Resolve reads the trailer of the running executable and finds the models by name.

const (
	bundleMagic   = "NOLLAMA-BUNDLE\x00\x01"
	trailerSize   = 32
	bundleAlign   = 64 << 10 // covers 4K and 16K pages and Windows' 64K mapping granularity
	bundleVersion = 1
)

// Bundled is a model inside an executable.
type Bundled struct {
	Name   string `json:"name"`
	Offset int64  `json:"offset"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type manifest struct {
	Version int       `json:"version"`
	Exe     int64     `json:"exe"` // the executable's size before any models
	Models  []Bundled `json:"models"`
}

// ReadBundle lists the models bundled in the file at path, and the size of the executable without them. A file with
// no bundle has no models, and exe is its whole size.
func ReadBundle(path string) (exe int64, list []Bundled, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return 0, nil, err
	}
	m, err := readManifest(f, st.Size())
	if err != nil || m == nil {
		return st.Size(), nil, err
	}
	return m.Exe, m.Models, nil
}

func readManifest(f io.ReaderAt, size int64) (*manifest, error) {
	pos, err := findTrailer(f, size)
	if err != nil || pos < 0 {
		return nil, err
	}
	var t [trailerSize]byte
	if _, err := f.ReadAt(t[:], pos); err != nil {
		return nil, err
	}
	off, n := int64(binary.LittleEndian.Uint64(t[16:])), int64(binary.LittleEndian.Uint64(t[24:]))
	if off < 0 || n <= 0 || n > 16<<20 || off+n != pos {
		return nil, errors.New("bundle: bad trailer")
	}
	b := make([]byte, n)
	if _, err := f.ReadAt(b, off); err != nil {
		return nil, err
	}
	var m manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("bundle: manifest: %w", err)
	}
	if m.Version != bundleVersion {
		return nil, fmt.Errorf("bundle: version %d, this nollama reads %d", m.Version, bundleVersion)
	}
	for _, b := range m.Models {
		if b.Offset < m.Exe || b.Size <= 0 || b.Offset+b.Size > off {
			return nil, fmt.Errorf("bundle: model %s out of range", b.Name)
		}
	}
	return &m, nil
}

// findTrailer returns where the bundle's trailer starts, or -1 when there is none. It is normally the file's last 32
// bytes; signing a Windows executable afterward appends its certificate after it, so the last 256 KiB are searched
// too, for a trailer whose manifest ends right where it starts.
func findTrailer(f io.ReaderAt, size int64) (int64, error) {
	if size < trailerSize {
		return -1, nil
	}
	tail := min(size, 256<<10)
	b := make([]byte, tail)
	if _, err := f.ReadAt(b, size-tail); err != nil && err != io.EOF {
		return -1, err
	}
	if string(b[tail-trailerSize:tail-trailerSize+16]) == bundleMagic {
		return size - trailerSize, nil
	}
	for end := len(b); ; {
		i := bytes.LastIndex(b[:end], []byte(bundleMagic))
		if i < 0 || i+trailerSize > len(b) {
			if i < 0 {
				return -1, nil
			}
			end = i
			continue
		}
		pos := size - tail + int64(i)
		off, n := int64(binary.LittleEndian.Uint64(b[i+16:])), int64(binary.LittleEndian.Uint64(b[i+24:]))
		if off >= 0 && n > 0 && off+n == pos {
			return pos, nil
		}
		end = i
	}
}

// Bundle writes to out the executable at exe followed by the models in srcs, so out runs them without downloading.
// A bundle already on exe is left off (only the executable is copied), so a bundled nollama can make another
// bundle. Each model is checked against its SHA-256 as it is copied when one is known, and the manifest records
// every model's SHA-256. out is written next to itself and renamed into place, with exe's file mode.
func Bundle(out, exe string, srcs []Source) (list []Bundled, err error) {
	ef, err := os.Open(exe)
	if err != nil {
		return nil, err
	}
	defer ef.Close()
	st, err := ef.Stat()
	if err != nil {
		return nil, err
	}
	base := st.Size()
	if m, err := readManifest(ef, base); err != nil {
		return nil, err
	} else if m != nil {
		base = m.Exe
	}
	tmp := fmt.Sprintf("%s.%d.tmp", out, os.Getpid())
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, st.Mode().Perm()|0o700)
	if err != nil {
		return nil, err
	}
	defer func() {
		f.Close()
		if err != nil {
			os.Remove(tmp)
		}
	}()
	if _, err := io.Copy(f, io.NewSectionReader(ef, 0, base)); err != nil {
		return nil, err
	}
	pos := base
	seen := map[string]bool{}
	for _, src := range srcs {
		if seen[src.Name] {
			return nil, fmt.Errorf("bundle: two models named %s", src.Name)
		}
		seen[src.Name] = true
		pad := (bundleAlign - pos%bundleAlign) % bundleAlign
		if _, err := f.Write(make([]byte, pad)); err != nil {
			return nil, err
		}
		pos += pad
		r, size, closer, err := openSource(src)
		if err != nil {
			return nil, fmt.Errorf("bundle: %s: %w", src.Name, err)
		}
		h := sha256.New()
		_, err = io.Copy(io.MultiWriter(f, h), r)
		closer()
		if err != nil {
			return nil, fmt.Errorf("bundle: %s: %w", src.Name, err)
		}
		sum := hex.EncodeToString(h.Sum(nil))
		if src.SHA256 != "" && sum != src.SHA256 {
			return nil, fmt.Errorf("bundle: %s: checksum %s, want %s", src.Name, sum, src.SHA256)
		}
		list = append(list, Bundled{Name: src.Name, Offset: pos, Size: size, SHA256: sum})
		pos += size
	}
	mb, err := json.Marshal(manifest{Version: bundleVersion, Exe: base, Models: list})
	if err != nil {
		return nil, err
	}
	var t [trailerSize]byte
	copy(t[:], bundleMagic)
	binary.LittleEndian.PutUint64(t[16:], uint64(pos))
	binary.LittleEndian.PutUint64(t[24:], uint64(len(mb)))
	if _, err := f.Write(append(mb, t[:]...)); err != nil {
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	return list, os.Rename(tmp, out)
}

// openSource returns a reader for the model's bytes and their count.
func openSource(src Source) (io.Reader, int64, func(), error) {
	if src.Data != nil {
		return bytes.NewReader(src.Data), int64(len(src.Data)), func() {}, nil
	}
	file := src.File
	if file == "" {
		file = src.Path
	}
	f, err := os.Open(file)
	if err != nil {
		return nil, 0, nil, err
	}
	size := src.Size
	if src.File == "" && src.Offset == 0 {
		st, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, 0, nil, err
		}
		size = st.Size()
	}
	return io.NewSectionReader(f, src.Offset, size), size, func() { f.Close() }, nil
}

// Verify reads the model's bytes and checks them against its SHA-256. Loading never does this (it would cost a
// read of the whole model at every start): downloads are checked as they arrive and bundles as they're made.
func Verify(src Source) error {
	if src.SHA256 == "" {
		return fmt.Errorf("%s: no SHA-256 known to check it against", src.Name)
	}
	r, _, closer, err := openSource(src)
	if err != nil {
		return err
	}
	defer closer()
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != src.SHA256 {
		return fmt.Errorf("%s: checksum %s, want %s", src.Name, got, src.SHA256)
	}
	return nil
}

// The models inside the running program: bundled at the end of its executable, or registered with Embed.
var inside struct {
	mu       sync.Mutex
	loaded   bool
	path     string // the file bundles are read from ("" for none)
	modTime  int64
	bundled  []Bundled
	embedded map[string]embedded
	err      error
}

type embedded struct {
	data    []byte
	sha256  string
	modTime int64
}

// UseBundle reads bundled models from the file at path instead of the running executable; "" turns bundled models
// off. Without a call, Resolve reads the running executable's own bundle the first time it's needed.
func UseBundle(path string) error {
	inside.mu.Lock()
	defer inside.mu.Unlock()
	inside.loaded = false
	return loadBundle(path)
}

func loadBundle(path string) error {
	inside.loaded, inside.path, inside.bundled, inside.err, inside.modTime = true, path, nil, nil, 0
	if path == "" {
		return nil
	}
	if st, err := os.Stat(path); err == nil {
		inside.modTime = st.ModTime().Unix()
	}
	_, inside.bundled, inside.err = ReadBundle(path)
	if inside.err != nil {
		inside.err = fmt.Errorf("models bundled in %s: %w", path, inside.err)
	}
	return inside.err
}

// bundle returns the bundled models, reading the running executable's bundle the first time.
func bundle() (path string, list []Bundled, modTime int64) {
	inside.mu.Lock()
	defer inside.mu.Unlock()
	if !inside.loaded {
		exe, err := os.Executable()
		if err == nil {
			exe, _ = filepath.EvalSymlinks(exe)
		}
		_ = loadBundle(exe) // a damaged bundle means no bundled models; BundleError says why
	}
	return inside.path, inside.bundled, inside.modTime
}

// BundleError reports why the executable's bundled models couldn't be read, if they couldn't.
func BundleError() error {
	bundle()
	inside.mu.Lock()
	defer inside.mu.Unlock()
	return inside.err
}

// Embed makes a GGUF in memory available by name, usually one compiled in with go:embed:
//
//	//go:embed qwen3-0.6b.gguf
//	var qwen []byte
//
//	models.Embed("qwen3-0.6b", qwen)
//
// Resolve then finds it before any download, and engine.Load(src.Path) reads it in place. A catalog name gets the
// catalog's SHA-256 (models.Verify checks it). data must not change afterward.
func Embed(name string, data []byte) {
	name = Normalize(name)
	e := embedded{data: data}
	if k, ok := Find(name); ok {
		e.sha256 = k.SHA256
	}
	if exe, err := os.Executable(); err == nil {
		if st, err := os.Stat(exe); err == nil {
			e.modTime = st.ModTime().Unix()
		}
	}
	inside.mu.Lock()
	defer inside.mu.Unlock()
	if inside.embedded == nil {
		inside.embedded = map[string]embedded{}
	}
	inside.embedded[name] = e
}

// The Path of a model inside the program: engine.Load opens these like files.
const (
	BundledPrefix  = "bundled:"
	EmbeddedPrefix = "embedded:"
)

// Builtin returns the model inside the program that path names: "bundled:<name>" for one bundled at the end of the
// executable, "embedded:<name>" for one registered with Embed. These are the Path of such a model's Source, so code
// that loads by path (engine.Load(src.Path)) works for every model.
func Builtin(path string) (Source, bool) {
	if name, ok := strings.CutPrefix(path, EmbeddedPrefix); ok {
		return findEmbedded(name)
	}
	if name, ok := strings.CutPrefix(path, BundledPrefix); ok {
		return findBundled(name)
	}
	return Source{}, false
}

func findEmbedded(name string) (Source, bool) {
	inside.mu.Lock()
	e, ok := inside.embedded[name]
	inside.mu.Unlock()
	if !ok {
		return Source{}, false
	}
	return Source{Name: name, Path: EmbeddedPrefix + name, Data: e.data, Size: int64(len(e.data)), SHA256: e.sha256, Inside: true}, true
}

func findBundled(name string) (Source, bool) {
	file, list, _ := bundle()
	for _, b := range list {
		if b.Name == name {
			return Source{Name: name, Path: BundledPrefix + name, File: file, Offset: b.Offset, Size: b.Size,
				SHA256: b.SHA256, Inside: true}, true
		}
	}
	return Source{}, false
}

// findInside returns the model named that's inside the program: embedded first, then bundled.
func findInside(name string) (Source, bool) {
	if src, ok := findEmbedded(name); ok {
		return src, true
	}
	return findBundled(name)
}

// listInside returns the models inside the program, embedded and bundled.
func listInside() []Local {
	var out []Local
	seen := map[string]bool{}
	inside.mu.Lock()
	for name, e := range inside.embedded {
		seen[name] = true
		out = append(out, Local{Name: name, Path: EmbeddedPrefix + name, Size: int64(len(e.data)), ModTime: e.modTime,
			SHA256: e.sha256, Inside: true})
	}
	inside.mu.Unlock()
	_, list, mod := bundle()
	for _, b := range list {
		if seen[b.Name] {
			continue
		}
		out = append(out, Local{Name: b.Name, Path: BundledPrefix + b.Name, Size: b.Size, ModTime: mod, SHA256: b.SHA256, Inside: true})
	}
	return out
}

// Available lists every model ready to load without a download: the ones inside the program (embedded or bundled)
// and the downloads under dir. A model both inside and downloaded is listed once, as the one inside, which is the
// one Resolve picks.
func Available(dir string) ([]Local, error) {
	out := listInside()
	seen := map[string]bool{}
	for _, l := range out {
		seen[l.Name] = true
	}
	local, err := List(dir)
	for _, l := range local {
		if !seen[l.Name] {
			out = append(out, l)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, err
}
