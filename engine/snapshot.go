package engine

import (
	"bufio"
	"compress/flate"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"
	"time"
)

// Snapshots let the app keep a thread's state, or the prompt cache's entries, itself: SaveThread writes the state to
// any io.Writer (a file, a database blob, object storage) and LoadThread reads it back, after a restart or on another
// machine, so a long conversation continues without reading the whole of it again. SavePromptCache and LoadPromptCache
// do the same for the prompt cache, e.g. a long system prompt read once at build time.
//
// A snapshot names the model it came from by a fingerprint of the GGUF file (its header and samples of its weights),
// so it loads only into the same model file, whatever the file is called, and a file with the same shapes but other
// weights (a finetune) refuses it. It's versioned, ends with a CRC-32, and is half precision unless
// SnapshotOptions.Full asks for exact floats.
//
// Layout (little endian):
//
//	magic "NLKV" | version u16 | kind u8 | flags u8 | fingerprint [32]byte |
//	arch, quant (u16 length + bytes) | layers, kvHeads, headDim u32 | entries u32 | created (unix seconds) i64
//	body (flate-compressed when flagCompressed), per entry: label (u16 + bytes) | n u32 | tokens [n]i32 |
//	    per layer: keys then values, kvHeads*headDim*n values each (f16, or f32 with flagFull)
//	crc32 (IEEE) of the body as written, u32
//
// Keys are stored [kvd][n], each channel's positions in a row, as the engine keeps them. Compressed f16 is
// byte-shuffled first (all low bytes, then all high bytes), which flate squeezes much better than interleaved halves.

const (
	snapshotMagic   = "NLKV"
	snapshotVersion = 1

	snapshotThread      = 1
	snapshotPromptCache = 2

	flagFull       = 1 << 0
	flagCompressed = 1 << 1
)

// SnapshotOptions say how SaveThread and SavePromptCache write, and where LoadThread keeps what it reads.
type SnapshotOptions struct {
	// Full writes exact 32-bit floats: a loaded state then continues bit for bit like the original. Off, it's half
	// precision (half the bytes), as thread files on disk are: answers match to within rounding.
	Full bool
	// Compress deflates the body (fastest level): see the README for what it saves; mostly worth it at full precision.
	Compress bool
	// ThreadDir, for SaveThread, also finds a thread kept only on disk there; for LoadThread, also writes the loaded
	// state there (within ThreadDisk bytes), as Generate would, so it survives memory being off or full.
	ThreadDir  string
	ThreadDisk int64
}

// SnapshotInfo describes a snapshot: what LoadThread and LoadPromptCache read, or ReadSnapshotInfo found.
type SnapshotInfo struct {
	Version     int
	Kind        string // "thread" or "prompt cache"
	Fingerprint string // hex, as Model.Fingerprint
	Arch, Quant string
	Entries     int
	Tokens      int // over all entries
	Labels      []string
	Full        bool
	Compressed  bool
	Created     time.Time
}

// ErrSnapshotModel means the snapshot belongs to another model file (or a different snapshot kind).
var ErrSnapshotModel = errors.New("engine: snapshot is for another model")

// ErrSnapshotCorrupt means the snapshot is truncated, damaged or not a snapshot.
var ErrSnapshotCorrupt = errors.New("engine: snapshot is damaged or not a snapshot")

// ErrNoThread means SaveThread found no state for the thread.
var ErrNoThread = errors.New("engine: no state kept for this thread")

// Fingerprint identifies the model file: sha256 over the GGUF header (metadata and tensor table) and 64 evenly spaced
// 4 KB samples of the weights, so a renamed copy matches and other weights don't. Computed once, at load (well under a
// millisecond: the samples are a few page reads).
func (m *Model) Fingerprint() string { return fmt.Sprintf("%x", m.fingerprint) }

func ggufFingerprint(g *ggufFile) [32]byte {
	h := sha256.New()
	d := g.data
	end := g.dataOff // the header: metadata and tensor table
	h.Write(d[:end])
	const samples, size = 64, 4096
	if span := len(d) - end; span > 0 {
		for i := 0; i < samples; i++ {
			at := end + int(int64(span)*int64(i)/samples)
			h.Write(d[at:min(at+size, len(d))])
		}
	}
	var out [32]byte
	h.Sum(out[:0])
	return out
}

// SaveThread writes the state kept for a thread (in memory, or in o.ThreadDir) to w.
func (m *Model) SaveThread(w io.Writer, scope, id string, o SnapshotOptions) error {
	e := m.findThread(scope, id, o.ThreadDir)
	if e == nil {
		return ErrNoThread
	}
	return m.writeSnapshot(w, snapshotThread, []snapEntry{{label: id, e: e}}, o)
}

// LoadThread reads a snapshot SaveThread wrote and keeps it as the state of thread id in scope (they needn't be the
// ones it was saved under), in memory and, with o.ThreadDir, on disk. The next Generate on that thread whose prompt
// starts with the saved conversation reads only what's new.
func (m *Model) LoadThread(r io.Reader, scope, id string, o SnapshotOptions) (SnapshotInfo, error) {
	info, entries, err := m.readSnapshot(r, snapshotThread)
	if err != nil {
		return info, err
	}
	if len(entries) != 1 {
		return info, ErrSnapshotCorrupt
	}
	e := entries[0].e
	e.scope = scope
	if int64(len(e.tokens))*m.tokenBytes() > m.threads.budget && o.ThreadDir == "" {
		return info, errors.New("engine: the thread doesn't fit in thread memory (LoadOptions.ThreadMemory); pass a ThreadDir")
	}
	m.keepThread(scope, id, e)
	if o.ThreadDir != "" {
		m.threads.diskMu.Lock()
		err := m.writeThread(m.threadFile(o.ThreadDir, scope, id), scope, id, e)
		if err == nil {
			evictDir(o.ThreadDir, o.ThreadDisk)
		}
		m.threads.diskMu.Unlock()
		if err != nil {
			return info, err
		}
	}
	return info, nil
}

// SavePromptCache writes the prompt cache's entries for scope to w (all of them, pinned or not). A host serving several
// apps saves each app's scope apart.
func (m *Model) SavePromptCache(w io.Writer, scope string, o SnapshotOptions) (int, error) {
	c := &m.prefix
	c.mu.Lock()
	var entries []snapEntry
	for _, e := range c.entries {
		if e.scope == scope {
			entries = append(entries, snapEntry{e: e}) // entries are never changed in place: safe to read unlocked
		}
	}
	c.mu.Unlock()
	return len(entries), m.writeSnapshot(w, snapshotPromptCache, entries, o)
}

// LoadPromptCache reads a snapshot SavePromptCache wrote into the prompt cache under scope, pinned with pin when it's
// not "" (as Warm pins), within the cache's budget: entries that don't fit are kept as far as the budget allows, and
// the info says how many tokens were kept. The cache must be on (LoadOptions.PromptCache).
func (m *Model) LoadPromptCache(r io.Reader, scope, pin string) (SnapshotInfo, int, error) {
	if m.prefix.budget <= 0 {
		return SnapshotInfo{}, 0, errors.New("engine: the prompt cache is off (LoadOptions.PromptCache)")
	}
	info, entries, err := m.readSnapshot(r, snapshotPromptCache)
	if err != nil {
		return info, 0, err
	}
	kept := 0
	for _, se := range entries {
		se.e.scope = scope
		kept += m.insertPrefix(se.e, pin)
	}
	return info, kept, nil
}

// ReadSnapshotInfo reads a snapshot's header without loading it or checking it against a model.
func ReadSnapshotInfo(r io.Reader) (SnapshotInfo, error) {
	info, _, err := readSnapshotHeader(bufio.NewReader(r))
	return info, err
}

// findThread returns the thread's whole kept state, from memory or dir.
func (m *Model) findThread(scope, id, dir string) *prefixEntry {
	t := &m.threads
	t.mu.Lock()
	te := t.mem[threadKey(scope, id)]
	t.mu.Unlock()
	if te != nil {
		return te.e
	}
	if dir == "" {
		return nil
	}
	m.FlushThreads()
	e, err := m.readThread(m.threadFile(dir, scope, id), scope, id)
	if err != nil {
		return nil
	}
	return e
}

// insertPrefix puts e (all computed) in the prompt cache, cut to the budget, evicting unpinned entries as storePrefix
// does. It returns the tokens kept.
func (m *Model) insertPrefix(e *prefixEntry, pin string) int {
	c := &m.prefix
	per := m.tokenBytes()
	c.mu.Lock()
	defer c.mu.Unlock()
	n := min(len(e.tokens), int(c.budget/per))
	if n < prefixMin {
		return 0
	}
	if n < len(e.tokens) { // keep the start: that's where prompts repeat
		kvd := m.hp.kvHeads * m.hp.headDim
		en := len(e.tokens)
		for li := range e.k {
			k := make([]float32, kvd*n)
			for i := 0; i < kvd; i++ {
				copy(k[i*n:(i+1)*n], e.k[li][i*en:i*en+n])
			}
			e.k[li] = k
			e.v[li] = e.v[li][:n*kvd]
		}
		e.tokens = e.tokens[:n]
	}
	if pin != "" {
		e.pins = map[string]bool{pin: true}
	}
	for i, cand := range c.entries { // the same prompt start is cached already: replace it
		if cand.scope == e.scope && equalInts(cand.tokens, e.tokens) {
			for p := range cand.pins {
				if e.pins == nil {
					e.pins = map[string]bool{}
				}
				e.pins[p] = true
			}
			c.used -= int64(len(cand.tokens)) * per
			c.entries = append(c.entries[:i], c.entries[i+1:]...)
			break
		}
	}
	size := int64(n) * per
	for c.used+size > c.budget {
		old := -1
		for i, cand := range c.entries {
			if len(cand.pins) == 0 && (old < 0 || cand.seq < c.entries[old].seq) {
				old = i
			}
		}
		if old < 0 {
			return 0
		}
		c.used -= int64(len(c.entries[old].tokens)) * per
		c.entries = append(c.entries[:old], c.entries[old+1:]...)
	}
	c.seq++
	e.seq = c.seq
	c.entries = append(c.entries, e)
	c.used += size
	return n
}

type snapEntry struct {
	label string
	e     *prefixEntry
}

func kindName(k uint8) string {
	switch k {
	case snapshotThread:
		return "thread"
	case snapshotPromptCache:
		return "prompt cache"
	}
	return fmt.Sprintf("kind %d", k)
}

func (m *Model) writeSnapshot(w io.Writer, kind uint8, entries []snapEntry, o SnapshotOptions) error {
	bw := bufio.NewWriterSize(w, 1<<20)
	var flags uint8
	if o.Full {
		flags |= flagFull
	}
	if o.Compress {
		flags |= flagCompressed
	}
	kvd := m.hp.kvHeads * m.hp.headDim
	hw := &errWriter{w: bw}
	hw.write([]byte(snapshotMagic))
	hw.le(uint16(snapshotVersion))
	hw.le(kind)
	hw.le(flags)
	hw.write(m.fingerprint[:])
	hw.str(m.info.Arch)
	hw.str(m.info.Quant)
	hw.le([3]uint32{uint32(m.hp.layers), uint32(m.hp.kvHeads), uint32(m.hp.headDim)})
	hw.le(uint32(len(entries)))
	hw.le(time.Now().Unix())
	if hw.err != nil {
		return hw.err
	}

	crc := crc32.NewIEEE()
	var body io.Writer = io.MultiWriter(bw, crc)
	var fw *flate.Writer
	if o.Compress {
		fw, _ = flate.NewWriter(body, flate.BestSpeed)
		body = fw
	}
	ew := &errWriter{w: body}
	var buf []byte
	for _, se := range entries {
		e := se.e
		n := len(e.tokens)
		ew.str(se.label)
		ew.le(uint32(n))
		tb := make([]byte, 4*n)
		for i, t := range e.tokens {
			binary.LittleEndian.PutUint32(tb[4*i:], uint32(int32(t)))
		}
		ew.write(tb)
		for li := range e.k {
			for _, part := range [][]float32{e.k[li], e.v[li]} {
				if len(part) != kvd*n {
					return fmt.Errorf("engine: snapshot: layer %d has %d values, want %d", li, len(part), kvd*n)
				}
				buf = encodeFloats(buf, part, o.Full, o.Compress)
				ew.write(buf)
			}
		}
		if ew.err != nil {
			return ew.err
		}
	}
	if fw != nil {
		if err := fw.Close(); err != nil {
			return err
		}
	}
	if err := binary.Write(bw, binary.LittleEndian, crc.Sum32()); err != nil {
		return err
	}
	return bw.Flush()
}

// encodeFloats writes vals as f32 or f16 into buf (reused); shuffled puts each byte plane (all lowest bytes, then the
// next) one after another.
func encodeFloats(buf []byte, vals []float32, full, shuffled bool) []byte {
	width := 2
	if full {
		width = 4
	}
	n := len(vals)
	if cap(buf) < width*n {
		buf = make([]byte, width*n)
	}
	buf = buf[:width*n]
	switch {
	case full && !shuffled:
		for i, f := range vals {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(f))
		}
	case full:
		b0, b1, b2, b3 := buf[:n], buf[n:2*n], buf[2*n:3*n], buf[3*n:]
		for i, f := range vals {
			u := math.Float32bits(f)
			b0[i], b1[i], b2[i], b3[i] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
		}
	case !shuffled:
		for i, f := range vals {
			binary.LittleEndian.PutUint16(buf[2*i:], f32ToF16(f))
		}
	default:
		lo, hi := buf[:n], buf[n:]
		for i, f := range vals {
			h := f32ToF16(f)
			lo[i], hi[i] = byte(h), byte(h>>8)
		}
	}
	return buf
}

func decodeFloats(out []float32, buf []byte, full, shuffled bool) {
	n := len(out)
	switch {
	case full && !shuffled:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
	case full:
		b0, b1, b2, b3 := buf[:n], buf[n:2*n], buf[2*n:3*n], buf[3*n:4*n]
		for i := range out {
			out[i] = math.Float32frombits(uint32(b0[i]) | uint32(b1[i])<<8 | uint32(b2[i])<<16 | uint32(b3[i])<<24)
		}
	case !shuffled:
		for i := range out {
			out[i] = f16Table[binary.LittleEndian.Uint16(buf[2*i:])]
		}
	default:
		lo, hi := buf[:n], buf[n:2*n]
		for i := range out {
			out[i] = f16Table[uint16(lo[i])|uint16(hi[i])<<8]
		}
	}
}

type snapHeader struct {
	kind, flags              uint8
	fingerprint              [32]byte
	layers, kvHeads, headDim int
	entries                  int
}

func readSnapshotHeader(r *bufio.Reader) (SnapshotInfo, snapHeader, error) {
	var info SnapshotInfo
	var h snapHeader
	er := &errReader{r: r}
	magic := make([]byte, 4)
	er.read(magic)
	if er.err != nil || string(magic) != snapshotMagic {
		return info, h, ErrSnapshotCorrupt
	}
	var version uint16
	er.le(&version)
	if er.err == nil && version != snapshotVersion {
		return info, h, fmt.Errorf("engine: snapshot version %d, this build reads %d", version, snapshotVersion)
	}
	er.le(&h.kind)
	er.le(&h.flags)
	er.read(h.fingerprint[:])
	info.Arch = er.str()
	info.Quant = er.str()
	var dims [3]uint32
	er.le(&dims)
	var entries uint32
	er.le(&entries)
	var created int64
	er.le(&created)
	if er.err != nil {
		return info, h, ErrSnapshotCorrupt
	}
	h.layers, h.kvHeads, h.headDim, h.entries = int(dims[0]), int(dims[1]), int(dims[2]), int(entries)
	info.Version = int(version)
	info.Kind = kindName(h.kind)
	info.Fingerprint = fmt.Sprintf("%x", h.fingerprint)
	info.Entries = h.entries
	info.Full = h.flags&flagFull != 0
	info.Compressed = h.flags&flagCompressed != 0
	info.Created = time.Unix(created, 0)
	return info, h, nil
}

func (m *Model) readSnapshot(r io.Reader, kind uint8) (SnapshotInfo, []snapEntry, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	info, h, err := readSnapshotHeader(br)
	if err != nil {
		return info, nil, err
	}
	if h.kind != kind || h.fingerprint != m.fingerprint || h.layers != m.hp.layers || h.kvHeads != m.hp.kvHeads || h.headDim != m.hp.headDim {
		return info, nil, ErrSnapshotModel
	}
	if h.entries > 1<<16 {
		return info, nil, ErrSnapshotCorrupt
	}
	full, compressed := h.flags&flagFull != 0, h.flags&flagCompressed != 0
	width := 2
	if full {
		width = 4
	}
	kvd := m.hp.kvHeads * m.hp.headDim

	crc := crc32.NewIEEE()
	tee := &crcReader{r: br, h: crc}
	var body io.Reader = tee
	var fr io.ReadCloser
	if compressed {
		fr = flate.NewReader(tee)
		body = fr
	}
	er := &errReader{r: body}
	entries := make([]snapEntry, 0, h.entries)
	var buf []byte
	for i := 0; i < h.entries; i++ {
		label := er.str()
		var n32 uint32
		er.le(&n32)
		n := int(n32)
		if er.err != nil || n <= 0 || n > m.hp.ctx {
			return info, nil, ErrSnapshotCorrupt
		}
		tb := make([]byte, 4*n)
		er.read(tb)
		e := &prefixEntry{tokens: make([]int, n), k: make([][]float32, h.layers), v: make([][]float32, h.layers)}
		for j := range e.tokens {
			t := int(int32(binary.LittleEndian.Uint32(tb[4*j:])))
			if t < 0 || t >= m.hp.vocab {
				return info, nil, ErrSnapshotCorrupt
			}
			e.tokens[j] = t
		}
		need := width * kvd * n
		if cap(buf) < need {
			buf = make([]byte, need)
		}
		buf = buf[:need]
		for li := 0; li < h.layers; li++ {
			for part := 0; part < 2; part++ {
				er.read(buf)
				if er.err != nil {
					return info, nil, ErrSnapshotCorrupt
				}
				out := make([]float32, kvd*n)
				decodeFloats(out, buf, full, compressed)
				if part == 0 {
					e.k[li] = out
				} else {
					e.v[li] = out
				}
			}
		}
		entries = append(entries, snapEntry{label: label, e: e})
		info.Labels = append(info.Labels, label)
		info.Tokens += n
	}
	if er.err != nil {
		return info, nil, ErrSnapshotCorrupt
	}
	if fr != nil {
		// drain flate's end of stream so the CRC covers every body byte
		if _, err := io.Copy(io.Discard, fr); err != nil {
			return info, nil, ErrSnapshotCorrupt
		}
		fr.Close()
	}
	sum := crc.Sum32()
	var want uint32
	if err := binary.Read(br, binary.LittleEndian, &want); err != nil || want != sum {
		return info, nil, ErrSnapshotCorrupt
	}
	return info, entries, nil
}

type errWriter struct {
	w   io.Writer
	err error
}

func (w *errWriter) write(b []byte) {
	if w.err == nil {
		_, w.err = w.w.Write(b)
	}
}
func (w *errWriter) le(v any) {
	if w.err == nil {
		w.err = binary.Write(w.w, binary.LittleEndian, v)
	}
}
func (w *errWriter) str(s string) {
	if len(s) > math.MaxUint16 {
		s = s[:math.MaxUint16]
	}
	w.le(uint16(len(s)))
	w.write([]byte(s))
}

type errReader struct {
	r   io.Reader
	err error
}

func (r *errReader) read(b []byte) {
	if r.err == nil {
		_, r.err = io.ReadFull(r.r, b)
	}
}
func (r *errReader) le(v any) {
	if r.err == nil {
		r.err = binary.Read(r.r, binary.LittleEndian, v)
	}
}
func (r *errReader) str() string {
	var l uint16
	r.le(&l)
	b := make([]byte, l)
	r.read(b)
	return string(b)
}

// crcReader hashes the bytes read through it. flate.NewReader wraps a reader without io.ByteReader in its own bufio
// reader and may read ahead past the end of the compressed stream, into the CRC, so it gets a ByteReader: flate then
// reads byte by byte and stops exactly at the end of its stream.
type crcReader struct {
	r interface {
		io.Reader
		io.ByteReader
	}
	h hash.Hash32
}

func (c *crcReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.h.Write(p[:n])
	return n, err
}

func (c *crcReader) ReadByte() (byte, error) {
	b, err := c.r.ReadByte()
	if err == nil {
		c.h.Write([]byte{b})
	}
	return b, err
}
