package engine

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"unicode/utf8"
)

// Archive format (compress13k spec v1, extension .13k): 0xA1 (magic and version), the model id, the token count as
// an unsigned LEB128 varint, then the arithmetic-coded body. Inputs longer than intSeg tokens are coded in segments
// of intSeg tokens, each a fresh context; the coder runs on across segments.
const (
	intMagicV1 = 0xA1
	intSeg     = 32768
	intTop     = 1<<32 - 1
	intHalf    = 1 << 31
	intQtr     = 1 << 30
)

//go:embed intdata/models-v1.json
var intManifestJSON []byte

// IntModelInfo is one entry of the integer codec's model manifest.
type IntModelInfo struct {
	ID      int    `json:"id"`
	Name    string `json:"name"`
	Profile string `json:"profile"` // good, better, best
	Source  struct {
		Repo     string   `json:"repo"`
		Revision string   `json:"revision"`
		Files    []string `json:"files"`
	} `json:"source"`
	Bundle struct {
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
		Bytes  int64  `json:"bytes"`
	} `json:"bundle"`
	URLs []string `json:"urls"`
}

// IntManifest lists the converted models an archive can name.
func IntManifest() []IntModelInfo {
	var d struct {
		Models []IntModelInfo `json:"models"`
	}
	if err := json.Unmarshal(intManifestJSON, &d); err != nil {
		panic(err)
	}
	return d.Models
}

// IntModelByID returns the manifest entry for an archive's model id.
func IntModelByID(id int) (IntModelInfo, bool) {
	for _, mi := range IntManifest() {
		if mi.ID == id {
			return mi, true
		}
	}
	return IntModelInfo{}, false
}

// IntModelByProfile returns the manifest entry for a profile (good, better, best).
func IntModelByProfile(p string) (IntModelInfo, bool) {
	for _, mi := range IntManifest() {
		if mi.Profile == p {
			return mi, true
		}
	}
	return IntModelInfo{}, false
}

// ID is the model's manifest id, found by its file name (0 if it isn't in the manifest).
func (m *IntModel) ID() int {
	for _, mi := range IntManifest() {
		if mi.Bundle.File == filepath.Base(m.path) {
			return mi.ID
		}
	}
	return 0
}

// IntArchiveInfo reads an archive's header: the model id and the token count.
func IntArchiveInfo(a []byte) (id, n int, body []byte, err error) {
	if len(a) < 3 || a[0] != intMagicV1 {
		return 0, 0, nil, errors.New("intcodec: not a v1 archive")
	}
	var v uint64
	i, shift := 2, 0
	for {
		if i >= len(a) || shift > 56 {
			return 0, 0, nil, errors.New("intcodec: bad token count")
		}
		b := a[i]
		v |= uint64(b&0x7f) << shift
		shift += 7
		i++
		if b&0x80 == 0 {
			break
		}
	}
	return int(a[1]), int(v), a[i:], nil
}

func intHeader(id, n int) []byte {
	out := []byte{intMagicV1, byte(id)}
	for {
		b := byte(n & 0x7f)
		n >>= 7
		if n != 0 {
			out = append(out, b|0x80)
		} else {
			return append(out, b)
		}
	}
}

// ---- Witten-Neal-Cleary coder (32-bit)

type intEncoder struct {
	low, high, pending uint64
	out                []byte
	acc, nb            uint8
}

func newIntEncoder() *intEncoder { return &intEncoder{high: intTop} }

func (e *intEncoder) bit(b uint8) {
	e.acc = e.acc<<1 | b
	e.nb++
	if e.nb == 8 {
		e.out = append(e.out, e.acc)
		e.acc, e.nb = 0, 0
	}
}

func (e *intEncoder) emit(b uint8) {
	e.bit(b)
	for ; e.pending > 0; e.pending-- {
		e.bit(1 - b)
	}
}

func (e *intEncoder) code(total, hi, lo uint64) {
	rng := e.high - e.low + 1
	e.high = e.low + rng*hi/total - 1
	e.low = e.low + rng*lo/total
	for {
		switch {
		case e.high < intHalf:
			e.emit(0)
		case e.low >= intHalf:
			e.emit(1)
			e.low -= intHalf
			e.high -= intHalf
		case e.low >= intQtr && e.high < intHalf+intQtr:
			e.pending++
			e.low -= intQtr
			e.high -= intQtr
		default:
			return
		}
		e.low, e.high = 2*e.low, 2*e.high+1
	}
}

func (e *intEncoder) finish() []byte {
	e.pending++
	if e.low < intQtr {
		e.emit(0)
	} else {
		e.emit(1)
	}
	for e.nb != 0 {
		e.bit(0)
	}
	return e.out
}

type intDecoder struct {
	data             []byte
	pos              int
	low, high, value uint64
}

func newIntDecoder(data []byte) *intDecoder {
	d := &intDecoder{data: data, high: intTop}
	for i := 0; i < 32; i++ {
		d.value = 2*d.value + d.bit()
	}
	return d
}

func (d *intDecoder) bit() uint64 {
	i := d.pos
	d.pos++
	if i>>3 >= len(d.data) {
		return 0
	}
	return uint64(d.data[i>>3]>>(7-uint(i&7))) & 1
}

// next decodes one token from a cumulative frequency table.
func (d *intDecoder) next(cum []int64) int {
	total := uint64(cum[len(cum)-1])
	rng := d.high - d.low + 1
	target := int64(((d.value-d.low+1)*total - 1) / rng)
	t := sort.Search(len(cum), func(i int) bool { return cum[i] > target })
	hi, lo := uint64(cum[t]), uint64(0)
	if t > 0 {
		lo = uint64(cum[t-1])
	}
	d.high = d.low + rng*hi/total - 1
	d.low = d.low + rng*lo/total
	for {
		switch {
		case d.high < intHalf:
		case d.low >= intHalf:
			d.value -= intHalf
			d.low -= intHalf
			d.high -= intHalf
		case d.low >= intQtr && d.high < intHalf+intQtr:
			d.value -= intQtr
			d.low -= intQtr
			d.high -= intQtr
		default:
			return t
		}
		d.low, d.high, d.value = 2*d.low, 2*d.high+1, 2*d.value+d.bit()
	}
}

// ---- text

// encodeText tokenizes text for packing: the standard tokenization if it decodes back to the exact bytes,
// else one token per byte (always exact). Any tokenization that round-trips is allowed by the spec.
func (m *IntModel) encodeText(text []byte) []int {
	if utf8.Valid(text) {
		ids := m.tok.encode(string(text))
		if bytes.Equal(m.decodeTokens(ids), text) {
			return ids
		}
	}
	ids := make([]int, len(text))
	for i, b := range text {
		ids[i] = m.tok.vocab[string(m.tok.byteEnc[b])]
	}
	return ids
}

func (m *IntModel) decodeTokens(ids []int) []byte {
	var out []byte
	for _, id := range ids {
		out = append(out, m.tok.tokenBytes(id)...)
	}
	return out
}

// ---- pack / unpack

// IntProgress, if set, is called after each token with the count done and the total.
type IntProgress func(done, total int)

// Pack compresses text into a v1 archive. The model must be in the manifest (its id goes in the header).
func (m *IntModel) Pack(text []byte, progress IntProgress) ([]byte, error) {
	id := m.ID()
	if id == 0 {
		return nil, fmt.Errorf("intcodec: %s is not in the model manifest", filepath.Base(m.path))
	}
	ids := m.encodeText(text)
	n := len(ids)
	enc := newIntEncoder()
	st := m.newState(min(max(n, 1), intSeg))
	for s0 := 0; s0 < n; s0 += intSeg {
		st.reset()
		prev := m.bos
		for k := s0; k < min(s0+intSeg, n); k++ {
			st.step(prev)
			t := ids[k]
			lo := int64(0)
			if t > 0 {
				lo = st.cum[t-1]
			}
			enc.code(uint64(st.cum[m.V-1]), uint64(st.cum[t]), uint64(lo))
			prev = t
			if progress != nil {
				progress(k+1, n)
			}
		}
	}
	return append(intHeader(id, n), enc.finish()...), nil
}

// Unpack decompresses a v1 archive made with this model.
func (m *IntModel) Unpack(a []byte, progress IntProgress) ([]byte, error) {
	id, n, body, err := IntArchiveInfo(a)
	if err != nil {
		return nil, err
	}
	if mid := m.ID(); mid != 0 && mid != id {
		return nil, fmt.Errorf("intcodec: archive needs model %d, this is model %d", id, mid)
	}
	dec := newIntDecoder(body)
	ids := make([]int, 0, n)
	st := m.newState(min(max(n, 1), intSeg))
	for s0 := 0; s0 < n; s0 += intSeg {
		st.reset()
		prev := m.bos
		for k := s0; k < min(s0+intSeg, n); k++ {
			st.step(prev)
			prev = dec.next(st.cum)
			ids = append(ids, prev)
			if progress != nil {
				progress(k+1, n)
			}
		}
	}
	return m.decodeTokens(ids), nil
}
