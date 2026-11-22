package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

// whisperFile is a Whisper model in whisper.cpp's ggml format (the ggml-*.bin files): a header of hyperparameters,
// the mel filterbank, the vocabulary, then tensors, none of them aligned.
type whisperFile struct {
	hp      whisperHParams
	filters []float32 // nMels rows of nFFT/2+1
	nFFT    int
	vocab   [][]byte // token bytes in id order (the BPE ranks)
	tensors map[string]*tensorInfo
	mapping []byte
	size    int64
}

type whisperHParams struct {
	vocab, audioCtx, audioState, audioHead, audioLayer int
	textCtx, textState, textHead, textLayer, mels      int
	ftype                                              int
}

const ggmlMagic = 0x67676d6c // "ggml"

func openWhisperFile(path string) (*whisperFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	mapping, data, err := mmapRange(f, 0, st.Size())
	if err != nil {
		return nil, err
	}
	w, err := parseWhisperFile(data)
	if err != nil {
		munmap(mapping)
		return nil, err
	}
	w.mapping, w.size = mapping, st.Size()
	return w, nil
}

func parseWhisperFile(b []byte) (*whisperFile, error) {
	off := 0
	bad := errors.New("engine: truncated or malformed whisper ggml file")
	i32 := func() (int, bool) {
		if off+4 > len(b) {
			return 0, false
		}
		v := int(int32(binary.LittleEndian.Uint32(b[off:])))
		off += 4
		return v, true
	}
	if m, ok := i32(); !ok || uint32(m) != ggmlMagic {
		return nil, errors.New("engine: not a whisper.cpp ggml model file")
	}
	w := &whisperFile{tensors: map[string]*tensorInfo{}}
	hp := []*int{&w.hp.vocab, &w.hp.audioCtx, &w.hp.audioState, &w.hp.audioHead, &w.hp.audioLayer,
		&w.hp.textCtx, &w.hp.textState, &w.hp.textHead, &w.hp.textLayer, &w.hp.mels, &w.hp.ftype}
	for _, p := range hp {
		v, ok := i32()
		if !ok || v < 0 {
			return nil, bad
		}
		*p = v
	}
	w.hp.ftype %= 1000 // GGML_QNT_VERSION_FACTOR
	nMel, ok1 := i32()
	nFFT, ok2 := i32()
	if !ok1 || !ok2 || nMel <= 0 || nFFT <= 0 || off+4*nMel*nFFT > len(b) {
		return nil, bad
	}
	w.nFFT = nFFT
	w.filters = make([]float32, nMel*nFFT)
	for i := range w.filters {
		w.filters[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[off+4*i:]))
	}
	off += 4 * nMel * nFFT
	nv, ok := i32()
	if !ok || nv < 0 || nv > 1<<20 {
		return nil, bad
	}
	w.vocab = make([][]byte, nv)
	for i := range w.vocab {
		n, ok := i32()
		if !ok || n < 0 || off+n > len(b) {
			return nil, bad
		}
		w.vocab[i] = b[off : off+n]
		off += n
	}
	for off < len(b) {
		nd, ok1 := i32()
		nl, ok2 := i32()
		tt, ok3 := i32()
		if !ok1 || !ok2 || !ok3 || nd < 1 || nd > 4 || nl <= 0 || nl > 256 {
			return nil, bad
		}
		t := &tensorInfo{Type: uint32(tt)}
		ne := 1
		for i := 0; i < nd; i++ {
			d, ok := i32()
			if !ok || d <= 0 {
				return nil, bad
			}
			t.Dims = append(t.Dims, uint64(d))
			ne *= d
		}
		if off+nl > len(b) {
			return nil, bad
		}
		t.Name = string(b[off : off+nl])
		off += nl
		epb, bpb, ok := blockInfo(t.Type)
		if !ok {
			return nil, fmt.Errorf("engine: whisper tensor %s has unsupported type %s", t.Name, typeName(t.Type))
		}
		size := ne / epb * bpb
		if off+size > len(b) {
			return nil, bad
		}
		t.Offset = uint64(off)
		t.Data = b[off : off+size]
		off += size
		w.tensors[t.Name] = t
	}
	return w, nil
}
