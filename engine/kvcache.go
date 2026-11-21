package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// The KV cache holds every position's attention keys and values, for each layer: hp.layers * kvHeads * headDim * 2
// numbers per token. At full precision that is 224 KB per token for Qwen3 0.6B and 1.7B, so a 32K context takes 7 GB.
// LoadOptions.KVCacheType stores it in a smaller format instead:
//
//	f32   4 bytes a number (the default): exactly what the model computed
//	f16   2 bytes: half the memory
//	q8_0  8-bit blocks of 32 (34 bytes per 32 numbers): about 27% of f32
//	q4_0  values in 4-bit blocks of 32 (18 bytes per 32 numbers), keys in q8_0: about 20% of f32. 4-bit keys
//	      more than double Qwen3's perplexity (attention scores are sensitive to key error; values aren't), so this
//	      type keeps keys at 8 bits, which costs almost nothing in quality.
//
// Layout, so attention runs on the matmul kernels instead of dequantizing:
//
//   - Keys: one row per position and KV head (headDim numbers, blocks along the head dimension), written as each
//     position is computed. Scores are row·query dot products, 4 queries at a time: f16 keys against the float
//     queries, q8_0 keys against the queries quantized to 8 bits (as llama.cpp does).
//   - Values: one row per channel (KV head × head dimension), positions along the row. The weighted sum of values is
//     then, for each channel, the dot product of its row with the 4 queries' attention weights. f16 rows are written a
//     position at a time. q8_0 and q4_0 rows are blocks of 32 positions, and a block can't be quantized until it's
//     full, so each layer keeps the newest block's values at full precision (stage) and quantizes them when the
//     block fills; attention reads the newest block from the stage, and the full blocks against the weights quantized
//     to 8 bits (q4_0 blocks unpacked to 8 bits first).

// KV cache types for LoadOptions.KVCacheType. The names match Ollama's OLLAMA_KV_CACHE_TYPE.
const (
	KVCacheF32  = "f32"
	KVCacheF16  = "f16"
	KVCacheQ8_0 = "q8_0"
	KVCacheQ4_0 = "q4_0"
)

// KVCacheTypes lists the KV cache types Load accepts.
var KVCacheTypes = []string{KVCacheF32, KVCacheF16, KVCacheQ8_0, KVCacheQ4_0}

const q40Bytes = 18 // d f16, qs[16]: element j in the low nibble of qs[j], element j+16 in the high nibble

func parseKVCacheType(s string) (uint32, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", KVCacheF32:
		return typeF32, nil
	case KVCacheF16:
		return typeF16, nil
	case KVCacheQ8_0:
		return typeQ8_0, nil
	case KVCacheQ4_0:
		return typeQ4_0, nil
	}
	return 0, fmt.Errorf("engine: unknown KV cache type %q (want one of %s)", s, strings.Join(KVCacheTypes, ", "))
}

func kvTypeName(t uint32) string {
	switch t {
	case typeF16:
		return KVCacheF16
	case typeQ8_0:
		return KVCacheQ8_0
	case typeQ4_0:
		return KVCacheQ4_0
	}
	return KVCacheF32
}

// kvBytes is the size of n numbers (n a multiple of 32 for the block types) in format t.
func kvBytes(t uint32, n int) int {
	switch t {
	case typeF16:
		return 2 * n
	case typeQ8_0:
		return n / 32 * q80Bytes
	case typeQ4_0:
		return n / 32 * q40Bytes
	}
	return 4 * n
}

// KVCacheBytes is the memory the KV cache takes per token of context, for the model's cache type.
func (m *Model) KVCacheBytes() int64 {
	kt, vt := kvTypes(m.kvTyp)
	kvd := m.hp.kvHeads * m.hp.headDim
	return int64(m.hp.layers) * int64(kvBytes(kt, kvd)+kvBytes(vt, kvd))
}

// f16Max keeps a stored value finite: an infinity would turn the zero weights past the context into NaNs.
const f16Max = 65504

// quantizeKVRow writes x (len a multiple of 32 for the block types) in format t.
func quantizeKVRow(t uint32, x []float32, out []byte) {
	switch t {
	case typeF16:
		for i, v := range x {
			binary.LittleEndian.PutUint16(out[2*i:], f32ToF16(max(-f16Max, min(f16Max, v))))
		}
	case typeQ8_0:
		for b := 0; b < len(x)/32; b++ {
			xb, ob := x[b*32:b*32+32], out[b*q80Bytes:(b+1)*q80Bytes]
			var amax float32
			for _, v := range xb {
				amax = max(amax, float32(math.Abs(float64(v))))
			}
			d := amax / 127
			id := float32(0)
			if d != 0 {
				id = 1 / d
			}
			binary.LittleEndian.PutUint16(ob, f32ToF16(d))
			for j, v := range xb {
				ob[2+j] = byte(int8(math.RoundToEven(float64(v * id))))
			}
		}
	case typeQ4_0:
		for b := 0; b < len(x)/32; b++ {
			xb, ob := x[b*32:b*32+32], out[b*q40Bytes:(b+1)*q40Bytes]
			var amax, mx float32 // the value with the largest magnitude, with its sign (ggml's q4_0)
			for _, v := range xb {
				if a := float32(math.Abs(float64(v))); a > amax {
					amax, mx = a, v
				}
			}
			d := mx / -8
			id := float32(0)
			if d != 0 {
				id = 1 / d
			}
			binary.LittleEndian.PutUint16(ob, f32ToF16(d))
			for j := 0; j < 16; j++ {
				q0 := max(0, min(15, int(xb[j]*id+8.5)))
				q1 := max(0, min(15, int(xb[j+16]*id+8.5)))
				ob[2+j] = byte(q0) | byte(q1)<<4
			}
		}
	default:
		panic("quantizeKVRow: unsupported type " + typeName(t))
	}
}

// dequantKVRow is quantizeKVRow's inverse.
func dequantKVRow(t uint32, row []byte, out []float32) {
	if t != typeQ4_0 {
		dequantRow(t, row, out)
		return
	}
	for b := 0; b < len(out)/32; b++ {
		blk := row[b*q40Bytes:]
		d := f16(blk)
		y := out[b*32:]
		for j := 0; j < 16; j++ {
			y[j] = d * float32(int(blk[2+j]&0xF)-8)
			y[j+16] = d * float32(int(blk[2+j]>>4)-8)
		}
	}
}

// unpackQ40 rewrites nb q4_0 blocks as q8_0 weight blocks (same scale, values -8..7), for the 8-bit kernels.
func unpackQ40(src, dst []byte, nb int) {
	const nib, bit3 = 0x0f0f0f0f0f0f0f0f, 0x0808080808080808
	// 8 nibbles at a time: v-8 for v in 0..15 is v^8 read as a signed 4-bit number, sign-extended to a byte by
	// setting the high nibble where bit 3 is set (bit3 bytes times 0x1e is 0xf0, no carries between bytes).
	ext := func(x uint64) uint64 { x ^= bit3; return x | (x&bit3)*0x1e }
	for b := 0; b < nb; b++ {
		s, d := src[b*q40Bytes:(b+1)*q40Bytes], dst[b*q80Bytes:(b+1)*q80Bytes]
		d[0], d[1] = s[0], s[1]
		w0, w1 := binary.LittleEndian.Uint64(s[2:]), binary.LittleEndian.Uint64(s[10:])
		binary.LittleEndian.PutUint64(d[2:], ext(w0&nib))
		binary.LittleEndian.PutUint64(d[10:], ext(w1&nib))
		binary.LittleEndian.PutUint64(d[18:], ext(w0>>4&nib))
		binary.LittleEndian.PutUint64(d[26:], ext(w1>>4&nib))
	}
}

// kvTypes returns the formats of keys and values for a cache type (q4_0 keeps keys at q8_0: see above).
func kvTypes(t uint32) (k, v uint32) {
	if t == typeQ4_0 {
		return typeQ8_0, typeQ4_0
	}
	return t, t
}

// compactKV is a state's compact cache.
type compactKV struct {
	ktyp   uint32   // format of keys
	vtyp   uint32   // format of values
	keyRow int      // bytes of one position's keys for one KV head
	k      [][]byte // per layer: [ctxP][kvHeads] key rows
	vRow   int      // bytes of one channel's values (f16: ctxP numbers; blocks: vBlocks blocks)
	v      [][]byte // per layer: [kvHeads*headDim] value rows
	// block types only: the newest block of values at full precision, per layer [kvd][32], and its index
	stage    [][]float32
	stageBlk []int
	scratch  []*kvScratch // per worker
}

// kvScratch is one worker's buffers for attention over a compact cache.
type kvScratch struct {
	q    []byte    // 4 rows (queries, or attention weights) quantized to 8 bits: q80x4 layout, or 4 rows of 36-byte blocks
	qr   []byte    // one row's 36-byte blocks
	row  []byte    // a q4_0 row unpacked to q8_0
	tail []float32 // the 4 rows' weights over the stage block
	dot4 [4]float32
}

func (m *Model) allocCompact(s *state) {
	hp := m.hp
	hd, kvh := hp.headDim, hp.kvHeads
	kvd := kvh * hd
	kt, vt := kvTypes(m.kvTyp)
	c := &compactKV{ktyp: kt, vtyp: vt, keyRow: kvBytes(kt, hd)}
	vn := s.ctxP // values per channel row
	if c.vtyp != typeF16 {
		vn = (s.ctxP + 31) &^ 31
		c.stage = make([][]float32, hp.layers)
		c.stageBlk = make([]int, hp.layers)
	}
	c.vRow = kvBytes(c.vtyp, vn)
	c.k = make([][]byte, hp.layers)
	c.v = make([][]byte, hp.layers)
	for i := range c.k {
		c.k[i] = make([]byte, s.ctxP*kvh*c.keyRow)
		c.v[i] = make([]byte, kvd*c.vRow)
		if c.stage != nil {
			c.stage[i] = make([]float32, kvd*32)
		}
	}
	rows := max(vn, hd) // a weight row covers up to vn positions; a query row hd numbers
	c.scratch = make([]*kvScratch, m.pool.n)
	for i := range c.scratch {
		c.scratch[i] = &kvScratch{
			q:    make([]byte, max(rows/32*q80x4Bytes, 4*rows/32*q80aBytes)),
			qr:   make([]byte, rows/32*q80aBytes),
			row:  make([]byte, rows/32*q80Bytes),
			tail: make([]float32, 4*32),
		}
	}
	s.kv = c
}

// compactBytes is what allocCompact allocates for ctxP positions (a multiple of 32).
func (m *Model) compactBytes(ctxP int) int64 {
	hp := m.hp
	hd, kvd := hp.headDim, hp.kvHeads*hp.headDim
	kt, vt := kvTypes(m.kvTyp)
	per := int64(ctxP*hp.kvHeads*kvBytes(kt, hd)) + int64(kvd*kvBytes(vt, ctxP)) // keys, values
	if vt != typeF16 {
		per += int64(kvd * 32 * 4) // stage
	}
	rows := max(ctxP, hd)
	scratch := int64(max(rows/32*q80x4Bytes, 4*rows/32*q80aBytes) + rows/32*q80aBytes + rows/32*q80Bytes + 4*32*4)
	return int64(hp.layers)*per + int64(m.pool.n)*scratch
}

// storeKey writes one position's keys (kvd numbers) at position p of layer li.
func (m *Model) storeKey(s *state, li, p int, k []float32) {
	c, hd := s.kv, m.hp.headDim
	for h := 0; h < m.hp.kvHeads; h++ {
		off := (p*m.hp.kvHeads + h) * c.keyRow
		quantizeKVRow(c.ktyp, k[h*hd:(h+1)*hd], c.k[li][off:off+c.keyRow])
	}
}

// storeValue writes one position's values (kvd numbers) at position p of layer li. Positions are written in order
// from 0 within a call (block types need that: the stage holds one block).
func (m *Model) storeValue(s *state, li, p int, v []float32) {
	c := s.kv
	if c.vtyp == typeF16 {
		row := c.v[li]
		for ch, x := range v {
			binary.LittleEndian.PutUint16(row[ch*c.vRow+2*p:], f32ToF16(max(-f16Max, min(f16Max, x))))
		}
		return
	}
	st, b, j := c.stage[li], p/32, p%32
	if b != c.stageBlk[li] || j == 0 {
		if j != 0 { // not in order (doesn't happen today): start from what the block holds
			tmp := make([]float32, 32)
			for ch := range v {
				dequantKVRow(c.vtyp, c.v[li][ch*c.vRow+b*kvBytes(c.vtyp, 32):][:kvBytes(c.vtyp, 32)], tmp)
				copy(st[ch*32:ch*32+32], tmp)
			}
		}
		c.stageBlk[li] = b
	}
	for ch, x := range v {
		st[ch*32+j] = x
	}
	if j == 31 { // the block is full: quantize it
		bb := kvBytes(c.vtyp, 32)
		for ch := range v {
			quantizeKVRow(c.vtyp, st[ch*32:ch*32+32], c.v[li][ch*c.vRow+b*bb:][:bb])
		}
	}
}

// dot4Q8 computes the dots of an 8-bit row (nb q8_0 weight blocks) with the 4 rows quantized in sw.q.
func dot4Q8(row []byte, nb int, sw *kvScratch) {
	if dotQ80x4 != nil {
		dotQ80x4(row, sw.q, nb, &sw.dot4)
		return
	}
	qs := len(sw.q) / 4
	for i := 0; i < 4; i++ {
		sw.dot4[i] = dotQ80(row, sw.q[i*qs:], nb)
	}
}

// quantize4 quantizes the first 32*nb numbers of the 4 rows x[i*stride:] into sw.q, in the layout dot4Q8 reads.
func quantize4(x []float32, stride, nb int, sw *kvScratch) {
	qs := len(sw.q) / 4
	for i := 0; i < 4; i++ {
		r := x[i*stride : i*stride+32*nb]
		if dotQ80x4 != nil {
			quantizeQ8_0(r, sw.qr)
			interleaveQ8_0(sw.qr, sw.q, i, nb)
		} else {
			quantizeQ8_0(r, sw.q[i*qs:])
		}
	}
}

// scoresQ computes the 4 queries in qb against keys [0, n) of KV head kvh, into sc rows (stride ctxP), like scores4.
func (m *Model) scoresQ(s *state, sw *kvScratch, li, kvh, n int, qb, sc []float32) {
	c, hd, kvhN, ctxP := s.kv, m.hp.headDim, m.hp.kvHeads, s.ctxP
	kc := c.k[li]
	row := func(p int) []byte { off := (p*kvhN + kvh) * c.keyRow; return kc[off : off+c.keyRow] }
	put := func(p int) {
		sc[p], sc[ctxP+p], sc[2*ctxP+p], sc[3*ctxP+p] = sw.dot4[0], sw.dot4[1], sw.dot4[2], sw.dot4[3]
	}
	switch c.ktyp {
	case typeF16:
		if dotF16x4 != nil {
			for p := 0; p < n; p++ {
				dotF16x4(row(p), qb, hd, &sw.dot4)
				put(p)
			}
			return
		}
		for p := 0; p < n; p++ {
			r := row(p)
			for i := 0; i < 4; i++ {
				sc[i*ctxP+p] = dotF16(r, qb[i*hd:(i+1)*hd])
			}
		}
	case typeQ8_0:
		quantize4(qb, hd, hd/32, sw)
		for p := 0; p < n; p++ {
			dot4Q8(row(p), hd/32, sw)
			put(p)
		}
	default:
		panic("scoresQ: unsupported key type " + typeName(c.ktyp)) // q4_0 caches keep keys at q8_0 (kvTypes)
	}
}

// wsumQ computes the 4 weighted sums of values [0, n) of KV head kvh (weights in sc rows, stride ctxP) into ob,
// like wsum4. Weights in [n, 16*ceil(n/16)) may be overwritten with zeros.
func (m *Model) wsumQ(s *state, sw *kvScratch, li, kvh, n int, sc, ob []float32) {
	c, hd, ctxP := s.kv, m.hp.headDim, s.ctxP
	vr := c.v[li]
	if c.vtyp == typeF16 {
		n16 := (n + 15) &^ 15
		for i := 0; i < 4; i++ {
			clear(sc[i*ctxP+n : i*ctxP+n16])
		}
		for d := 0; d < hd; d++ {
			ch := kvh*hd + d
			r := vr[ch*c.vRow : ch*c.vRow+2*n16]
			if dotF16x4 != nil {
				dotF16x4(r, sc, ctxP, &sw.dot4)
			} else {
				for i := 0; i < 4; i++ {
					sw.dot4[i] = dotF16(r, sc[i*ctxP:i*ctxP+n16])
				}
			}
			ob[d], ob[hd+d], ob[2*hd+d], ob[3*hd+d] = sw.dot4[0], sw.dot4[1], sw.dot4[2], sw.dot4[3]
		}
		return
	}
	// full blocks below the stage: 8-bit weights against the quantized rows; the stage's positions in floats
	sb := c.stageBlk[li]
	nb := min(sb, (n+31)/32) // blocks read from the quantized rows
	t0, tn := 32*sb, max(0, n-32*sb)
	if nb > 0 {
		for i := 0; i < 4; i++ { // the last block read may run past n: zero weights there
			clear(sc[i*ctxP+min(n, 32*nb) : i*ctxP+32*nb])
		}
		quantize4(sc, ctxP, nb, sw)
	}
	for i := 0; i < 4; i++ {
		copy(sw.tail[i*32:i*32+tn], sc[i*ctxP+t0:i*ctxP+t0+tn])
	}
	st := c.stage[li]
	bb := kvBytes(c.vtyp, 32)
	for d := 0; d < hd; d++ {
		ch := kvh*hd + d
		var o [4]float32
		if nb > 0 {
			r := vr[ch*c.vRow : ch*c.vRow+nb*bb]
			if c.vtyp == typeQ4_0 {
				unpackQ40(r, sw.row, nb)
				r = sw.row
			}
			dot4Q8(r, nb, sw)
			o = sw.dot4
		}
		if tn > 0 {
			sv := st[ch*32 : ch*32+tn]
			for i := 0; i < 4; i++ {
				o[i] += dotF32(sv, sw.tail[i*32:i*32+tn])
			}
		}
		ob[d], ob[hd+d], ob[2*hd+d], ob[3*hd+d] = o[0], o[1], o[2], o[3]
	}
}

// saveKV copies the first n positions of layer li out of s, as full-precision keys [kvd][n] (each channel's positions
// in a row) and values [n][kvd]: the prompt cache's and threads' form.
func (m *Model) saveKV(s *state, li, n int) (k, v []float32) {
	hd, kvh := m.hp.headDim, m.hp.kvHeads
	kvd := kvh * hd
	k = make([]float32, kvd*n)
	if s.kv == nil {
		for i := 0; i < kvd; i++ {
			copy(k[i*n:(i+1)*n], s.kc[li][i*s.ctxP:i*s.ctxP+n])
		}
		return k, append([]float32(nil), s.vc[li][:n*kvd]...)
	}
	c := s.kv
	v = make([]float32, n*kvd)
	row := make([]float32, max(hd, (n+31)&^31))
	for p := 0; p < n; p++ {
		for h := 0; h < kvh; h++ {
			off := (p*kvh + h) * c.keyRow
			dequantKVRow(c.ktyp, c.k[li][off:off+c.keyRow], row[:hd])
			for d, x := range row[:hd] {
				k[(h*hd+d)*n+p] = x
			}
		}
	}
	if c.vtyp == typeF16 {
		for ch := 0; ch < kvd; ch++ {
			dequantRow(typeF16, c.v[li][ch*c.vRow:], row[:n])
			for p, x := range row[:n] {
				v[p*kvd+ch] = x
			}
		}
		return k, v
	}
	sb := c.stageBlk[li]
	nq := min(n, 32*sb) // positions in quantized blocks
	nqb := (nq + 31) / 32
	for ch := 0; ch < kvd; ch++ {
		dequantKVRow(c.vtyp, c.v[li][ch*c.vRow:ch*c.vRow+kvBytes(c.vtyp, 32*nqb)], row[:32*nqb])
		for p := 0; p < nq; p++ {
			v[p*kvd+ch] = row[p]
		}
		for p := nq; p < n; p++ {
			v[p*kvd+ch] = c.stage[li][ch*32+p-32*sb]
		}
	}
	return k, v
}

// loadKV writes the first n positions of a saved layer (keys [kvd][en], values [en][kvd]) into layer li of s.
func (m *Model) loadKV(s *state, li int, k, v []float32, en, n int) {
	hd, kvh := m.hp.headDim, m.hp.kvHeads
	kvd := kvh * hd
	if s.kv == nil {
		for i := 0; i < kvd; i++ {
			copy(s.kc[li][i*s.ctxP:i*s.ctxP+n], k[i*en:i*en+n])
		}
		copy(s.vc[li][:n*kvd], v[:n*kvd])
		return
	}
	kp := make([]float32, kvd)
	for p := 0; p < n; p++ {
		for i := range kp {
			kp[i] = k[i*en+p]
		}
		m.storeKey(s, li, p, kp)
		m.storeValue(s, li, p, v[p*kvd:(p+1)*kvd])
	}
}
