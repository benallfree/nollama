package engine

import (
	"container/heap"
	"errors"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// tokenizer is a GPT-2 style byte-level BPE with the Qwen2 pre-tokenizer.
type tokenizer struct {
	tokens   []string
	types    []int32
	vocab    map[string]int
	ranks    map[string]int // "a b" -> merge rank
	byteEnc  [256]rune
	byteDec  map[rune]byte
	specials []string // longest first
	special  map[string]int
	eos      map[int]bool
	cache    sync.Map // word -> []int
}

func newTokenizer(g *ggufFile) (*tokenizer, error) {
	if m := g.getString("tokenizer.ggml.model"); m != "gpt2" {
		return nil, errors.New("engine: unsupported tokenizer model " + m)
	}
	toks, _ := g.KV["tokenizer.ggml.tokens"].([]string)
	merges, _ := g.KV["tokenizer.ggml.merges"].([]string)
	if len(toks) == 0 || len(merges) == 0 {
		return nil, errors.New("engine: missing tokenizer tokens/merges")
	}
	t := &tokenizer{tokens: toks, vocab: make(map[string]int, len(toks)), ranks: make(map[string]int, len(merges)),
		byteDec: map[rune]byte{}, special: map[string]int{}, eos: map[int]bool{}}
	t.types, _ = g.KV["tokenizer.ggml.token_type"].([]int32)
	for i, s := range toks {
		t.vocab[s] = i
	}
	for i, s := range merges {
		t.ranks[s] = i
	}
	// GPT-2 bytes_to_unicode
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			t.byteEnc[b] = rune(b)
		} else {
			t.byteEnc[b] = rune(256 + n)
			n++
		}
		t.byteDec[t.byteEnc[b]] = byte(b)
	}
	for i, ty := range t.types {
		if ty == 3 || ty == 4 { // control, user-defined
			t.special[toks[i]] = i
			t.specials = append(t.specials, toks[i])
		}
	}
	sort.Slice(t.specials, func(a, b int) bool { return len(t.specials[a]) > len(t.specials[b]) })
	for _, k := range []string{"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"} {
		if v, ok := g.getInt(k); ok {
			t.eos[v] = true
		}
	}
	for _, s := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := t.vocab[s]; ok {
			t.eos[id] = true
		}
	}
	return t, nil
}

func (t *tokenizer) isSpecial(id int) bool {
	return id < len(t.types) && (t.types[id] == 3 || t.types[id] == 4)
}

// isControl is a control token (<|im_start|>), as opposed to a user-defined one (<tool_call>, <think>) that stands for
// its text.
func (t *tokenizer) isControl(id int) bool {
	return id < len(t.types) && t.types[id] == 3
}

// encode tokenizes text, matching special tokens as whole tokens first.
func (t *tokenizer) encode(s string) []int {
	var out []int
	for len(s) > 0 {
		// find the earliest special token occurrence
		best, bestAt := "", -1
		if strings.Contains(s, "<") {
			for _, sp := range t.specials {
				if i := strings.Index(s, sp); i >= 0 && (bestAt < 0 || i < bestAt || (i == bestAt && len(sp) > len(best))) {
					best, bestAt = sp, i
				}
			}
		}
		if bestAt < 0 {
			out = t.encodeText(s, out)
			break
		}
		out = t.encodeText(s[:bestAt], out)
		out = append(out, t.special[best])
		s = s[bestAt+len(best):]
	}
	return out
}

func (t *tokenizer) encodeText(s string, out []int) []int {
	for _, w := range preTokenize(s) {
		out = append(out, t.bpeWord(w)...)
	}
	return out
}

func (t *tokenizer) bpeWord(w string) []int {
	if v, ok := t.cache.Load(w); ok {
		return v.([]int)
	}
	syms := t.bpeMerge(w)
	ids := make([]int, 0, len(syms))
	for _, sy := range syms {
		if id, ok := t.vocab[sy]; ok {
			ids = append(ids, id)
		} else {
			for _, r := range sy { // fall back to single byte tokens
				if id, ok := t.vocab[string(r)]; ok {
					ids = append(ids, id)
				}
			}
		}
	}
	if len(w) < 64 {
		t.cache.Store(w, ids)
	}
	return ids
}

// bpeMerge applies the merges to a word's bytes: repeatedly the adjacent pair with the lowest merge rank (the leftmost
// of equal ones) becomes one symbol. A heap of candidate pairs over a linked list of symbols makes it O(n log n) in
// the word's length (a rescan per merge would be O(n^2): hours for a megabyte with no spaces).
func (t *tokenizer) bpeMerge(w string) []string {
	n := len(w)
	type sym struct {
		s          string
		prev, next int // -1 at the ends
		ver        int // bumped when s grows; a pair records its symbols' versions
		dead       bool
	}
	syms := make([]sym, n)
	for i := 0; i < n; i++ {
		syms[i] = sym{s: string(t.byteEnc[w[i]]), prev: i - 1, next: i + 1}
	}
	if n > 0 {
		syms[n-1].next = -1
	}
	h := &bpeHeap{}
	push := func(i, j int) {
		if i < 0 || j < 0 {
			return
		}
		if r, ok := t.ranks[syms[i].s+" "+syms[j].s]; ok {
			heap.Push(h, bpePair{rank: r, i: i, j: j, vi: syms[i].ver, vj: syms[j].ver})
		}
	}
	for i := 0; i+1 < n; i++ {
		push(i, i+1)
	}
	for h.Len() > 0 {
		p := heap.Pop(h).(bpePair)
		a, b := &syms[p.i], &syms[p.j]
		if a.dead || b.dead || a.next != p.j || a.ver != p.vi || b.ver != p.vj {
			continue // stale: a symbol merged since
		}
		a.s += b.s
		a.ver++
		a.next = b.next
		b.dead = true
		if b.next >= 0 {
			syms[b.next].prev = p.i
		}
		push(a.prev, p.i)
		push(p.i, a.next)
	}
	var out []string
	for i := 0; i >= 0 && i < n; i = syms[i].next {
		out = append(out, syms[i].s)
	}
	return out
}

type bpePair struct{ rank, i, j, vi, vj int }

// bpeHeap orders pairs by merge rank, then position (leftmost first).
type bpeHeap []bpePair

func (h bpeHeap) Len() int { return len(h) }
func (h bpeHeap) Less(a, b int) bool {
	return h[a].rank < h[b].rank || (h[a].rank == h[b].rank && h[a].i < h[b].i)
}
func (h bpeHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h *bpeHeap) Push(x any)   { *h = append(*h, x.(bpePair)) }
func (h *bpeHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// tokenBytes returns the raw bytes a token stands for.
func (t *tokenizer) tokenBytes(id int) []byte {
	if id < 0 || id >= len(t.tokens) {
		return nil
	}
	s := t.tokens[id]
	if t.isSpecial(id) {
		return []byte(s)
	}
	b := make([]byte, 0, len(s))
	for _, r := range s {
		if c, ok := t.byteDec[r]; ok {
			b = append(b, c)
		} else {
			b = utf8.AppendRune(b, r)
		}
	}
	return b
}

// preTokenize splits s the way the Qwen2 regex does:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
func preTokenize(s string) []string {
	rs := []rune(s)
	var out []string
	isL := func(r rune) bool { return unicode.IsLetter(r) }
	isN := func(r rune) bool { return unicode.IsNumber(r) }
	isS := func(r rune) bool { return unicode.IsSpace(r) }
	isNL := func(r rune) bool { return r == '\r' || r == '\n' }
	n := len(rs)
	at := func(i int) rune {
		if i < n {
			return rs[i]
		}
		return -1
	}
	for i := 0; i < n; {
		c := rs[i]
		end := -1
		// 1. contractions
		if c == '\'' && i+1 < n {
			c1, c2 := unicode.ToLower(at(i+1)), unicode.ToLower(at(i+2))
			switch {
			case c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd':
				end = i + 2
			case (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'):
				end = i + 3
			}
		}
		// 2. [^\r\n\p{L}\p{N}]?\p{L}+
		if end < 0 {
			j := i
			if !isNL(c) && !isL(c) && !isN(c) && i+1 < n && isL(rs[i+1]) {
				j = i + 1
			}
			if j < n && isL(rs[j]) {
				for j < n && isL(rs[j]) {
					j++
				}
				end = j
			}
		}
		// 3. \p{N}
		if end < 0 && isN(c) {
			end = i + 1
		}
		// 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
		if end < 0 {
			j := i
			if c == ' ' {
				j++
			}
			k := j
			for k < n && !isS(rs[k]) && !isL(rs[k]) && !isN(rs[k]) {
				k++
			}
			if k > j {
				for k < n && isNL(rs[k]) {
					k++
				}
				end = k
			}
		}
		if end < 0 && isS(c) {
			k := i
			for k < n && isS(rs[k]) {
				k++
			}
			// 5. \s*[\r\n]+ : up to the last newline in the whitespace run
			last := -1
			for j := i; j < k; j++ {
				if isNL(rs[j]) {
					last = j
				}
			}
			switch {
			case last >= 0:
				end = last + 1
			case k == n: // 6. \s+(?!\S) at end of text
				end = k
			case k-i >= 2: // 6. leave the last space for the next word
				end = k - 1
			default: // 7. \s+
				end = k
			}
		}
		if end < 0 {
			end = i + 1
		}
		out = append(out, string(rs[i:end]))
		i = end
	}
	return out
}
