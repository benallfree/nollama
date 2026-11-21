package engine

import (
	"sync"
	"time"
)

// The prompt cache keeps the attention keys and values the model computed for the start of recent prompts, so a prompt
// that begins the same way (a chat's preamble, the turns so far) skips reading that part again: the cache's positions
// are copied into the new call's state and prefill resumes after them. It's the same idea as the prompt caching big
// providers offer. Positions don't move, so a reused prefix is exactly what the model would have computed.
//
// Keys and values take hp.layers * kvHeads * headDim * 2 floats per token (about 230 KB for Qwen3 0.6B to 4B), so the
// cache has a memory budget, and an entry keeps only as much of the start of its prompt as fits in it, which is where
// prompts repeat.

// prefixMin is the shortest prefix worth reusing: copying is cheaper than reading, but not free.
const prefixMin = 32

type prefixEntry struct {
	scope  string // whose prompt it was (GenerateOptions.CacheScope): entries are only reused in the same scope
	tokens []int
	k, v   [][]float32     // per layer: k [kvd][n] (each channel's positions in a row), v [n*kvd]
	seq    uint64          // last use, for eviction
	used   time.Time       // last use, for evicting across models when a shared memory budget is short
	pins   map[string]bool // Warm's pins: a pinned entry is never evicted
}

type prefixCache struct {
	mu      sync.Mutex
	budget  int64 // bytes; <= 0: off
	used    int64
	seq     uint64
	entries []*prefixEntry
	hits    uint64
}

func (m *Model) tokenBytes() int64 {
	return int64(m.hp.layers) * int64(m.hp.kvHeads*m.hp.headDim) * 2 * 4
}

// lookup finds the cached entry sharing the longest prefix with ids (at least prefixMin tokens, and always leaving
// the last token of ids to compute, since generation needs its logits). n is how many tokens it covers.
func (m *Model) lookupPrefix(scope string, ids []int) (e *prefixEntry, n int) {
	c := &m.prefix
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.budget <= 0 {
		return nil, 0
	}
	for _, cand := range c.entries {
		if cand.scope != scope {
			continue
		}
		l := 0
		for l < len(cand.tokens) && l < len(ids)-1 && cand.tokens[l] == ids[l] {
			l++
		}
		if l > n {
			e, n = cand, l
		}
	}
	if n < prefixMin {
		return nil, 0
	}
	c.seq++
	e.seq, e.used = c.seq, time.Now()
	c.hits++
	return e, n
}

// restorePrefix copies the first n positions of e into s.
func (m *Model) restorePrefix(s *state, e *prefixEntry, n int) {
	for li := range e.k {
		m.loadKV(s, li, e.k[li], e.v[li], len(e.tokens), n)
	}
}

// storePrefix keeps the first tokens of ids, whose keys and values are in s, unless the cache already has them. A pin
// (Warm's) keeps the entry from being evicted until Unpin. It returns how many tokens the entry holds (0: not kept).
func (m *Model) storePrefix(scope string, s *state, ids []int, pin string) int {
	c := &m.prefix
	per := m.tokenBytes()
	c.mu.Lock()
	budget := c.budget
	limit := 0
	if budget > 0 {
		limit = int(budget / per) // an entry may fill the budget; older entries make room for it
	}
	n := min(len(ids), limit)
	if n < prefixMin {
		c.mu.Unlock()
		return 0
	}
	for _, cand := range c.entries { // already cached (as far as an entry would keep)?
		if cand.scope == scope && len(cand.tokens) >= n && equalInts(cand.tokens[:n], ids[:n]) {
			c.seq++
			cand.seq, cand.used = c.seq, time.Now()
			if pin != "" {
				if cand.pins == nil {
					cand.pins = map[string]bool{}
				}
				cand.pins[pin] = true
			}
			c.mu.Unlock()
			return len(cand.tokens)
		}
	}
	c.mu.Unlock()

	// copy outside the lock: it's tens of megabytes
	e := &prefixEntry{scope: scope, tokens: append([]int(nil), ids[:n]...), k: make([][]float32, m.hp.layers), v: make([][]float32, m.hp.layers)}
	for li := range e.k {
		e.k[li], e.v[li] = m.saveKV(s, li, n)
	}
	size := int64(n) * per

	if pin != "" {
		e.pins = map[string]bool{pin: true}
	}
	if !m.res.reserveCache(size) { // a cache: when the memory budget is short, don't keep it
		return 0
	}

	var freed int64 // evicted here, released to the budget once the lock is let go
	defer func() { m.res.release(freed) }()
	c.mu.Lock()
	defer c.mu.Unlock()
	for c.used+size > c.budget { // evict the least recently used entry that isn't pinned
		old := -1
		for i, cand := range c.entries {
			if len(cand.pins) == 0 && (old < 0 || cand.seq < c.entries[old].seq) {
				old = i
			}
		}
		if old < 0 {
			freed += size
			return 0 // everything left is pinned: no room
		}
		b := int64(len(c.entries[old].tokens)) * per
		c.used -= b
		freed += b
		c.entries = append(c.entries[:old], c.entries[old+1:]...)
	}
	c.seq++
	e.seq, e.used = c.seq, time.Now()
	c.entries = append(c.entries, e)
	c.used += size
	return n
}

// evictPrefix drops e from the prompt cache (unless it's gone or pinned by now) and returns the bytes it held, for
// the caller to release.
func (m *Model) evictPrefix(e *prefixEntry) int64 {
	c := &m.prefix
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, cand := range c.entries {
		if cand == e && len(e.pins) == 0 {
			b := int64(len(e.tokens)) * m.tokenBytes()
			c.used -= b
			c.entries = append(c.entries[:i], c.entries[i+1:]...)
			return b
		}
	}
	return 0
}

// Unpin releases pin (Warm's) from every entry; unpinned entries are evicted like the rest.
func (m *Model) Unpin(pin string) {
	c := &m.prefix
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.entries {
		delete(e.pins, pin)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// PromptCacheStats says how the prompt cache is doing: how many entries it holds, the bytes they take, and how many
// calls reused a prefix.
type PromptCacheStats struct {
	Entries int
	Bytes   int64
	Pinned  int64 // bytes of entries a Warm pinned
	Hits    uint64
}

// PromptCache reports the prompt cache's state.
func (m *Model) PromptCache() PromptCacheStats {
	c := &m.prefix
	c.mu.Lock()
	defer c.mu.Unlock()
	st := PromptCacheStats{Entries: len(c.entries), Bytes: c.used, Hits: c.hits}
	for _, e := range c.entries {
		if len(e.pins) > 0 {
			st.Pinned += int64(len(e.tokens)) * m.tokenBytes()
		}
	}
	return st
}
