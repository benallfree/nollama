package engine

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
	"time"
)

// bpeMergeRescan is the plain merge loop (a rescan per merge), the reference for bpeMerge.
func (t *tokenizer) bpeMergeRescan(w string) []string {
	syms := make([]string, 0, len(w))
	for i := 0; i < len(w); i++ {
		syms = append(syms, string(t.byteEnc[w[i]]))
	}
	for len(syms) > 1 {
		bestRank, bestI := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[syms[i]+" "+syms[i+1]]; ok && (bestRank < 0 || r < bestRank) {
				bestRank, bestI = r, i
			}
		}
		if bestI < 0 {
			break
		}
		syms[bestI] += syms[bestI+1]
		syms = append(syms[:bestI+1], syms[bestI+2:]...)
	}
	return syms
}

// TestBPEMerge: the heap merge gives the rescan's symbols on real words and random ones, and a long run of letters
// takes milliseconds.
func TestBPEMerge(t *testing.T) {
	m := loadIntTest(t, "0.5B")
	tk := m.tok
	var words []string
	for _, f := range []string{"bjj.html", "slice.html"} {
		words = append(words, preTokenize(string(readVec(t, f)))...)
	}
	rng := rand.New(rand.NewSource(1))
	const alpha = "aaabbcdeeeilnorst  _-=.()0123456789\\n\\t\\xc3\\xa9"
	for k := 0; k < 3000; k++ {
		b := make([]byte, 1+rng.Intn(80))
		for i := range b {
			b[i] = alpha[rng.Intn(len(alpha))]
		}
		words = append(words, string(b))
	}
	words = append(words, strings.Repeat("a", 300), strings.Repeat("ab", 200), strings.Repeat("the", 150))
	for _, w := range words {
		if got, want := tk.bpeMerge(w), tk.bpeMergeRescan(w); !slices.Equal(got, want) {
			t.Fatalf("%q: %q, want %q", w, got, want)
		}
	}
	long := strings.Repeat("abcdefghij", 1600) // 16 KB of letters: one pre-token
	t0 := time.Now()
	tk.bpeMerge(long)
	if d := time.Since(t0); d > time.Second {
		t.Fatalf("16 KB word took %v", d)
	} else {
		t.Logf("%d words agree; a 16 KB word merges in %v", len(words), d)
	}
}
