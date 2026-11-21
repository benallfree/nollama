package engine

import (
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// makeQ8K4 quantizes 4 rows of nb*qk values and returns them in both layouts.
func makeQ8K4(r *rand.Rand, nb int) (plain [4][]byte, grp []byte) {
	grp = make([]byte, nb*q8k4Bytes)
	for j := 0; j < 4; j++ {
		x := make([]float32, nb*qk)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		plain[j] = make([]byte, nb*q8kBytes)
		quantizeQ8K(x, plain[j])
		packQ8K4(plain[j], grp, j, nb)
	}
	return
}

func TestGemmQ4K2x4(t *testing.T) {
	if gemmQ4K2x4 == nil {
		t.Skip("no 2x4 Q4_K kernel on this CPU")
	}
	r := rand.New(rand.NewPCG(3, 4))
	for _, nb := range []int{1, 2, 3, 8, 24} {
		w := randBlocks(r, 2*nb, q4kBytes, []int{0, 2})
		w0, w1 := w[:nb*q4kBytes], w[nb*q4kBytes:]
		plain, grp := makeQ8K4(r, nb)
		var o [8]float32
		s0, s1 := make([]byte, nb*q4kScaleBytes), make([]byte, nb*q4kScaleBytes)
		q4kScales(w0, s0, nb)
		q4kScales(w1, s1, nb)
		gemmQ4K2x4(w0, w1, s0, s1, grp, nb, &o)
		for c := 0; c < 4; c++ {
			for ri, wr := range [][]byte{w0, w1} {
				want := dotQ4KQ8KGo(wr, plain[c], nb)
				if math.Abs(float64(o[ri*4+c]-want)) > 1e-4*(1+math.Abs(float64(want))) {
					t.Errorf("nb=%d row %d col %d: %v, want %v", nb, ri, c, o[ri*4+c], want)
				}
			}
		}
	}
}

// GGUF=<file> go test -run TestPrefillMatchesReference -v
// Reads prompts with the prompt (GEMM) kernels on and off and compares the
// last position's logits and the greedy continuation, against the spread the
// old path shows between batched and token-by-token reading. NLAYERS=k
// truncates the model.
func TestPrefillMatchesReference(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m, err := Load(p, LoadOptions{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if nl, _ := strconv.Atoi(os.Getenv("NLAYERS")); nl > 0 {
		m.layers = m.layers[:nl]
	}
	run := func(ids []int, batch int) ([]float32, []int) {
		s := m.newState(len(ids)+9, min(len(ids), batch))
		for pos := 0; pos < len(ids); pos += batch {
			m.forward(s, ids[pos:min(pos+batch, len(ids))], pos)
		}
		logits := append([]float32(nil), s.logits...)
		var gen []int
		for i := 0; i < 8; i++ {
			best := 0
			for j, v := range s.logits {
				if v > s.logits[best] {
					best = j
				}
			}
			gen = append(gen, best)
			m.forward(s, []int{best}, len(ids)+i)
		}
		return logits, gen
	}
	text := strings.Repeat("The harbor town woke slowly; gulls argued over the nets while the ferry captain counted crates. ", 40)
	all := m.Tokenize(text)
	saved := gemmQ4K2x4
	defer func() { gemmQ4K2x4 = saved }()
	for _, n := range []int{5, 37, 300, 600} {
		ids := all[:n]
		gemmQ4K2x4 = saved
		fast, fgen := run(ids, prefillBatch)
		gemmQ4K2x4 = nil
		ref, rgen := run(ids, prefillBatch)
		one, _ := run(ids, 1) // token by token: the old engine's own spread
		diff := func(a, b []float32) (d float64) {
			for i := range a {
				d = max(d, math.Abs(float64(a[i]-b[i])))
			}
			return d
		}
		dFast, dOne := diff(fast, ref), diff(one, ref)
		t.Logf("n=%d: max |logit diff| new vs old %.3f, old token-by-token vs old batched %.3f; greedy new %v old %v", n, dFast, dOne, fgen, rgen)
		// Float rounding in a different order moves the logits by about as
		// much as reading the prompt token by token does; more is a bug.
		if dFast > 2*dOne+0.05 {
			t.Errorf("n=%d: new path moved logits by %.3f, the old engine's own spread is %.3f", n, dFast, dOne)
		}
		if !slices.Equal(fgen[:4], rgen[:4]) {
			t.Errorf("n=%d: greedy continuation differs: %v vs %v", n, fgen, rgen)
		}
	}
}
