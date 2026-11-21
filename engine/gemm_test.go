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
	for _, nb := range []int{1, 2, 3, 5, 8, 11, 24, 40} {
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
				if math.Float32bits(o[ri*4+c]) != math.Float32bits(want) { // the reference order, bit for bit
					t.Errorf("nb=%d row %d col %d: %v, want %v", nb, ri, c, o[ri*4+c], want)
				}
			}
		}
	}
}

// GGUF=<file> go test -run TestPrefillMatchesReference -v
// Reads prompts with the prompt kernels, with the 4-column kernels, and token
// by token, and checks that the last position's logits and the greedy
// continuation are identical. NLAYERS=k truncates the model.
func TestPrefillMatchesReference(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m, err := Load(p, LoadOptions{Threads: 4, GPU: "off"})
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
	for _, n := range []int{5, 37, 300, 600} {
		ids := all[:n]
		fast, fgen := run(ids, prefillBatch)
		restore := promptKernelsOff()
		ref, rgen := run(ids, prefillBatch)
		restore()
		one, ogen := run(ids, 1) // token by token
		same := func(a, b []float32) bool {
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					return false
				}
			}
			return true
		}
		t.Logf("n=%d: greedy new %v old %v", n, fgen, rgen)
		// every kernel follows one reference order, so the prompt kernels,
		// the 4-column ones and token-by-token reading agree bit for bit
		if !same(fast, ref) || !slices.Equal(fgen, rgen) {
			t.Errorf("n=%d: prompt kernels change the logits", n)
		}
		if !same(one, ref) || !slices.Equal(ogen, rgen) {
			t.Errorf("n=%d: token-by-token reading changes the logits", n)
		}
	}
}

func TestGemmF16x2x4(t *testing.T) {
	if gemmF16x2x4 == nil {
		t.Skip("no 2x4 F16 kernel on this CPU")
	}
	r := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{16, 48, 1024, 40} {
		w := make([]byte, 4*n)
		for i := 0; i < 2*n; i++ {
			h := uint16(0x3000+r.IntN(0x1000)) | uint16(r.IntN(2))<<15
			w[2*i], w[2*i+1] = byte(h), byte(h>>8)
		}
		w0, w1 := w[:2*n], w[2*n:]
		stride := n + 8
		y := make([]float32, 4*stride)
		for i := range y {
			y[i] = float32(r.NormFloat64())
		}
		var o [8]float32
		gemmF16x2x4(w0, w1, y, stride, &o)
		for ri, wr := range [][]byte{w0, w1} {
			for c := 0; c < 4; c++ {
				if want := dotF16F32Go(wr, y[c*stride:c*stride+n]); math.Float32bits(o[ri*4+c]) != math.Float32bits(want) {
					t.Errorf("n=%d row %d col %d: %v, want %v", n, ri, c, o[ri*4+c], want)
				}
			}
		}
	}
}

func TestGemmQ6K1x4(t *testing.T) {
	if gemmQ6K1x4 == nil {
		t.Skip("no Q6_K prompt kernel on this CPU")
	}
	r := rand.New(rand.NewPCG(9, 10))
	for _, nb := range []int{1, 2, 3, 5, 8, 11, 24, 40} {
		w := randBlocks(r, nb, q6kBytes, []int{208})
		grp := make([]byte, nb*q8k6Bytes)
		var plain [4][]byte
		for j := 0; j < 4; j++ {
			x := make([]float32, nb*qk)
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
			plain[j] = make([]byte, nb*q8kBytes)
			quantizeQ8K(x, plain[j])
			packQ8K6(plain[j], grp, j, nb)
		}
		var o [4]float32
		gemmQ6K1x4(w, grp, nb, &o)
		for c := 0; c < 4; c++ {
			want := dotQ6KQ8KGo(w, plain[c], nb)
			if math.Float32bits(o[c]) != math.Float32bits(want) { // the reference order, bit for bit
				t.Errorf("nb=%d col %d: %v, want %v", nb, c, o[c], want)
			}
		}
	}
}
