package engine

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

var compactTypes = []uint32{typeF16, typeQ8_0, typeQ4_0}

func randRow(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64()) * 3
	}
	return x
}

// Each format's rows come back within its rounding, and re-quantizing a dequantized row gives the same bytes (so a
// prompt cache entry restored into a compact cache is exactly what was saved).
func TestKVRowRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	const hd = 128
	for _, typ := range compactTypes {
		rb := kvBytes(typ, hd)
		row, row2 := make([]byte, rb), make([]byte, rb)
		out := make([]float32, hd)
		for trial := 0; trial < 200; trial++ {
			x := randRow(r, hd)
			if trial == 0 {
				clear(x[:32]) // an all-zero block
			}
			quantizeKVRow(typ, x, row)
			dequantKVRow(typ, row, out)
			for b := 0; b < hd/32; b++ {
				var amax float64
				for _, v := range x[b*32 : b*32+32] {
					amax = math.Max(amax, math.Abs(float64(v)))
				}
				// q4_0 clips one side at 7/8 of the block's largest value (ggml's format), so its error reaches amax/8
				tol := map[uint32]float64{typeF16: amax / 1024, typeQ8_0: amax/127*0.51 + amax/2048, typeQ4_0: amax/8 + amax/2048}[typ]
				for i := b * 32; i < b*32+32; i++ {
					if d := math.Abs(float64(out[i] - x[i])); d > tol {
						t.Fatalf("%s: element %d: %g -> %g (error %g > %g)", kvTypeName(typ), i, x[i], out[i], d, tol)
					}
				}
			}
			quantizeKVRow(typ, out, row2)
			if !bytes.Equal(row, row2) {
				t.Fatalf("%s: re-quantizing a dequantized row changed it", kvTypeName(typ))
			}
		}
	}
}

func TestParseKVCacheType(t *testing.T) {
	for _, s := range []string{"", "f32", "F16", "q8_0", " Q4_0 "} {
		if _, err := parseKVCacheType(s); err != nil {
			t.Errorf("%q: %v", s, err)
		}
	}
	if _, err := parseKVCacheType("q5_1"); err == nil {
		t.Error("q5_1: no error")
	}
}

// kvTestState fills a compact cache for one layer with n random positions, in order, as forward does.
func kvTestState(typ uint32, hd, kvh, n int, r *rand.Rand) (*Model, *state) {
	m := &Model{hp: hparams{headDim: hd, kvHeads: kvh, layers: 1}, kvTyp: typ, pool: newPool(1, nil)}
	s := &state{maxCtx: n, ctxP: (n + 31) &^ 31}
	m.allocCompact(s)
	for p := 0; p < n; p++ {
		m.storeKey(s, 0, p, randRow(r, kvh*hd))
		m.storeValue(s, 0, p, randRow(r, kvh*hd))
	}
	return m, s
}

// scoresQ and wsumQ (with and without the 4-way assembly kernels) match a float64 reference computed from what the
// cache holds (saveKV's full-precision view), over enough positions to span several blocks and a partial stage.
func TestKVAttentionKernels(t *testing.T) {
	const hd, kvh = 128, 2
	r := rand.New(rand.NewPCG(3, 4))
	qb := randRow(r, 4*hd)
	for _, n := range []int{549, 512, 7} {
		for _, typ := range compactTypes {
			m, s := kvTestState(typ, hd, kvh, n, r)
			defer m.pool.close()
			kv, vv := m.saveKV(s, 0, n) // keys [kvd][n], values [n][kvd]
			const h = 1                 // the head under test
			wantS := make([]float64, 4*n)
			var smax float64
			for i := 0; i < 4; i++ {
				for p := 0; p < n; p++ {
					var d float64
					for j := 0; j < hd; j++ {
						d += float64(qb[i*hd+j]) * float64(kv[(h*hd+j)*n+p])
					}
					wantS[i*n+p] = d
					smax = math.Max(smax, math.Abs(d))
				}
			}
			w := make([]float32, 4*s.ctxP)
			for i := 0; i < 4; i++ {
				for p := 0; p < n; p++ {
					w[i*s.ctxP+p] = r.Float32() / float32(n)
				}
				for p := n; p < s.ctxP; p++ {
					w[i*s.ctxP+p] = float32(math.NaN()) // past the context: must not be read
				}
			}
			wantO := make([]float64, 4*hd)
			var omax float64
			for i := 0; i < 4; i++ {
				for j := 0; j < hd; j++ {
					for p := 0; p < n; p++ {
						wantO[i*hd+j] += float64(w[i*s.ctxP+p]) * float64(vv[p*kvh*hd+h*hd+j])
					}
					omax = math.Max(omax, math.Abs(wantO[i*hd+j]))
				}
			}
			saved80, saved16 := dotQ80x4, dotF16x4
			for _, accel := range []bool{true, false} {
				if !accel {
					dotQ80x4, dotF16x4 = nil, nil
				}
				name := fmt.Sprintf("%s n=%d accel=%v", kvTypeName(typ), n, accel)
				sw := s.kv.scratch[0]
				sc := make([]float32, 4*s.ctxP)
				m.scoresQ(s, sw, 0, h, n, qb, sc)
				tol := smax * 1e-5
				if typ != typeF16 { // keys are scored against 8-bit queries
					tol = smax * 0.02
				}
				for i := 0; i < 4; i++ {
					for p := 0; p < n; p++ {
						if d := math.Abs(float64(sc[i*s.ctxP+p]) - wantS[i*n+p]); d > tol {
							t.Fatalf("%s: score %d,%d = %g, want %g", name, i, p, sc[i*s.ctxP+p], wantS[i*n+p])
						}
					}
				}
				ww := append([]float32(nil), w...)
				for i := 0; i < 4; i++ { // the caller zeroes weights past each query's context, up to a multiple of 4
					clear(ww[i*s.ctxP+n : i*s.ctxP+min(s.ctxP, (n+3)&^3)])
				}
				ob := make([]float32, 4*hd)
				m.wsumQ(s, sw, 0, h, n, ww, ob)
				otol := omax * 1e-5
				if typ != typeF16 { // weights of full blocks are quantized to 8 bits
					otol = omax * 0.02
				}
				for i := range ob {
					if d := math.Abs(float64(ob[i]) - wantO[i]); !(d <= otol) {
						t.Fatalf("%s: output %d = %g, want %g", name, i, ob[i], wantO[i])
					}
				}
			}
			dotQ80x4, dotF16x4 = saved80, saved16
		}
	}
}

// Saving a compact layer and loading it into another state gives back the same numbers.
func TestKVSaveLoad(t *testing.T) {
	const hd, kvh, n = 128, 2, 77
	r := rand.New(rand.NewPCG(7, 8))
	for _, typ := range compactTypes {
		m, s := kvTestState(typ, hd, kvh, n, r)
		k, v := m.saveKV(s, 0, n)
		s2 := &state{maxCtx: n + 40, ctxP: (n + 40 + 31) &^ 31}
		m.allocCompact(s2)
		m.loadKV(s2, 0, k, v, n, n)
		k2, v2 := m.saveKV(s2, 0, n)
		for i := range k {
			if k[i] != k2[i] {
				t.Fatalf("%s: key %d: %g -> %g", kvTypeName(typ), i, k[i], k2[i])
			}
		}
		for i := range v {
			if v[i] != v2[i] {
				t.Fatalf("%s: value %d: %g -> %g", kvTypeName(typ), i, v[i], v2[i])
			}
		}
		m.pool.close()
	}
}

// kvQualityText is a fixed text for comparing cache types (about 600 tokens).
var kvQualityText = strings.Repeat("The harbor town sat at the mouth of a cold grey river, where the water slowed and spread into a wide harbor before meeting the sea. ", 2) +
	"For three hundred years it had lived on fishing, on salt, and on the small trade that came down the river in flat-bottomed boats loaded with timber, wool and iron from the hill country. " +
	"The houses were built of dark stone, tall and narrow, with steep slate roofs that shed the rain that fell on most days of the year. " +
	"At the end of the long stone pier stood the lighthouse. It was older than most of the town, older than the church with its crooked spire. " +
	"Nobody living remembered who had built it. The keepers had come and gone, each one serving for twenty or thirty years before handing the keys to the next, " +
	"and each one leaving behind a shelf of logbooks that recorded the weather, the ships that passed, the lamps that were trimmed and the oil that was burned. " +
	"Mara was twelve years old the winter the storms came. She lived with her grandmother in a house on Rope Street, two doors down from the sailmaker, " +
	"and she spent her afternoons on the pier watching the boats come in. She knew every boat by its shape and color long before she could read the name painted on its bow. " +
	"Tobias had been keeper for thirty-one years. He said the sea had moods like a person, and that a good keeper learned to read them the way a mother learns to read a child. " +
	"He said the light was a promise: as long as it burned, any sailor out in the dark would know where home was. That winter the promise was tested."

// Each cache type's log-probabilities over a fixed text stay close to the full-precision cache's.
// GGUF=<a Qwen3 .gguf> go test -run TestKVCacheQuality -v
func TestKVCacheQuality(t *testing.T) {
	path := os.Getenv("GGUF")
	if path == "" {
		t.Skip("set GGUF to a Qwen3 GGUF file")
	}
	var ref []float64
	var refTop []int
	var refPPL float64
	for _, name := range KVCacheTypes {
		m, err := Load(path, LoadOptions{Threads: 4, KVCacheType: name})
		if err != nil {
			t.Fatal(err)
		}
		ids := m.Tokenize(kvQualityText)
		top, lp := m.teacherForce(ids[:1], ids[1:])
		m.Close()
		var nll float64
		for _, v := range lp {
			nll -= v
		}
		ppl := math.Exp(nll / float64(len(lp)))
		if ref == nil {
			ref, refTop, refPPL = lp, top, ppl
			t.Logf("%-5s %d tokens, perplexity %.4f", name, len(lp), ppl)
			continue
		}
		var sumD, maxD float64
		agree := 0
		for i := range lp {
			d := math.Abs(lp[i] - ref[i])
			sumD += d
			maxD = math.Max(maxD, d)
			if top[i] == refTop[i] {
				agree++
			}
		}
		ratio := ppl / refPPL
		t.Logf("%-5s perplexity %.4f (%+.1f%%), mean |Δ logprob| %.4f, max %.3f, same top token %.1f%%", name, ppl,
			100*(ratio-1), sumD/float64(len(lp)), maxD, 100*float64(agree)/float64(len(lp)))
		// Small models are sensitive: rounding keys or values to f16 alone (Ollama's default cache) moves Qwen3
		// 0.6B's log-probabilities by about 0.1 and flips some near-tied top tokens, so the test bounds perplexity.
		if limit := map[string]float64{KVCacheF16: 1.03, KVCacheQ8_0: 1.05, KVCacheQ4_0: 1.15}[name]; ratio > limit {
			t.Errorf("%s: perplexity %.3f is %.1f%% above f32's %.3f (limit %.0f%%)", name, ppl, 100*(ratio-1), refPPL, 100*(limit-1))
		}
	}
}

// With a compact cache, resuming from the prompt cache lands close to a full read, and concurrent calls give the same
// answers as a single one. It runs on a tiny random-weight model (head size 32), or on $GGUF.
func TestKVCacheCompactReuse(t *testing.T) {
	path := os.Getenv("GGUF")
	if path == "" {
		path = filepath.Join(t.TempDir(), "tiny.gguf")
		if err := tinymodel.Write(path, tinymodel.Config{Dim: 128}); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{KVCacheF16, KVCacheQ8_0, KVCacheQ4_0} {
		m, err := Load(path, LoadOptions{Threads: 4, KVCacheType: name, PromptCache: 64 << 20})
		if err != nil {
			t.Fatal(err)
		}
		pre := "You answer questions about Acme Notes. " + strings.Repeat("Notes sync across devices within a few seconds of saving. ", 8)
		a := m.ChatPrompt([]Message{{Role: "system", Content: pre}, {Role: "user", Content: "How fast do notes sync?"}})
		b := m.ChatPrompt([]Message{{Role: "system", Content: pre}, {Role: "user", Content: "Can I share a notebook?"}})
		if _, err := m.Generate(context.Background(), a, GenerateOptions{MaxTokens: 2}, nil); err != nil {
			t.Fatal(err)
		}
		ids := m.Tokenize(b)
		e, n := m.lookupPrefix("", ids)
		if e == nil {
			t.Fatalf("%s: no cached prefix", name)
		}
		// resuming after the cached prefix lands close to reading b whole (prefill splits and block boundaries
		// differ, so not bit for bit; TestKVSaveLoad checks the copy itself is exact)
		resumed := m.newState(len(ids)+8, prefillBatch)
		m.restorePrefix(resumed, e, n)
		m.forward(resumed, ids[n:], n)
		full := m.newState(len(ids)+8, prefillBatch)
		m.forward(full, ids, 0)
		var worst float64
		for i := range full.logits {
			worst = math.Max(worst, math.Abs(float64(full.logits[i]-resumed.logits[i])))
		}
		if argmax(full.logits) != argmax(resumed.logits) || worst > 1.5 {
			t.Fatalf("%s: after restoring %d cached tokens, logits differ from a full read by up to %.3f (top %d vs %d)",
				name, n, worst, argmax(resumed.logits), argmax(full.logits))
		}
		t.Logf("%s: %d tokens from the prompt cache; largest logit difference from a full read %.3f", name, n, worst)

		want, err := m.Generate(context.Background(), b, GenerateOptions{MaxTokens: 12}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for i := 0; i < 3; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				got, err := m.Generate(context.Background(), b, GenerateOptions{MaxTokens: 12}, nil)
				if err != nil || got.Text != want.Text {
					t.Errorf("%s: concurrent answer %q, want %q (%v)", name, got.Text, want.Text, err)
				}
			}()
		}
		wg.Wait()
		if m.Info().KVCacheType != name {
			t.Errorf("Info().KVCacheType = %q, want %q", m.Info().KVCacheType, name)
		}
		m.Close()
	}
}

// One decode step's attention for one KV head over 2,048 positions, f32 against each compact type.
// go test -run '^$' -bench BenchmarkKVAttention ./engine
func BenchmarkKVAttention(b *testing.B) {
	const hd, kvh, n = 128, 1, 2048
	r := rand.New(rand.NewPCG(5, 6))
	qb := randRow(r, 4*hd)
	w := make([]float32, 4*n)
	for i := range w {
		w[i] = r.Float32() / n
	}
	sc := make([]float32, 4*n)
	ob := make([]float32, 4*hd)
	kT, v := randRow(r, hd*n), randRow(r, n*hd)
	b.Run("f32/scores", func(b *testing.B) {
		for b.Loop() {
			scores4(qb, kT, n, n/16, sc, n)
		}
	})
	b.Run("f32/wsum", func(b *testing.B) {
		for b.Loop() {
			wsum4(w, n, v, hd, n/4, ob, hd, hd)
		}
	})
	for _, typ := range compactTypes {
		m, s := kvTestState(typ, hd, kvh, n, r)
		sw := s.kv.scratch[0]
		b.Run(kvTypeName(typ)+"/scores", func(b *testing.B) {
			for b.Loop() {
				m.scoresQ(s, sw, 0, 0, n, qb, sc)
			}
		})
		b.Run(kvTypeName(typ)+"/wsum", func(b *testing.B) {
			for b.Loop() {
				m.wsumQ(s, sw, 0, 0, n, w, ob)
			}
		})
		m.pool.close()
	}
}

// unpackQ40 gives q8_0 blocks that dequantize to the same numbers as the q4_0 blocks.
func TestUnpackQ40(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	src := make([]byte, 4*q40Bytes)
	for i := range src {
		src[i] = byte(r.Uint32())
	}
	for b := 0; b < 4; b++ {
		src[b*q40Bytes+1] = 0x3c // a sane f16 scale
	}
	dst := make([]byte, 4*q80Bytes)
	unpackQ40(src, dst, 4)
	a, b := make([]float32, 128), make([]float32, 128)
	dequantKVRow(typeQ4_0, src, a)
	dequantRow(typeQ8_0, dst, b)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("element %d: q4_0 %g, unpacked %g", i, a[i], b[i])
		}
	}
}

// compactBytes, which the memory budget reserves, matches what allocCompact allocates.
func TestKVCompactBytes(t *testing.T) {
	for _, typ := range compactTypes {
		m := &Model{hp: hparams{headDim: 128, kvHeads: 8, layers: 3}, kvTyp: typ, pool: newPool(2, nil)}
		s := &state{ctxP: 320}
		m.allocCompact(s)
		c := s.kv
		var got int64
		for li := range c.k {
			got += int64(len(c.k[li]) + len(c.v[li]))
			if c.stage != nil {
				got += int64(4 * len(c.stage[li]))
			}
		}
		for _, sw := range c.scratch {
			got += int64(len(sw.q) + len(sw.qr) + len(sw.row) + 4*len(sw.tail))
		}
		if want := m.compactBytes(320); got != want {
			t.Errorf("%s: allocCompact allocates %d bytes, compactBytes says %d", kvTypeName(typ), got, want)
		}
		m.pool.close()
	}
}
