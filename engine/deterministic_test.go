package engine

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/benallfree/nollama/internal/tinymodel"
)

// Deterministic mode: the logits for every position must be the same bits whether the prompt is read one token at a
// time (as decode does) or in batches, at any thread count, after a prompt-cache restore, and on every architecture.
// The golden hashes below were computed on arm64 (Apple M1 Max) and checked on amd64; a mismatch on another CPU
// means deterministic mode isn't.

const detTokens = 40

// detGolden: sha256 of the tiny model's GGUF bytes and of the float32 bits of the serial logits for each fixture.
var detGolden = map[string][2]string{
	"f32":  {"76fe28e655f344d9f1fb4cf0d355ae5e311898ccf277f57ba774c30e3017c487", "9e65b308975754ada935ec86ae69540f9cf0514fd0eeff84c3cc5ea4f9f2f244"},
	"q8_0": {"dc78f121bafe9d25f37d2345b8ff76d7be2a3001aac544035eaa0a8133cdd57f", "048d9eec0c033c2c1d878ac1fc757d7dc4fa14479b2dd0c732ecaefa7a625a45"},
}

func detFixture(t *testing.T, kind string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "tiny.gguf")
	if err := tinymodel.Write(p, tinymodel.Config{Seed: 7, Dim: 64, Layers: 2}); err != nil {
		t.Fatal(err)
	}
	if kind == "q8_0" {
		q := filepath.Join(dir, "tiny-q8_0.gguf")
		if err := requantize(p, q, true); err != nil {
			t.Fatal(err)
		}
		p = q
	}
	return p
}

func fileHash(t *testing.T, p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func detIDs(m *Model, n int) []int {
	ids := m.Tokenize("the quick note about sync across devices: write a short title, then count one two three four five, and tell me what the sky color is when the app is fast")
	for len(ids) < n {
		ids = append(ids, ids...)
	}
	return ids[:n]
}

// detLogits runs ids through m reading `batch` tokens per forward (0: all at once) and returns the logits after each
// position (batched forwards only produce the last position's, so each prefix gets its own run when batch > 1).
func detLogits(m *Model, ids []int, batch int) [][]float32 {
	out := make([][]float32, len(ids))
	if batch == 1 {
		s := m.newState(len(ids)+1, 1)
		for i := range ids {
			m.forward(s, ids[i:i+1], i)
			out[i] = append([]float32(nil), s.logits...)
		}
		return out
	}
	for p := range ids {
		b := batch
		if b <= 0 {
			b = p + 1
		}
		s := m.newState(len(ids)+1, b)
		for pos := 0; pos <= p; pos += b {
			m.forward(s, ids[pos:min(pos+b, p+1)], pos)
		}
		out[p] = append([]float32(nil), s.logits...)
	}
	return out
}

func logitsHash(ls [][]float32) string {
	h := sha256.New()
	var b [4]byte
	for _, l := range ls {
		for _, v := range l {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func firstDiff(a, b [][]float32) (int, float64) {
	for p := range a {
		var mx float64
		for i := range a[p] {
			if math.Float32bits(a[p][i]) != math.Float32bits(b[p][i]) {
				mx = math.Max(mx, math.Abs(float64(a[p][i]-b[p][i])))
			}
		}
		if mx > 0 || math.IsNaN(mx) {
			return p, mx
		}
	}
	return -1, 0
}

func TestDeterministic(t *testing.T) {
	for _, kind := range []string{"f32", "q8_0"} {
		t.Run(kind, func(t *testing.T) {
			p := detFixture(t, kind)
			load := func(threads int) *Model {
				m, err := Load(p, LoadOptions{Threads: threads, Deterministic: true, PromptCache: 64 << 20})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { m.Close() })
				return m
			}
			m1 := load(1)
			if m1.Info().Kernels != kernelName+" (deterministic)" {
				t.Fatalf("kernels %q", m1.Info().Kernels)
			}
			ids := detIDs(m1, detTokens)
			ref := detLogits(m1, ids, 1)
			m3 := load(3)
			for _, c := range []struct {
				name string
				got  [][]float32
			}{
				{"batch=all", detLogits(m1, ids, 0)},
				{"batch=4", detLogits(m1, ids, 4)},
				{"batch=7", detLogits(m1, ids, 7)},
				{"threads=3 serial", detLogits(m3, ids, 1)},
				{"threads=3 batch=all", detLogits(m3, ids, 0)},
			} {
				if pos, d := firstDiff(ref, c.got); pos >= 0 {
					t.Errorf("%s: logits differ from serial at position %d (max |d| %g)", c.name, pos, d)
				}
			}

			// a prompt-cache hit restores batch-computed keys and values: the continuation must match serial decode
			s := m1.newState(len(ids)+1, len(ids))
			m1.forward(s, ids[:32], 0)
			m1.storePrefix("", s, ids[:32], "")
			s2 := m1.newState(len(ids)+1, 1)
			e, n := m1.lookupPrefix("", ids)
			if e == nil || n != 32 {
				t.Fatalf("prompt cache: got %d tokens", n)
			}
			m1.restorePrefix(s2, e, n)
			for i := n; i < len(ids); i++ {
				m1.forward(s2, ids[i:i+1], i)
				if pos, d := firstDiff(ref[i:i+1], [][]float32{s2.logits}); pos >= 0 {
					t.Fatalf("after a prompt-cache restore: position %d differs (max |d| %g)", i, d)
				}
			}

			// Score in batches must give the same logprobs, bit for bit, as Score one token at a time
			sc1, err := m1.Score(context.Background(), ids, ScoreOptions{Batch: 1, TopLogprobs: 3})
			if err != nil {
				t.Fatal(err)
			}
			for _, b := range []int{4, 7, 64} {
				scb, err := m3.Score(context.Background(), ids, ScoreOptions{Batch: b, TopLogprobs: 3})
				if err != nil {
					t.Fatal(err)
				}
				for i := range sc1 {
					if math.Float64bits(sc1[i].Logprob.Logprob) != math.Float64bits(scb[i].Logprob.Logprob) ||
						sc1[i].Top[0] != scb[i].Top[0] {
						t.Fatalf("Score batch %d differs from batch 1 at token %d: %v vs %v", b, i+1, scb[i], sc1[i])
					}
				}
			}

			// the default (assembly) path computes the same logits: every kernel follows the reference order
			md, err := Load(p, LoadOptions{Threads: 2})
			if err != nil {
				t.Fatal(err)
			}
			defer md.Close()
			if pos, d := firstDiff(ref, detLogits(md, ids, 0)); pos >= 0 {
				t.Errorf("default path, batch=all: position %d differs from deterministic serial (max |d| %g)", pos, d)
			}

			gf, gl := fileHash(t, p), logitsHash(ref)
			want := detGolden[kind]
			t.Logf("fixture %s, logits %s", gf, gl)
			switch {
			case want[0] == "":
				t.Logf("no golden hash recorded for %s", kind)
			case gf != want[0]:
				t.Errorf("the tiny model's bytes differ from the golden fixture (tinymodel isn't portable?): %s", gf)
			case gl != want[1]:
				t.Errorf("deterministic logits differ from the golden hash: got %s, want %s", gl, want[1])
			}
		})
	}
}

// With GGUF=<a real model>, the same invariance on its quantized weights (Q4_K, Q6_K, ...); DET_GOLDEN=<hash> also
// checks the serial logits against a hash from another machine.
func TestDeterministicGGUF(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m1, err := Load(p, LoadOptions{Threads: 1, Deterministic: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m1.Close()
	m8, err := Load(p, LoadOptions{Threads: 8, Deterministic: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m8.Close()
	ids := detIDs(m1, 24)
	ref := detLogits(m8, ids, 1)
	if pos, d := firstDiff(ref, detLogits(m8, ids, 0)); pos >= 0 {
		t.Errorf("batch: position %d differs (max |d| %g)", pos, d)
	}
	if pos, d := firstDiff(detLogits(m1, ids[:12], 1), ref); pos >= 0 {
		t.Errorf("1 thread: position %d differs (max |d| %g)", pos, d)
	}
	h := logitsHash(ref)
	t.Logf("serial logits %s (kernels on this CPU otherwise: %s)", h, kernelName)
	if g := os.Getenv("DET_GOLDEN"); g != "" && g != h {
		t.Errorf("differs from DET_GOLDEN %s", g)
	}
}

// With GGUF=<a Q4_K/Q6_K model>, the default (assembly) path: a prompt read in batches gives the same logits as one
// read a token at a time, since those kernels follow the reference order.
func TestBatchInvariantGGUF(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" {
		t.Skip("GGUF not set")
	}
	m, err := Load(p, LoadOptions{Threads: 8})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Logf("kernels %s, tensor types %v", m.Info().Kernels, m.Info().TensorTypes)
	ids := detIDs(m, 24)
	ref := detLogits(m, ids, 1)
	for _, b := range []int{0, 4, 7} {
		if pos, d := firstDiff(ref, detLogits(m, ids, b)); pos >= 0 {
			t.Errorf("batch %d: position %d differs (max |d| %g)", b, pos, d)
		}
	}
	t.Logf("default-path serial logits %s", logitsHash(ref))
}
