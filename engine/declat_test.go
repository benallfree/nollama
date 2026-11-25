//go:build darwin

package engine

// Decode-step latency on Metal: wall time vs GPU time for the search (int_find) and the step, per position range.
// NOLLAMA_DECLAT=<path to an int model gguf>.

import (
	"math/rand"
	"os"
	"testing"
	"time"
)

func TestDecodeLatency(t *testing.T) {
	path := os.Getenv("NOLLAMA_DECLAT")
	if path == "" {
		t.Skip("set NOLLAMA_DECLAT to an int model")
	}
	m, err := LoadIntModelOptions(path, IntOptions{GPU: "auto"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Logf("device %s", m.Device())
	n := 2048
	s, err := m.NewSession(n)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	g := s.g
	if err := s.Start(m.BOS()); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(1))
	var wf, ws, gf, gs time.Duration
	cnt := 0
	report := func(at int) {
		c := time.Duration(cnt)
		t.Logf("pos %4d: find wall %6.3f ms gpu %6.3f | step wall %6.3f ms gpu %6.3f (%d dispatches) | %.0f tok/s",
			at, ms(wf/c), ms(gf/c), ms(ws/c), ms(gs/c), len(g.ds), float64(time.Second)/float64((wf+ws)/c))
		wf, ws, gf, gs, cnt = 0, 0, 0, 0, 0
	}
	for pos := 1; pos < n; pos++ {
		num, den := uint64(r.Int63n(1<<30)+1), uint64(1<<30)+1
		t0 := time.Now()
		tok, _, err := g.findTok(func(total int64) int64 { return int64((num*uint64(total) - 1) / den) })
		if err != nil {
			t.Fatal(err)
		}
		t1 := time.Now()
		gf += m.gpu.dev.lastGPU()
		if err := g.step(tok, pos); err != nil {
			t.Fatal(err)
		}
		t2 := time.Now()
		gs += m.gpu.dev.lastGPU()
		wf += t1.Sub(t0)
		ws += t2.Sub(t1)
		cnt++
		if pos%256 == 0 {
			report(pos)
		}
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func TestDecodeDispatches(t *testing.T) {
	path := os.Getenv("NOLLAMA_DECLAT")
	if path == "" {
		t.Skip("set NOLLAMA_DECLAT to an int model")
	}
	m, err := LoadIntModelOptions(path, IntOptions{GPU: "auto"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s, err := m.NewSession(256)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	hist := map[string]int{}
	var seq []string
	for i, d := range s.g.ds {
		hist[d.K]++
		if i < 20 {
			seq = append(seq, d.K)
		}
	}
	t.Logf("%d dispatches: %v", len(s.g.ds), hist)
	t.Logf("first 20: %v", seq)
	t.Logf("D=%d I=%d V=%d nq=%d nkv=%d hd=%d layers=%d", m.D, m.I, m.V, m.nq, m.nkv, m.hd, len(m.layers))
}
