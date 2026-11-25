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

// TestDecodeProfile: each dispatch of a decode step run alone (best of 10, after real steps so the buffers hold real
// data), summed per kernel, against the whole step. NOLLAMA_DECLAT=<model>, NOLLAMA_DECPOS (default 1000).
func TestDecodeProfile(t *testing.T) {
	path := os.Getenv("NOLLAMA_DECLAT")
	if path == "" {
		t.Skip("set NOLLAMA_DECLAT to an int model")
	}
	m, err := LoadIntModelOptions(path, IntOptions{GPU: "auto"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	at := envInt("NOLLAMA_DECPOS", 1000)
	s, err := m.NewSession(at + 64)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	g := s.g
	for pos := 0; pos < at; pos++ {
		if err := g.step(100+pos%500, pos); err != nil {
			t.Fatal(err)
		}
	}
	best := func(f func() error) time.Duration {
		b := time.Hour
		for i := 0; i < 10; i++ {
			if err := f(); err != nil {
				t.Fatal(err)
			}
			b = min(b, m.gpu.dev.lastGPU())
		}
		return b
	}
	whole := best(func() error { return g.step(123, at) })
	var wall time.Duration
	for i := 0; i < 10; i++ {
		t0 := time.Now()
		g.step(123, at)
		wall += time.Since(t0)
	}
	type kt struct {
		n int
		d time.Duration
	}
	per := map[string]*kt{}
	var sum time.Duration
	for i := range g.ds {
		d := best(func() error { return m.gpu.dev.run(g.ds[i : i+1]) })
		k := per[g.ds[i].K]
		if k == nil {
			k = &kt{}
			per[g.ds[i].K] = k
		}
		k.n++
		k.d += d
		sum += d
	}
	t.Logf("pos %d: whole step GPU %.3f ms (wall %.3f ms), %d dispatches; sum of dispatches alone %.3f ms", at,
		ms(whole), ms(wall/10), len(g.ds), ms(sum))
	for k, v := range per {
		t.Logf("  %-12s x%3d  %7.3f ms  (%5.1f%%)  %6.1f us each", k, v.n, ms(v.d), 100*float64(v.d)/float64(sum),
			float64(v.d.Microseconds())/float64(v.n))
	}
}

// TestDecodeMarginal: the step's GPU time with each kernel left out (outputs then garbage; costs here don't depend on
// the data), so each kernel's in-stream cost. NOLLAMA_DECLAT=<model>, NOLLAMA_DECPOS (default 1000).
func TestDecodeMarginal(t *testing.T) {
	path := os.Getenv("NOLLAMA_DECLAT")
	if path == "" {
		t.Skip("set NOLLAMA_DECLAT to an int model")
	}
	m, err := LoadIntModelOptions(path, IntOptions{GPU: "auto"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	at := envInt("NOLLAMA_DECPOS", 1000)
	s, err := m.NewSession(at + 64)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	g := s.g
	for pos := 0; pos < at; pos++ {
		if err := g.step(100+pos%500, pos); err != nil {
			t.Fatal(err)
		}
	}
	all := append([]idisp(nil), g.ds...)
	nch := uint((at + intACH) / intACH)
	for i := range all {
		if attnChunked(all[i].K) {
			all[i].Groups.H = nch
		}
	}
	timeList := func(ds []idisp) time.Duration {
		b := time.Hour
		for i := 0; i < 15; i++ {
			if err := m.gpu.dev.run(ds); err != nil {
				t.Fatal(err)
			}
			b = min(b, m.gpu.dev.lastGPU())
		}
		return b
	}
	whole := timeList(all)
	kinds := map[string]bool{}
	for _, d := range all {
		kinds[d.K] = true
	}
	t.Logf("pos %d: whole step %.3f ms", at, ms(whole))
	for k := range kinds {
		var rest []idisp
		n := 0
		for _, d := range all {
			if d.K != k {
				rest = append(rest, d)
			} else {
				n++
			}
		}
		without := timeList(rest)
		t.Logf("  %-12s x%3d  costs %6.3f ms (%5.1f%%)  %5.1f us each", k, n, ms(whole-without),
			100*float64(whole-without)/float64(whole), float64((whole-without).Microseconds())/float64(n))
	}
}
