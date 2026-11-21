package engine

import (
	"os"
	"strconv"
	"testing"
	"time"
)

// GGUF=... [THREADS=4] [PTOK=590] [NOGEMM=1] go test -run XXX -bench Prefill -cpuprofile cpu.out
// NOGEMM=1 turns off the 2-row prompt kernels, for before/after numbers.
// Measures prompt reading alone: one forward pass over PTOK tokens, in
// prefillBatch chunks, as Generate does.
func BenchmarkPrefill(b *testing.B) {
	p := os.Getenv("GGUF")
	if p == "" {
		b.Skip("GGUF not set")
	}
	th, _ := strconv.Atoi(os.Getenv("THREADS"))
	if th == 0 {
		th = 4
	}
	n, _ := strconv.Atoi(os.Getenv("PTOK"))
	if n == 0 {
		n = 590
	}
	if os.Getenv("NOGEMM") != "" {
		defer promptKernelsOff()()
	}
	m, err := Load(p, LoadOptions{Threads: th, GPU: benchGPU()})
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	ids := make([]int, n)
	for i := range ids {
		ids[i] = 1000 + (i*7919)%50000
	}
	s := m.newState(n+1, min(n, prefillBatch))
	run := func() {
		for pos := 0; pos < n; pos += prefillBatch {
			m.forward(s, ids[pos:min(pos+prefillBatch, n)], pos)
		}
	}
	run() // warm-up: pages the weights in
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run()
	}
	b.ReportMetric(float64(n*b.N)/b.Elapsed().Seconds(), "tok/s")
}

// GGUF=... PAB=1 [THREADS=1] [PTOK=128] [TRIALS=8] go test -run PrefillAB -v
// Prompt reading with the 2-row kernels off and on, alternating, fastest of
// TRIALS runs each: steadier than a benchmark on a busy machine.
func TestPrefillAB(t *testing.T) {
	p := os.Getenv("GGUF")
	if p == "" || os.Getenv("PAB") == "" {
		t.Skip("GGUF and PAB not set")
	}
	th, _ := strconv.Atoi(os.Getenv("THREADS"))
	if th == 0 {
		th = 1
	}
	n, _ := strconv.Atoi(os.Getenv("PTOK"))
	if n == 0 {
		n = 128
	}
	trials, _ := strconv.Atoi(os.Getenv("TRIALS"))
	if trials == 0 {
		trials = 8
	}
	m, err := Load(p, LoadOptions{Threads: th, GPU: benchGPU()})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := make([]int, n)
	for i := range ids {
		ids[i] = 1000 + (i*7919)%50000
	}
	restore := promptKernelsOff()
	defer restore()
	set := func(on bool) {
		restore()
		if !on {
			restore = promptKernelsOff()
		}
	}
	states := map[bool]*state{}
	for _, on := range []bool{false, true} {
		set(on)
		states[on] = m.newState(n+1, min(n, prefillBatch))
	}
	best := map[bool]float64{false: 1e18, true: 1e18}
	for i := 0; i < trials; i++ {
		for _, on := range []bool{false, true} {
			set(on)
			s := states[on]
			t0 := time.Now()
			for pos := 0; pos < n; pos += prefillBatch {
				m.forward(s, ids[pos:min(pos+prefillBatch, n)], pos)
			}
			best[on] = min(best[on], time.Since(t0).Seconds())
		}
	}
	t.Logf("threads %d, %d tokens: old %.1f tok/s, new %.1f tok/s (%.2fx)", th, n,
		float64(n)/best[false], float64(n)/best[true], best[false]/best[true])
}
