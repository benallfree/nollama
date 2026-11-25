package engine

import (
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"
)

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil {
		return v
	}
	return def
}

// TestIntCross checks that the CPU, the GPU stepping one token at a time and the GPU's batches give the same tables
// (the CPU and GPU steps: every table, by hash; batches: each position's counts for the next token).
//
//	INTMODEL     model size (0.5B)
//	INTCROSS_IDS token ids in testdata/intmodel (slice.ids.json)
//	INTCROSS_N   positions (600)
//	INTCROSS_WIN, INTCROSS_SINK  attention window (0: full attention) and sink
//	INTCROSS_CHUNK  positions per FeedBatch (700, so batches cross the 512-row boundary)
func TestIntCross(t *testing.T) {
	size := os.Getenv("INTMODEL")
	if size == "" {
		size = "0.5B"
	}
	ids := vecIDs(t, cmpOr(os.Getenv("INTCROSS_IDS"), "slice.ids.json"))
	n := min(envInt("INTCROSS_N", 600), len(ids))
	win, sink, chunk := envInt("INTCROSS_WIN", 0), envInt("INTCROSS_SINK", 0), envInt("INTCROSS_CHUNK", 700)
	ids = ids[:n]
	cpu := loadIntDev(t, size, "off")
	gpu := loadIntDev(t, size, "auto")
	toks := append([]int{cpu.BOS()}, ids[:n-1]...)

	t0 := time.Now()
	cs, err := cpu.NewWindowSession(n, win, sink)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]string, n)
	wantC := make([]IntCount, n)
	for k, tok := range toks {
		cum, err := cs.Feed(tok)
		if err != nil {
			t.Fatal(err)
		}
		want[k], wantC[k] = tableHash(cum), countOf(cum, ids[k])
	}
	t.Logf("cpu: %d positions in %.1f s", n, time.Since(t0).Seconds())

	t0 = time.Now()
	gs, err := gpu.NewWindowSession(n, win, sink)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	for k, tok := range toks {
		cum, err := gs.Feed(tok)
		if err != nil {
			t.Fatal(err)
		}
		if h := tableHash(cum); h != want[k] {
			t.Fatalf("gpu step, position %d: table %s, cpu %s", k, h, want[k])
		}
	}
	t.Logf("%s step: %d positions match in %.1f s", gpu.Device(), n, time.Since(t0).Seconds())

	t0 = time.Now()
	intForceSlow = os.Getenv("INTCROSS_SLOW") != "" // the fallback attention (int_attnb*) for every batch
	defer func() { intForceSlow = false }()
	bs, err := gpu.NewWindowSession(n, win, sink)
	if err != nil {
		t.Fatal(err)
	}
	defer bs.Close()
	for k := 0; k < n; k += chunk {
		e := min(k+chunk, n)
		got, err := bs.FeedBatch(toks[k:e], ids[k:e])
		if err != nil {
			t.Fatal(err)
		}
		for i, c := range got {
			if c != wantC[k+i] {
				t.Fatalf("gpu batch, position %d: %+v, cpu %+v", k+i, c, wantC[k+i])
			}
		}
	}
	t.Logf("%s batch: %d positions match in %.1f s (slow batches %d)", gpu.Device(), n, time.Since(t0).Seconds(),
		intBatchSlow)
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// TestIntHashes prints the CPU's table hashes ("k hash", as intref.py idhashes does) for INTCROSS_IDS, INTCROSS_N
// positions and INTCROSS_WIN / INTCROSS_SINK, to compare with the reference.
func TestIntHashes(t *testing.T) {
	if os.Getenv("INTCROSS_HASHES") == "" {
		t.Skip("set INTCROSS_HASHES=1")
	}
	size := cmpOr(os.Getenv("INTMODEL"), "0.5B")
	ids := vecIDs(t, cmpOr(os.Getenv("INTCROSS_IDS"), "slice.ids.json"))
	n := min(envInt("INTCROSS_N", 40), len(ids))
	m := loadIntDev(t, size, "off")
	s, err := m.NewWindowSession(n, envInt("INTCROSS_WIN", 0), envInt("INTCROSS_SINK", 0))
	if err != nil {
		t.Fatal(err)
	}
	prev := m.BOS()
	for k := 0; k < n; k++ {
		cum, err := s.Feed(prev)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(k, tableHash(cum))
		prev = ids[k]
	}
}
