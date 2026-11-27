//go:build darwin

package engine

import (
	"encoding/binary"
	"fmt"
	"math/bits"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/benallfree/nollama/engine/internal/metal"
)

// TestAttnProfile: where a compress batch's GPU time goes as the context fills, on real data (the key and value
// caches hold a real file's keys, so the weights' sparsity is real). Feeds ATTNPROF_FILE batch by batch; at each
// checkpoint re-runs the last batch (idempotent: the embedding rewrites h, the caches get the same values) whole and
// per kernel type, best of 3 GPU times.
//
//	NOLLAMA_ATTNPROF=<file>  input text
//	INTMODEL                 model size (0.5B)
//	ATTNPROF_AT              checkpoints, comma-separated positions (2048,8192,16384,24576,32768)
func TestAttnProfile(t *testing.T) {
	file := os.Getenv("NOLLAMA_ATTNPROF")
	if file == "" {
		t.Skip("set NOLLAMA_ATTNPROF=<text file>")
	}
	size := cmpOr(os.Getenv("INTMODEL"), "0.5B")
	src, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	m := loadIntDev(t, size, "auto")
	ids := m.Tokenize(string(src))
	var at []int
	for _, f := range strings.Split(cmpOr(os.Getenv("ATTNPROF_AT"), "2048,8192,16384,24576,32768"), ",") {
		var v int
		fmt.Sscan(f, &v)
		if v <= len(ids) {
			at = append(at, v)
		}
	}
	maxlen := 32768
	gs, err := m.gpu.newState(m, maxlen, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.release()
	toks := append([]int{m.BOS()}, ids...)
	pos := 0
	t0 := time.Now()
	for _, cp := range at {
		for pos < cp {
			P := 512
			if gs.b != nil {
				P = gs.b.P
			}
			n := min(P, cp-pos)
			out := make([]IntCount, n)
			if err := gs.feedBatch(toks[pos:pos+n], toks[pos+1:pos+n+1], pos, out); err != nil {
				t.Fatal(err)
			}
			pos += n
		}
		b := gs.b
		R := b.P
		p0 := pos - R
		gs.g.dev.write(b.params, 0, i32b(R, p0))
		nch := uint((p0+R-1)/intKC + 1)
		list := make([]idisp, 0, len(b.ds)+2*len(b.head))
		for i, d := range b.ds {
			switch b.kind[i] {
			case scoreG:
				d.Groups.H, d.Groups.D = uint((R+b.scoreBR-1)/b.scoreBR), nch
			case pvG:
				d.Groups.H, d.Groups.D = uint((R+b.pvBR-1)/b.pvBR), nch
			}
			list = append(list, d)
		}
		for _, h := range b.head {
			list = append(list, h[0], h[1])
		}
		run := func(ds []idisp) float64 {
			if err := gs.g.dev.run(ds); err != nil {
				t.Fatal(err)
			}
			best := 1e9
			for i := 0; i < 3; i++ {
				gs.g.dev.run(ds)
				best = min(best, float64(gs.g.dev.lastGPU().Microseconds())/1000)
			}
			return best
		}
		full := run(list)
		per := map[string]float64{}
		var names []string
		for _, d := range list {
			if _, ok := per[d.K]; !ok {
				per[d.K] = 0
				names = append(names, d.K)
			}
		}
		attn := 0.0
		line := ""
		for _, k := range names {
			var sub []idisp
			for _, d := range list {
				if d.K == k {
					sub = append(sub, d)
				}
			}
			ms := run(sub)
			if strings.HasPrefix(k, "int_bscores") || strings.HasPrefix(k, "int_bpv") || k == "int_bcomb" {
				attn += ms
			}
			line += fmt.Sprintf(" %s %.1f", k, ms)
		}
		if os.Getenv("ATTNPROF_DELTA") != "" { // the value shifts' spread within 32-key tiles (int_bpv_sg's digits)
			vb := gs.g.dev.host(gs.vc)
			hist := make([]int, 13)
			for l := 0; l < m.L; l++ {
				for j := 0; j < m.nkv; j++ {
					for k0 := 0; k0+32 <= pos; k0 += 32 {
						lo, hi := 99, -1
						for k := k0; k < k0+32; k++ {
							var or uint32
							off := l*gs.kvLayer + 4*((j*maxlen+k)*m.hd)
							for e := 0; e < m.hd; e++ {
								v := int32(binary.LittleEndian.Uint32(vb[off+4*e:]))
								if v < 0 {
									v = -v
								}
								or |= uint32(v)
							}
							s := max(0, bits.Len32(or)-11)
							lo, hi = min(lo, s), max(hi, s)
						}
						hist[hi-lo]++
					}
				}
			}
			t.Logf("value shift spread per 32-key tile (0..12): %v", hist)
		}
		if md, ok := gs.g.dev.(*metalIntDev); ok && os.Getenv("ATTNPROF_AB") != "" {
			// old (gpu_int.metal) vs new (gpu_int_sga.metal) attention kernels in this process, alternating, on the
			// same data: per kernel and for the three attention passes together
			sfx := "64"
			if m.hd == 128 {
				sfx = "128"
			}
			oldPS, err := md.dev.Compile(metalIntSource, "int_bscores"+sfx, "int_bpv"+sfx)
			if err != nil {
				t.Fatal(err)
			}
			newPS := map[string]*metal.Pipeline{"int_bscores" + sfx: md.ps["int_bscores"+sfx], "int_bpv" + sfx: md.ps["int_bpv"+sfx]}
			newBR := b.pvBR
			variant := func(old bool) []idisp {
				br := newBR
				for k := range newPS {
					if old {
						md.ps[k] = oldPS[k]
					} else {
						md.ps[k] = newPS[k]
					}
				}
				if old {
					br = map[int]int{64: 64, 128: 32}[m.hd]
				}
				var sub []idisp
				for i, d := range b.ds {
					if b.kind[i] == pvG {
						d.Groups.H, d.Groups.D = uint((R+br-1)/br), nch
					}
					if b.kind[i] == scoreG {
						d.Groups.H, d.Groups.D = uint((R+b.scoreBR-1)/b.scoreBR), nch
					}
					sub = append(sub, d)
				}
				return sub
			}
			for _, k := range []string{"int_bscores" + sfx, "int_bpv" + sfx, "attention", "batch"} {
				best := [2]float64{1e9, 1e9}
				for rep := 0; rep < 4; rep++ {
					for v := 0; v < 2; v++ {
						all := variant(v == 0)
						var sub []idisp
						for _, d := range all {
							if d.K == k || k == "batch" || (k == "attention" && (strings.HasPrefix(d.K, "int_bscores") || strings.HasPrefix(d.K, "int_bpv") || d.K == "int_bcomb")) {
								sub = append(sub, d)
							}
						}
						if k == "batch" {
							for _, h := range b.head {
								sub = append(sub, h[0], h[1])
							}
						}
						if err := gs.g.dev.run(sub); err != nil {
							t.Fatal(err)
						}
						best[v] = min(best[v], float64(gs.g.dev.lastGPU().Microseconds())/1000)
					}
				}
				t.Logf("A/B ctx %5d %-14s old %8.1f ms  new %8.1f ms  %.2fx", pos, k, best[0], best[1], best[0]/best[1])
			}
			variant(false)
		}
		t.Logf("%s ctx %5d (rows %d..%d): batch %.1f ms = %.0f tok/s; attention %.1f ms (%.0f%%) |%s  [fed in %.0f s]",
			size, pos, p0, pos-1, full, float64(R)/full*1000, attn, 100*attn/full, line, time.Since(t0).Seconds())
	}
}
