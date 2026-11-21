package engine

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"testing"
	"time"
)

// minTime runs f trials times and returns the fastest run: on a busy machine
// the minimum of many short runs is far steadier than an average.
func minTime(trials int, f func()) time.Duration {
	best := time.Duration(1 << 62)
	for i := 0; i < trials; i++ {
		t0 := time.Now()
		f()
		if d := time.Since(t0); d < best {
			best = d
		}
	}
	return best
}

// KAB=1 go test -run KernelAB -v: single-thread kernel throughput, min of short runs.
func TestKernelAB(t *testing.T) {
	if os.Getenv("KAB") == "" {
		t.Skip("KAB not set")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r := rand.New(rand.NewPCG(1, 2))
	rows, cols := 64, 2048 // weights stay in L1/L2 like a matmul tile
	nb := cols / qk
	report := func(name string, macs int, d time.Duration) {
		fmt.Printf("%-14s %6.1f GMAC/s  (%v)\n", name, float64(macs)/d.Seconds()/1e9, d)
	}
	for _, k := range []struct {
		name string
		wb   int
		x4   func(x, y []byte, stride, nb int, out *[4]float32)
	}{{"q4k x4", q4kBytes, dotQ4Kx4}, {"q6k x4", q6kBytes, dotQ6Kx4}} {
		if k.x4 == nil {
			continue
		}
		w := randBlocks(r, rows*nb, k.wb, []int{0})
		x := make([]float32, 4*cols)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		y := make([]byte, 4*nb*q8kBytes)
		quantizeQ8K(x, y)
		rb := nb * k.wb
		var o [4]float32
		reps := 20
		d := minTime(300, func() {
			for i := 0; i < reps; i++ {
				for j := 0; j < rows; j++ {
					k.x4(w[j*rb:], y, nb*q8kBytes, nb, &o)
				}
			}
		})
		report(k.name, reps*rows*cols*4, d)
	}
	if gemmQ4K2x4 != nil {
		w := randBlocks(r, rows*nb, q4kBytes, []int{0})
		_, grp := makeQ8K4(r, nb)
		rb, sb := nb*q4kBytes, nb*q4kScaleBytes
		sc := make([]byte, rows*sb)
		for j := 0; j < rows; j++ {
			q4kScales(w[j*rb:], sc[j*sb:], nb)
		}
		var o [8]float32
		reps := 20
		d := minTime(300, func() {
			for i := 0; i < reps; i++ {
				for j := 0; j < rows; j += 2 {
					gemmQ4K2x4(w[j*rb:], w[(j+1)*rb:], sc[j*sb:], sc[(j+1)*sb:], grp, nb, &o)
				}
			}
		})
		report("q4k 2x4", reps*rows*cols*4, d)
	}
}
