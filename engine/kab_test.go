package engine

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// KAB=1 go test -run KernelAB -v: single-thread throughput of the prompt
// kernels, old (4-column) and new, on one L1/L2-resident weight tile. Every
// trial runs every kernel once and each keeps its fastest trial, so all of
// them see the same machine conditions (frequency, other load).
func TestKernelAB(t *testing.T) {
	if os.Getenv("KAB") == "" {
		t.Skip("KAB not set")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r := rand.New(rand.NewPCG(1, 2))
	rows, cols, reps, trials := 32, 2048, 10, 400
	if n, err := strconv.Atoi(os.Getenv("KABROWS")); err == nil { // a weight set bigger than L2: read from memory
		rows, reps, trials = n, 1, 20
	}
	nb := cols / qk
	type kern struct {
		name string
		run  func()
	}
	toks := map[string]int{} // tokens per call, when not 4
	var ks []kern
	acts := func() (plain [4][]byte, k4, k6 []byte) {
		k4, k6 = make([]byte, nb*q8k4Bytes), make([]byte, nb*q8k6Bytes)
		for j := 0; j < 4; j++ {
			x := make([]float32, cols)
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
			plain[j] = make([]byte, nb*q8kBytes)
			quantizeQ8K(x, plain[j])
			packQ8K4(plain[j], k4, j, nb)
			packQ8K6(plain[j], k6, j, nb)
		}
		return
	}
	plain, k4, k6 := acts()
	y := make([]byte, 0, 4*nb*q8kBytes)
	for j := range plain {
		y = append(y, plain[j]...)
	}
	var o4 [4]float32
	var o8 [8]float32
	for _, k := range []struct {
		name string
		wb   int
		x4   func(x, y []byte, stride, nb int, out *[4]float32)
	}{{"q4k x4 (old)", q4kBytes, dotQ4Kx4}, {"q6k x4 (old)", q6kBytes, dotQ6Kx4}} {
		if k.x4 == nil {
			continue
		}
		w, rb, x4 := randBlocks(r, rows*nb, k.wb, []int{0}), nb*k.wb, k.x4
		ks = append(ks, kern{k.name, func() {
			for j := 0; j < rows; j++ {
				x4(w[j*rb:], y, nb*q8kBytes, nb, &o4)
			}
		}})
	}
	if gemmQ4K2x4 != nil {
		w := randBlocks(r, rows*nb, q4kBytes, []int{0})
		rb, sb := nb*q4kBytes, nb*q4kScaleBytes
		sc := make([]byte, rows*sb)
		for j := 0; j < rows; j++ {
			q4kScales(w[j*rb:], sc[j*sb:], nb)
		}
		ks = append(ks, kern{"q4k 2x4 (new)", func() {
			for j := 0; j < rows; j += 2 {
				gemmQ4K2x4(w[j*rb:], w[(j+1)*rb:], sc[j*sb:], sc[(j+1)*sb:], k4, nb, &o8)
			}
		}})
	}
	if gemmQ4K2x4Plain != nil {
		w := randBlocks(r, rows*nb, q4kBytes, []int{0})
		rb, sb := nb*q4kBytes, nb*q4kScaleSize
		sc := make([]byte, rows*sb)
		for j := 0; j < rows; j++ {
			q4kScalesPlain(w[j*rb:], sc[j*sb:], nb)
		}
		ks = append(ks, kern{"q4k 2x4 plain (new)", func() {
			for j := 0; j < rows; j += 2 {
				gemmQ4K2x4Plain(w[j*rb:], w[(j+1)*rb:], sc[j*sb:], sc[(j+1)*sb:], y, nb*q8kBytes, nb, &o8)
			}
		}})
	}
	if gemmQ6K2x4Plain != nil {
		w := randBlocks(r, rows*nb, q6kBytes, []int{208})
		rb, sb := nb*q6kBytes, nb*q6kScaleSize
		sc := make([]byte, rows*sb)
		for j := 0; j < rows; j++ {
			q6kScales(w[j*rb:], sc[j*sb:], nb)
		}
		ks = append(ks, kern{"q6k 2x4 plain (new)", func() {
			for j := 0; j < rows; j += 2 {
				gemmQ6K2x4Plain(w[j*rb:], w[(j+1)*rb:], sc[j*sb:], sc[(j+1)*sb:], y, nb*q8kBytes, nb, &o8)
			}
		}})
	}
	if gemmQ6K1x4 != nil {
		w, rb := randBlocks(r, rows*nb, q6kBytes, []int{208}), nb*q6kBytes
		ks = append(ks, kern{"q6k 1x4 (new)", func() {
			for j := 0; j < rows; j++ {
				gemmQ6K1x4(w[j*rb:], k6, nb, &o4)
			}
		}})
	}
	if dotF16x4 != nil {
		w := make([]byte, rows*cols*2)
		for i := 0; i < rows*cols; i++ {
			h := uint16(0x3000 + r.IntN(0x1000))
			w[2*i], w[2*i+1] = byte(h), byte(h>>8)
		}
		yf := make([]float32, 4*cols)
		for i := range yf {
			yf[i] = float32(r.NormFloat64())
		}
		rb := cols * 2
		ks = append(ks, kern{"f16 x4 (old)", func() {
			for j := 0; j < rows; j++ {
				dotF16x4(w[j*rb:(j+1)*rb], yf, cols, &o4)
			}
		}})
		if gemmF16x2x4 != nil {
			ks = append(ks, kern{"f16 2x4 (new)", func() {
				for j := 0; j < rows; j += 2 {
					gemmF16x2x4(w[j*rb:(j+1)*rb], w[(j+1)*rb:(j+2)*rb], yf, cols, &o8)
				}
			}})
		}
	}
	for _, k := range kab4x16(r, rows, nb) {
		ks, toks[k.name] = append(ks, kern{k.name, k.run}), k.toks
	}
	best := make([]time.Duration, len(ks))
	for i := range best {
		best[i] = 1 << 62
	}
	for trial := 0; trial < trials; trial++ {
		for i, k := range ks {
			t0 := time.Now()
			for rep := 0; rep < reps; rep++ {
				k.run()
			}
			best[i] = min(best[i], time.Since(t0))
		}
	}
	for i, k := range ks {
		n := max(toks[k.name], 4)
		fmt.Printf("%-14s %6.1f GMAC/s\n", k.name, float64(reps*rows*cols*n)/best[i].Seconds()/1e9)
	}
}
