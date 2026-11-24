//go:build !windows

// Command transcribe turns WAV files into text with a Whisper model in whisper.cpp's ggml format (experimental).
//
//	go run ./tools/transcribe -m ggml-tiny.bin clip.wav...
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"syscall"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/x/audio"
)

func main() {
	log.SetFlags(0)
	model := flag.String("m", "", "whisper.cpp ggml model file (ggml-tiny.bin, ggml-base.bin, ...)")
	lang := flag.String("l", "", "language code; empty detects it")
	threads := flag.Int("t", 0, "threads (0: GOMAXPROCS capped at 8)")
	translate := flag.Bool("translate", false, "write English instead of transcribing")
	noTS := flag.Bool("notimestamps", false, "decode each 30 s window as plain text, without segment timestamps")
	segs := flag.Bool("segments", false, "print each segment with its start and end")
	runs := flag.Int("runs", 1, "transcribe each file this many times and report the best time")
	flag.Parse()
	if *model == "" || flag.NArg() == 0 {
		log.Fatal("usage: transcribe -m model.bin file.wav...")
	}
	t0 := time.Now()
	w, err := engine.LoadWhisper(*model, engine.WhisperOptions{Threads: *threads})
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()
	info := w.Info()
	fmt.Fprintf(os.Stderr, "loaded %s (%s, %d+%d layers, width %d, %d threads, %s) in %v\n", info.Name, info.Type,
		info.AudioLayers, info.TextLayers, info.Width, info.Threads, info.Kernels, time.Since(t0).Round(time.Millisecond))
	for _, path := range flag.Args() {
		pcm, err := audio.ReadWAVFile(path)
		if err != nil {
			log.Fatal(err)
		}
		dur := time.Duration(len(pcm)) * time.Second / audio.SampleRate
		var best *engine.Transcript
		var bestT time.Duration
		for r := 0; r < *runs; r++ {
			ts := time.Now()
			tr, err := w.Transcribe(context.Background(), pcm, engine.TranscribeOptions{Language: *lang, Translate: *translate, NoTimestamps: *noTS})
			if err != nil {
				log.Fatal(err)
			}
			if el := time.Since(ts); best == nil || el < bestT {
				best, bestT = tr, el
			}
		}
		tm := best.Timings
		if *segs {
			for _, sg := range best.Segments {
				fmt.Printf("  [%6.2f -> %6.2f] %s\n", sg.Start.Seconds(), sg.End.Seconds(), sg.Text)
			}
		}
		fmt.Printf("%s [%s] %s\n", path, best.Language, best.Text)
		fmt.Fprintf(os.Stderr, "  audio %.2fs  total %v (mel %v, encode %v, decode %v for %d tokens, %d windows)  RTF %.3f\n",
			dur.Seconds(), bestT.Round(time.Millisecond), tm.Mel.Round(time.Millisecond), tm.Encode.Round(time.Millisecond),
			tm.Decode.Round(time.Millisecond), tm.Tokens, tm.Windows, bestT.Seconds()/dur.Seconds())
	}
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	maxrss := ru.Maxrss
	if runtime.GOOS == "linux" {
		maxrss *= 1024
	}
	fmt.Fprintf(os.Stderr, "peak RSS %.0f MB\n", float64(maxrss)/(1<<20))
}
