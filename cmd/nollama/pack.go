package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/benallfree/nollama/engine"
)

// pack and unpack: the integer codec. An archive (.13k) names its converted model by id; the converted models
// (GGUF, "qwen2-intcodec") live in the models directory under the manifest's file names.

func intModelFile(mi engine.IntModelInfo, override string) (string, error) {
	if override != "" {
		return override, nil
	}
	p := filepath.Join(modelsDir(), mi.Bundle.File)
	if _, err := os.Stat(p); err != nil {
		return "", fmt.Errorf("model %d (%s, from %s) is not in %s yet: run nollama convert --profile %s "+
			"(downloads the official weights once and converts them), or pass --model", mi.ID, mi.Profile, mi.Source.Repo,
			modelsDir(), mi.Profile)
	}
	return p, nil
}

type intProgress struct {
	what  string
	start time.Time
	last  time.Time
}

func (p *intProgress) update(done, total int) {
	now := time.Now()
	if done < total && now.Sub(p.last) < time.Second {
		return
	}
	p.last = now
	el := now.Sub(p.start).Seconds()
	rate := float64(done) / max(el, 1e-9)
	eta := time.Duration(float64(total-done)/max(rate, 1e-9)) * time.Second
	fmt.Fprintf(os.Stderr, "\r%s: %d/%d tokens, %.1f tok/s, %s left   ", p.what, done, total, rate, eta.Round(time.Second))
	if done == total {
		fmt.Fprintln(os.Stderr)
	}
}

func readInput(path string) ([]byte, error) {
	if path == "" || path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(path)
}

func writeOutput(path string, b []byte) error {
	if path == "-" {
		_, err := os.Stdout.Write(b)
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func cmdPack(args []string) error {
	fs := flag.NewFlagSet("pack", flag.ExitOnError)
	profile := fs.String("profile", "best", "good (0.5B), better (1.5B) or best (7B)")
	model := fs.String("model", "", "path to the converted model (default: the profile's file in the models directory)")
	out := fs.String("o", "", "output file (default: the input file + .13k, or stdout when reading stdin)")
	threads := fs.Int("threads", 0, "threads (default: all CPUs)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: nollama pack [--profile good|better|best] [--model file.gguf] [-o out.13k] [file]")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	in := fs.Arg(0)
	mi, ok := engine.IntModelByProfile(*profile)
	if !ok {
		return fmt.Errorf("unknown profile %q", *profile)
	}
	path, err := intModelFile(mi, *model)
	if err != nil {
		return err
	}
	text, err := readInput(in)
	if err != nil {
		return err
	}
	m, err := engine.LoadIntModel(path, *threads)
	if err != nil {
		return err
	}
	defer m.Close()
	if m.ID() == 0 {
		return fmt.Errorf("%s is not in the model manifest (archives name their model by id)", filepath.Base(path))
	}
	p := &intProgress{what: "pack", start: time.Now()}
	a, err := m.Pack(text, p.update)
	if err != nil {
		return err
	}
	dst := *out
	if dst == "" {
		dst = "-"
		if in != "" && in != "-" {
			dst = in + ".13k"
		}
	}
	if err := writeOutput(dst, a); err != nil {
		return err
	}
	if dst != "-" {
		fmt.Fprintf(os.Stderr, "%s: %d -> %d bytes (%s, %.1f s)\n", dst, len(text), len(a), mi.Profile,
			time.Since(p.start).Seconds())
	}
	return nil
}

func cmdUnpack(args []string) error {
	fs := flag.NewFlagSet("unpack", flag.ExitOnError)
	model := fs.String("model", "", "path to the converted model (default: the archive's model in the models directory)")
	out := fs.String("o", "", "output file (default: the input file without .13k, or stdout when reading stdin)")
	threads := fs.Int("threads", 0, "threads (default: all CPUs)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: nollama unpack [--model file.gguf] [-o out] file.13k")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	in := fs.Arg(0)
	a, err := readInput(in)
	if err != nil {
		return err
	}
	id, n, _, err := engine.IntArchiveInfo(a)
	if err != nil {
		return err
	}
	mi, ok := engine.IntModelByID(id)
	if !ok {
		return fmt.Errorf("the archive needs model %d, which this nollama doesn't know (update nollama)", id)
	}
	path, err := intModelFile(mi, *model)
	if err != nil {
		return err
	}
	m, err := engine.LoadIntModel(path, *threads)
	if err != nil {
		return err
	}
	defer m.Close()
	p := &intProgress{what: "unpack", start: time.Now()}
	text, err := m.Unpack(a, p.update)
	if err != nil {
		return err
	}
	dst := *out
	if dst == "" {
		dst = "-"
		if in != "" && in != "-" {
			if !strings.HasSuffix(in, ".13k") {
				return errors.New("pass -o (the input doesn't end in .13k)")
			}
			dst = strings.TrimSuffix(in, ".13k")
		}
	}
	if err := writeOutput(dst, text); err != nil {
		return err
	}
	if dst != "-" {
		fmt.Fprintf(os.Stderr, "%s: %d tokens, %d bytes (%s, %.1f s)\n", dst, n, len(text), mi.Profile,
			time.Since(p.start).Seconds())
	}
	return nil
}
