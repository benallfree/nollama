package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/models"
)

// convert: make a converted model for pack/unpack from the official Qwen2.5 weights. The conversion is
// deterministic, so the result must have the manifest's SHA-256 (it is refused otherwise).

func cmdConvert(args []string) error {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	profile := fs.String("profile", "", "good (0.5B), better (1.5B) or best (7B): download the official weights and convert")
	keep := fs.Bool("keep-source", false, "keep the downloaded official weights")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: nollama convert --profile good|better|best    (downloads the official weights)")
		fmt.Fprintln(os.Stderr, "       nollama convert <snapshot dir>               (a local copy: config.json, tokenizer.json, *.safetensors)")
		fs.PrintDefaults()
	}
	fs.Parse(args)
	var src string
	var want *engine.IntModelInfo
	switch {
	case *profile != "" && fs.NArg() == 0:
		mi, ok := engine.IntModelByProfile(*profile)
		if !ok {
			return fmt.Errorf("unknown profile %q", *profile)
		}
		want = &mi
		dir, err := fetchIntSource(mi)
		if err != nil {
			return err
		}
		src = dir
		if !*keep {
			defer os.RemoveAll(dir)
		}
	case *profile == "" && fs.NArg() == 1:
		src = fs.Arg(0)
	default:
		fs.Usage()
		return errors.New("give --profile or a snapshot directory")
	}
	if err := os.MkdirAll(modelsDir(), 0o755); err != nil {
		return err
	}
	tmp := filepath.Join(modelsDir(), fmt.Sprintf("intcodec-convert-%d.gguf", os.Getpid()))
	defer os.Remove(tmp)
	t0 := time.Now()
	err := engine.ConvertIntModel(src, tmp, func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "\rconvert: "+format+"   ", a...)
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	h, err := engine.FileSHA256(tmp)
	if err != nil {
		return err
	}
	var mi *engine.IntModelInfo
	for _, m := range engine.IntManifest() {
		if m.Bundle.SHA256 == h {
			m := m
			mi = &m
		}
	}
	if mi == nil || (want != nil && mi.ID != want.ID) {
		return fmt.Errorf("the converted file's sha256 %s is not the manifest's%s; not installed", h,
			map[bool]string{true: " for " + *profile, false: ""}[want != nil])
	}
	dst := filepath.Join(modelsDir(), mi.Bundle.File)
	if err := os.Rename(tmp, dst); err != nil {
		return err
	}
	fmt.Printf("%s (model %d, %s): sha256 checked, %.0f s\n", dst, mi.ID, mi.Profile, time.Since(t0).Seconds())
	return nil
}

// fetchIntSource downloads the manifest's source files (the official repo at the pinned revision).
func fetchIntSource(mi engine.IntModelInfo) (string, error) {
	dir := filepath.Join(modelsDir(), "intcodec-src", filepath.Base(mi.Source.Repo)+"@"+mi.Source.Revision[:12])
	for _, f := range mi.Source.Files {
		src := models.Source{Name: mi.Source.Repo + "/" + f, Path: filepath.Join(dir, f),
			URL: "https://huggingface.co/" + mi.Source.Repo + "/resolve/" + mi.Source.Revision + "/" + f}
		var last time.Time
		err := models.Fetch(src, nil, func(done, total int64) {
			if time.Since(last) > time.Second || done == total {
				last = time.Now()
				fmt.Fprintf(os.Stderr, "\r%s: %s / %s   ", f, size(done), size(total))
			}
		})
		fmt.Fprintln(os.Stderr)
		if err != nil {
			return "", err
		}
	}
	return dir, nil
}
