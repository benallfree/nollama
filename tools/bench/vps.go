package main

// bench vps: what a small server sees. Per model it measures time to the first token from a fresh process, both with
// the model file cold on disk (page cache dropped, as root) and warm in the page cache; then, with the model loaded,
// time to first token and reading/writing speed for a short and a long prompt, and the process's peak RAM. It
// optionally runs Ollama on the same GGUF files the same way. Output: JSON (-json) and a markdown table (stdout).
//
//	bench vps -models qwen3-0.6b:Q8_0=/path/a.gguf,qwen3-0.6b:Q4_K_M=qwen3:0.6b [-ollama] [-threads N] [-gen 64] [-runs 3]

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/benallfree/nollama/engine"
)

// System describes the machine a run happened on.
type System struct {
	Host      string  `json:"host"`
	OS        string  `json:"os"`
	Arch      string  `json:"arch"`
	CPU       string  `json:"cpu"`
	CPUFlags  string  `json:"cpu_flags"` // the SIMD ones that matter here
	VCPUs     int     `json:"vcpus"`
	RAMMB     int64   `json:"ram_mb"`
	SwapMB    int64   `json:"swap_mb"`
	Disk      string  `json:"disk"`
	Kernel    string  `json:"kernel"`
	Uptime    string  `json:"uptime"`
	Nollama   string  `json:"nollama"`
	Ollama    string  `json:"ollama,omitempty"`
	Threads   int     `json:"threads"`
	Gen       int     `json:"gen_tokens"`
	Runs      int     `json:"runs"`
	TrulyCold bool    `json:"truly_cold"` // page cache dropped before each cold start
	StartedAt string  `json:"started_at"`
	DurationS float64 `json:"duration_s"`
	Label     string  `json:"label,omitempty"` // e.g. the cloud plan
}

// Phase is one prompt size with the model already loaded (medians over runs).
type Phase struct {
	PromptTokens int     `json:"prompt_tokens"`
	OutTokens    int     `json:"out_tokens"`
	TTFT         float64 `json:"first_token_s"`
	Prefill      float64 `json:"prefill_tok_s"`
	Decode       float64 `json:"decode_tok_s"`
}

// VPSRow is one model on one engine.
type VPSRow struct {
	Model     string  `json:"model"`
	Quant     string  `json:"quant"`
	Engine    string  `json:"engine"`
	File      string  `json:"file"`
	FileMB    int64   `json:"file_mb"`
	Kernels   string  `json:"kernels,omitempty"`
	ColdTTFT  float64 `json:"cold_first_token_s"` // fresh process (or freshly loaded Ollama model), page cache dropped
	ColdLoad  float64 `json:"cold_load_s"`
	WarmTTFT  float64 `json:"warm_first_token_s"` // same, file in page cache
	WarmLoad  float64 `json:"warm_load_s"`
	Short     Phase   `json:"short"`
	Long      Phase   `json:"long"`
	PeakRSSMB int64   `json:"peak_rss_mb"`
	Skipped   string  `json:"skipped,omitempty"`
	Error     string  `json:"error,omitempty"`
}

type VPSReport struct {
	System System   `json:"system"`
	Rows   []VPSRow `json:"rows"`
}

// ---- child process: one fresh start of the engine ----

type vpsChild struct {
	Load       float64 `json:"load_s"`
	FirstToken int64   `json:"first_token_unix_ns"` // of the first (unmeasured) short prompt
	Short      Phase   `json:"short"`
	Long       Phase   `json:"long"`
	MaxRSS     int64   `json:"max_rss"`
	Kernels    string  `json:"kernels"`
	Quant      string  `json:"quant"`
}

// genTimed streams one answer and returns the wall time to its first token.
func genTimed(m *engine.Model, p string, gen int) (engine.Result, time.Duration, time.Time, error) {
	t0 := time.Now()
	var first time.Time
	res, err := m.Generate(context.Background(), p, engine.GenerateOptions{MaxTokens: gen, IgnoreEOS: true}, func(string) bool {
		if first.IsZero() {
			first = time.Now()
		}
		return true
	})
	if first.IsZero() {
		first = t0.Add(res.Prefill)
	}
	return res, first.Sub(t0), first, err
}

func cmdVPSRun(args []string) error {
	fs := flag.NewFlagSet("vpsrun", flag.ExitOnError)
	gguf := fs.String("gguf", "", "model file")
	threads := fs.Int("threads", 0, "threads (0: engine default)")
	gen := fs.Int("gen", 128, "tokens to generate per measured run")
	runs := fs.Int("runs", 3, "measured runs per prompt")
	coldOnly := fs.Bool("cold-only", false, "stop after the first token")
	fs.Parse(args)
	t0 := time.Now()
	// sized like `nollama serve`'s defaults: a 256 MB prompt cache (the KV state grows with each answer, so the
	// 4096-token default window costs nothing until it's used)
	m, err := engine.Load(*gguf, engine.LoadOptions{Threads: *threads, PromptCache: 256 << 20})
	if err != nil {
		return err
	}
	defer m.Close()
	info := m.Info()
	c := vpsChild{Load: time.Since(t0).Seconds(), Kernels: info.Kernels, Quant: info.Quant}
	n := 8
	if !*coldOnly {
		n = *gen
	}
	res, _, first, err := genTimed(m, promptText("short", 0), n)
	if err != nil {
		return err
	}
	c.FirstToken = first.UnixNano()
	logf("  first answer: %d prompt tokens, first token after %.2fs in this process", res.PromptTokens, first.Sub(t0).Seconds())
	if !*coldOnly {
		run := 1
		for _, pn := range []string{"short", "long"} {
			var ttft, pre, dec []float64
			var pt, ot []int
			for i := 0; i < *runs; i++ {
				res, tt, _, err := genTimed(m, promptText(pn, run), *gen)
				run++
				if err != nil {
					return err
				}
				pt, ot = append(pt, res.PromptTokens), append(ot, res.OutputTokens)
				ttft = append(ttft, tt.Seconds())
				pre = append(pre, float64(res.PromptTokens)/res.Prefill.Seconds())
				dec = append(dec, float64(res.OutputTokens-1)/res.Decode.Seconds())
				logf("  %s %d/%d: %d prompt tokens, first token %.2fs, read %.1f tok/s, wrote %.1f tok/s", pn, i+1, *runs,
					res.PromptTokens, tt.Seconds(), pre[len(pre)-1], dec[len(dec)-1])
			}
			ph := Phase{medianInt(pt), medianInt(ot), median(ttft), median(pre), median(dec)}
			if pn == "short" {
				c.Short = ph
			} else {
				c.Long = ph
			}
		}
	}
	c.MaxRSS = maxRSS()
	return json.NewEncoder(os.Stdout).Encode(c)
}

// ---- parent ----

func readFile(p string) string { b, _ := os.ReadFile(p); return strings.TrimSpace(string(b)) }

func run(name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, _ := exec.CommandContext(ctx, name, args...).Output()
	return strings.TrimSpace(string(out))
}

func meminfo(key string) int64 {
	for _, l := range strings.Split(readFile("/proc/meminfo"), "\n") {
		if f := strings.Fields(l); len(f) >= 2 && f[0] == key+":" {
			kb, _ := strconv.ParseInt(f[1], 10, 64)
			return kb >> 10
		}
	}
	return 0
}

func nollamaVersion() string {
	v := "dev"
	if bi, ok := debug.ReadBuildInfo(); ok {
		var rev, dirty string
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				rev = s.Value
			case "vcs.modified":
				if s.Value == "true" {
					dirty = "-dirty"
				}
			}
		}
		if len(rev) > 12 {
			rev = rev[:12]
		}
		if rev != "" {
			v = rev + dirty
		}
	}
	return v
}

func systemInfo() System {
	s := System{OS: runtime.GOOS, Arch: runtime.GOARCH, VCPUs: runtime.NumCPU(), Nollama: nollamaVersion()}
	s.Host, _ = os.Hostname()
	s.Kernel = run("uname", "-r")
	s.Uptime = run("uptime")
	if runtime.GOOS != "linux" {
		s.CPU = run("sysctl", "-n", "machdep.cpu.brand_string")
		if n, err := strconv.ParseInt(run("sysctl", "-n", "hw.memsize"), 10, 64); err == nil {
			s.RAMMB = n >> 20
		}
		return s
	}
	var flags []string
	for _, l := range strings.Split(readFile("/proc/cpuinfo"), "\n") {
		k, v, ok := strings.Cut(l, ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch {
		case s.CPU == "" && (k == "model name" || k == "Model" || k == "CPU part"):
			s.CPU = v
		case flags == nil && (k == "flags" || k == "Features"):
			for _, f := range strings.Fields(v) {
				switch f {
				case "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx_vnni", "avx512_bf16", "asimddp", "sve", "i8mm", "bf16":
					flags = append(flags, f)
				}
			}
		}
	}
	s.CPUFlags = strings.Join(flags, " ")
	s.RAMMB, s.SwapMB = meminfo("MemTotal"), meminfo("SwapTotal")
	// the device under / and whether it spins
	src := run("findmnt", "-no", "SOURCE", "/")
	if src != "" {
		s.Disk = src + " " + strings.Join(strings.Fields(run("lsblk", "-dno", "NAME,SIZE,ROTA,TYPE,MODEL", "-s", src)), " ")
	}
	return s
}

func dropCaches() error {
	exec.Command("sync").Run()
	return os.WriteFile("/proc/sys/vm/drop_caches", []byte("3\n"), 0)
}

type vpsModel struct{ name, quant, spec, path string }

func parseVPSModels(s string) ([]vpsModel, error) {
	var out []vpsModel
	for _, item := range strings.Split(s, ",") {
		label, spec, ok := strings.Cut(strings.TrimSpace(item), "=")
		if !ok {
			return nil, fmt.Errorf("model %q: want name:QUANT=path.gguf or name:QUANT=ollama:tag", item)
		}
		name, quant, _ := strings.Cut(label, ":")
		path, err := resolveModel(spec)
		if err != nil {
			return nil, fmt.Errorf("%s: %v", spec, err)
		}
		out = append(out, vpsModel{name, quant, spec, path})
	}
	return out, nil
}

func cmdVPS(args []string) error {
	fs := flag.NewFlagSet("vps", flag.ExitOnError)
	modelsFlag := fs.String("models", "", "comma-separated name:QUANT=spec; spec is a .gguf path or an Ollama tag")
	threads := fs.Int("threads", runtime.NumCPU(), "threads for both engines")
	gen := fs.Int("gen", 128, "tokens generated per measured run")
	runs := fs.Int("runs", 3, "measured runs per prompt (medians)")
	withOllama := fs.Bool("ollama", false, "also run Ollama (needs it serving at OLLAMA_URL) on the same files")
	jsonOut := fs.String("json", "", "write the JSON report here")
	label := fs.String("label", "", "a label for the machine, e.g. the cloud plan")
	ramMargin := fs.Int64("ram-margin-mb", 350, "skip a model whose file plus this doesn't fit in RAM")
	fs.Parse(args)
	ms, err := parseVPSModels(*modelsFlag)
	if err != nil {
		return err
	}
	start := time.Now()
	rep := VPSReport{System: systemInfo()}
	rep.System.Threads, rep.System.Gen, rep.System.Runs, rep.System.Label = *threads, *gen, *runs, *label
	rep.System.StartedAt = start.UTC().Format(time.RFC3339)
	rep.System.TrulyCold = dropCaches() == nil
	if !rep.System.TrulyCold {
		logf("can't drop the page cache (not root, or not Linux): cold starts are only fresh processes")
	}
	if *withOllama {
		rep.System.Ollama = run("ollama", "--version")
	}
	logf("%s, %d vCPUs, %d MB RAM, %s; nollama %s, %d threads", rep.System.CPU, rep.System.VCPUs, rep.System.RAMMB,
		rep.System.Kernel, rep.System.Nollama, *threads)
	save := func() {
		rep.System.DurationS = time.Since(start).Seconds()
		if *jsonOut != "" {
			b, _ := json.MarshalIndent(rep, "", "  ")
			os.WriteFile(*jsonOut, b, 0o644)
		}
	}
	exe, _ := os.Executable()
	for _, m := range ms {
		st, err := os.Stat(m.path)
		if err != nil {
			return err
		}
		base := VPSRow{Model: m.name, Quant: m.quant, File: m.spec, FileMB: st.Size() >> 20}
		fits := rep.System.RAMMB == 0 || base.FileMB+*ramMargin <= rep.System.RAMMB
		r := base
		r.Engine = "nollama"
		if !fits {
			r.Skipped = fmt.Sprintf("model file %d MB doesn't fit in %d MB RAM", base.FileMB, rep.System.RAMMB)
			logf("%s %s: %s", m.name, m.quant, r.Skipped)
		} else {
			logf("nollama %s %s (%d MB)", m.name, m.quant, base.FileMB)
			if err := vpsNollama(exe, m.path, *threads, *gen, *runs, rep.System.TrulyCold, &r); err != nil {
				r.Error = err.Error()
				logf("  error: %v", err)
			}
		}
		rep.Rows = append(rep.Rows, r)
		save()
		if *withOllama {
			o := base
			o.Engine = "ollama"
			o.Skipped = r.Skipped
			if fits {
				logf("ollama %s %s", m.name, m.quant)
				if err := vpsOllama(m, *threads, *gen, *runs, rep.System.TrulyCold, &o); err != nil {
					o.Error = err.Error()
					logf("  error: %v", err)
				}
			}
			rep.Rows = append(rep.Rows, o)
			save()
		}
	}
	save()
	printVPSTable(os.Stdout, rep)
	return nil
}

func vpsChildRun(exe, path string, threads, gen, runs int, coldOnly bool) (vpsChild, time.Time, error) {
	args := []string{"vpsrun", "-gguf", path, "-threads", strconv.Itoa(threads), "-gen", strconv.Itoa(gen), "-runs", strconv.Itoa(runs)}
	if coldOnly {
		args = append(args, "-cold-only")
	}
	cmd := exec.Command(exe, args...)
	cmd.Stderr = os.Stderr
	var c vpsChild
	t0 := time.Now()
	out, err := cmd.Output()
	if err != nil {
		return c, t0, fmt.Errorf("vpsrun: %v", err)
	}
	return c, t0, json.Unmarshal(out, &c)
}

func vpsNollama(exe, path string, threads, gen, runs int, canDrop bool, r *VPSRow) error {
	if canDrop {
		if err := dropCaches(); err != nil {
			return err
		}
	}
	c, t0, err := vpsChildRun(exe, path, threads, gen, runs, true)
	if err != nil {
		return err
	}
	r.ColdTTFT, r.ColdLoad = time.Unix(0, c.FirstToken).Sub(t0).Seconds(), c.Load
	logf("  cold start: first token %.2fs after exec (load %.2fs)", r.ColdTTFT, r.ColdLoad)
	c, t0, err = vpsChildRun(exe, path, threads, gen, runs, false)
	if err != nil {
		return err
	}
	r.WarmTTFT, r.WarmLoad = time.Unix(0, c.FirstToken).Sub(t0).Seconds(), c.Load
	r.Short, r.Long, r.PeakRSSMB, r.Kernels = c.Short, c.Long, c.MaxRSS>>20, c.Kernels
	if c.Quant != "" {
		r.Quant = c.Quant // what the file says, over the label
	}
	logf("  warm start: first token %.2fs after exec; loaded: short %.2fs, long %.2fs; peak RSS %d MB",
		r.WarmTTFT, r.Short.TTFT, r.Long.TTFT, r.PeakRSSMB)
	return nil
}

// ---- Ollama ----

type ollamaStream struct {
	TTFT time.Duration
	ollamaResp
}

// ollamaGenStream streams one answer and times its first non-empty piece on the client's clock.
func ollamaGenStream(model, prompt string, opts map[string]any, timeout time.Duration) (*ollamaStream, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "prompt": prompt, "raw": true, "stream": true,
		"options": opts, "keep_alive": "30m"})
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", ollamaURL+"/api/generate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	t0 := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out ollamaStream
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r ollamaResp
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("ollama: %s", sc.Bytes())
		}
		if r.Error != "" {
			return nil, errors.New("ollama: " + r.Error)
		}
		if out.TTFT == 0 && (r.Response != "" || r.Done) {
			out.TTFT = time.Since(t0)
		}
		if r.Done {
			out.ollamaResp = r
			return &out, nil
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return nil, io.ErrUnexpectedEOF
}

// ollamaEnsure makes the model available to Ollama: a tag as is, a GGUF file imported as nb-<name>-<quant>.
func ollamaEnsure(m vpsModel) (string, error) {
	if !strings.HasSuffix(m.spec, ".gguf") && !strings.HasPrefix(m.spec, "/") {
		return m.spec, nil
	}
	tag := strings.ToLower(strings.NewReplacer(".", "-", "_", "-").Replace("nb-" + m.name + "-" + m.quant))
	if out := run("ollama", "show", tag, "--modelfile"); out != "" {
		return tag, nil
	}
	mf, err := os.CreateTemp("", "Modelfile")
	if err != nil {
		return "", err
	}
	defer os.Remove(mf.Name())
	fmt.Fprintf(mf, "FROM %s\n", m.path)
	mf.Close()
	logf("  importing %s into Ollama as %s", m.path, tag)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ollama", "create", tag, "-f", mf.Name())
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	return tag, cmd.Run()
}

func vpsOllama(m vpsModel, threads, gen, runs int, canDrop bool, r *VPSRow) error {
	tag, err := ollamaEnsure(m)
	if err != nil {
		return err
	}
	opts := func(n int) map[string]any {
		o := greedyOpts(n)
		o["num_gpu"], o["num_thread"], o["num_ctx"] = 0, threads, 4096
		return o
	}
	var peak int64
	done := make(chan struct{})
	defer close(done)
	go func() {
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				peak = max(peak, runnerRSS())
			}
		}
	}()
	start := func(cold bool) (*ollamaStream, error) {
		ollamaUnloadAll([]string{tag})
		if cold && canDrop {
			if err := dropCaches(); err != nil {
				return nil, err
			}
		}
		return ollamaGenStream(tag, promptText("short", 0), opts(8), 15*time.Minute)
	}
	s, err := start(true)
	if err != nil {
		return err
	}
	r.ColdTTFT, r.ColdLoad = s.TTFT.Seconds(), float64(s.LoadDuration)/1e9
	logf("  cold start: first token %.2fs after the request (load %.2fs)", r.ColdTTFT, r.ColdLoad)
	if s, err = start(false); err != nil {
		return err
	}
	r.WarmTTFT, r.WarmLoad = s.TTFT.Seconds(), float64(s.LoadDuration)/1e9
	run := 1
	for _, pn := range []string{"short", "long"} {
		var ttft, pre, dec []float64
		var pt, ot []int
		for i := 0; i < runs; i++ {
			s, err := ollamaGenStream(tag, promptText(pn, run), opts(gen), 15*time.Minute)
			run++
			if err != nil {
				return err
			}
			pt, ot = append(pt, s.PromptEvalCount), append(ot, s.EvalCount)
			ttft = append(ttft, s.TTFT.Seconds())
			pre = append(pre, float64(s.PromptEvalCount)/(float64(s.PromptEvalDuration)/1e9))
			dec = append(dec, float64(s.EvalCount)/(float64(s.EvalDuration)/1e9))
			logf("  %s %d/%d: %d prompt tokens, first token %.2fs, read %.1f tok/s, wrote %.1f tok/s", pn, i+1, runs,
				s.PromptEvalCount, s.TTFT.Seconds(), pre[len(pre)-1], dec[len(dec)-1])
		}
		ph := Phase{medianInt(pt), medianInt(ot), median(ttft), median(pre), median(dec)}
		if pn == "short" {
			r.Short = ph
		} else {
			r.Long = ph
		}
	}
	r.PeakRSSMB = peak >> 20
	ollamaUnloadAll([]string{tag})
	logf("  warm start: first token %.2fs; loaded: short %.2fs, long %.2fs; peak RSS %d MB",
		r.WarmTTFT, r.Short.TTFT, r.Long.TTFT, r.PeakRSSMB)
	return nil
}

// ---- table ----

func secs(s float64) string {
	switch {
	case s == 0:
		return "–"
	case s < 1:
		return fmt.Sprintf("%.0f ms", s*1000)
	case s < 10:
		return fmt.Sprintf("%.1f s", s)
	}
	return fmt.Sprintf("%.0f s", s)
}

func mb(n int64) string {
	if n >= 1024 {
		return fmt.Sprintf("%.1f GB", float64(n)/1024)
	}
	return fmt.Sprintf("%d MB", n)
}

// Machine is the table's first column: the label (e.g. the cloud plan), vCPUs and RAM.
func (s System) Machine() string {
	l := s.Label
	if l == "" {
		l = s.CPU
	}
	return fmt.Sprintf("%s: %d vCPU, %s", l, s.VCPUs, mb(s.RAMMB))
}

func printVPSTable(w io.Writer, rep VPSReport) {
	s := rep.System
	cold := "page cache dropped"
	if !s.TrulyCold {
		cold = "page cache NOT dropped"
	}
	fmt.Fprintf(w, "%s: %s (%s), %s, kernel %s. nollama %s", s.Machine(), s.CPU, s.CPUFlags, s.Disk, s.Kernel, s.Nollama)
	if s.Ollama != "" {
		fmt.Fprintf(w, ", %s", s.Ollama)
	}
	fmt.Fprintf(w, ". %d threads, %d tokens written per run, medians of %d runs. Cold: fresh start, %s. Load average before: %s\n",
		s.Threads, s.Gen, s.Runs, cold, s.Uptime)
	for _, eng := range []string{"nollama", "ollama"} {
		var rows []VPSRow
		for _, r := range rep.Rows {
			if r.Engine == eng {
				rows = append(rows, r)
			}
		}
		if len(rows) == 0 {
			continue
		}
		fmt.Fprintf(w, "\n%s:\n\n", eng)
		fmt.Fprintln(w, "| Machine | Model | Quant | Cold first token | Warm first token | First token, loaded (short / long) | Reading tok/s (short / long) | Writing tok/s | Peak RAM |")
		fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|")
		for _, r := range rows {
			if r.Skipped != "" || r.Error != "" {
				fmt.Fprintf(w, "| %s | %s | %s | %s ||||||\n", s.Machine(), r.Model, r.Quant, orDash(r.Skipped+r.Error))
				continue
			}
			fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s / %s | %.0f / %.0f | %.1f | %s |\n", s.Machine(), r.Model, r.Quant,
				secs(r.ColdTTFT), secs(r.WarmTTFT), secs(r.Short.TTFT), secs(r.Long.TTFT), r.Short.Prefill, r.Long.Prefill,
				r.Short.Decode, mb(r.PeakRSSMB))
		}
	}
}

func orDash(s string) string {
	if s == "" {
		return "–"
	}
	return s
}

func cmdVPSTable(args []string) error {
	for _, p := range args {
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		var rep VPSReport
		if err := json.Unmarshal(b, &rep); err != nil {
			return fmt.Errorf("%s: %v", p, err)
		}
		printVPSTable(os.Stdout, rep)
		fmt.Println()
	}
	return nil
}
