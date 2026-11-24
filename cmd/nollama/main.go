// Command nollama runs language models in one Go binary and answers Ollama's API and OpenAI's.
//
//	nollama serve              answer Ollama's and OpenAI's APIs on 127.0.0.1:11434
//	nollama pull qwen3-1.7b    download a model
//	nollama run qwen3-1.7b     chat with a model in the terminal (or: nollama run qwen3-1.7b "a prompt")
//	nollama list               the models downloaded
//	nollama rm qwen3-1.7b      delete a downloaded model
//	nollama bundle -o myai qwen3-0.6b   a copy of nollama with the model inside, so it downloads nothing
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/benallfree/nollama/catalog"
	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/server"
)

const usage = `nollama runs language models in one binary and answers Ollama's API and OpenAI's.

Usage:
  nollama serve [flags]          answer both APIs (default 127.0.0.1:11434)
  nollama pull <model>           download a model
  nollama run <model> [prompt]   chat in the terminal, or answer one prompt
  nollama list                   the models downloaded
  nollama rm <model>             delete a downloaded model
  nollama bundle -o <file> <model>...
                                 write a copy of nollama (or -exe, any program built
                                 with nollama) with the models inside it
  nollama bundle -check <file>   list the models inside a file and check their SHA-256
  nollama version

Models: %s, hf:<org>/<repo>/<file>.gguf, or a path to a .gguf file.
Ollama spellings work too (qwen3:1.7b).

Environment:
  NOLLAMA_HOST             address serve listens on (default 127.0.0.1:11434)
  NOLLAMA_MODELS           where models are downloaded (default %s)
  NOLLAMA_API_KEY          key serve requires (Authorization: Bearer <key>)
  NOLLAMA_ORIGINS          browser origins serve allows, comma-separated
  NOLLAMA_ALLOWED_HOSTS    host names serve answers to, comma-separated
  NOLLAMA_THREADS          serve's --threads
  NOLLAMA_MAX_MEMORY       serve's --max-memory (such as 3GB)
  NOLLAMA_LOW_PRIORITY     serve's --low-priority (1 or true)
  NOLLAMA_DETERMINISTIC    serve's --deterministic (1 or true)
  NOLLAMA_DRAFT            serve's --draft: a model that guesses tokens (experimental, slower on the CPU today)
`

var version = "dev"

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, usage, catalog.Names(), catalog.DefaultDir())
		os.Exit(2)
	}
	var err error
	switch cmd, args := os.Args[1], os.Args[2:]; cmd {
	case "serve":
		err = cmdServe(args)
	case "pull":
		err = cmdPull(args)
	case "run":
		err = cmdRun(args)
	case "list", "ls":
		err = cmdList(args)
	case "rm":
		err = cmdRm(args)
	case "bundle":
		err = cmdBundle(args)
	case "version", "--version", "-v":
		fmt.Println("nollama", version)
	case "help", "-h", "--help":
		fmt.Printf(usage, catalog.Names(), catalog.DefaultDir())
	default:
		err = fmt.Errorf("unknown command %q (try nollama help)", cmd)
	}
	if err != nil {
		log.Fatal("nollama: ", err)
	}
}

func modelsDir() string {
	if d := os.Getenv("NOLLAMA_MODELS"); d != "" {
		return catalog.ExpandHome(d)
	}
	return catalog.DefaultDir()
}

// parseKeepAlive reads a keep-alive the way Ollama's OLLAMA_KEEP_ALIVE does: seconds ("300", "-1") or a duration ("5m").
func parseKeepAlive(v string) (time.Duration, error) {
	if f, err := strconv.ParseFloat(v, 64); err == nil {
		return time.Duration(f * float64(time.Second)), nil
	}
	return time.ParseDuration(v)
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := os.Getenv("NOLLAMA_HOST")
	if addr == "" {
		addr = "127.0.0.1:11434"
	}
	fs.StringVar(&addr, "addr", addr, "address to listen on")
	threads := fs.Int("threads", envInt("NOLLAMA_THREADS"), "CPU threads, shared by every answer and model (0: the CPUs Go uses, up to 8)")
	maxMem := fs.String("max-memory", os.Getenv("NOLLAMA_MAX_MEMORY"), "memory budget for the model, answers and caches, such as 3GB or 800MB (empty or 0: no limit)")
	lowPrio := fs.Bool("low-priority", envBool("NOLLAMA_LOW_PRIORITY"), "run the engine's threads at a lower OS priority, so other programs win the CPU")
	det := fs.Bool("deterministic", envBool("NOLLAMA_DETERMINISTIC"), "the same logits, bit for bit, on any CPU or GPU (Metal, CUDA), thread count and batch size")
	parallel := fs.Int("parallel", 1, "answers written at once; the rest wait their turn")
	ka := 5 * time.Minute
	if v := os.Getenv("NOLLAMA_KEEP_ALIVE"); v != "" {
		d, err := parseKeepAlive(v)
		if err != nil {
			return fmt.Errorf("NOLLAMA_KEEP_ALIVE: %w", err)
		}
		ka = d
	}
	keepAlive := fs.Duration("keep-alive", ka, "how long an idle model stays loaded (0: unload once answered; negative: forever)")
	numCtx := fs.Int("num-ctx", 4096, "context window when a request doesn't set num_ctx")
	truncate := fs.Bool("truncate", false, "drop a chat's oldest messages when it doesn't fit the context window, instead of answering 400 (requests can still send \"truncate\": false)")
	cacheMB := fs.Int("prompt-cache-mb", 256, "memory for reusing the start of recent prompts (0: off)")
	kvType := kvCacheFlag(fs)
	gpu := gpuFlag(fs)
	quiet := fs.Bool("quiet", false, "don't log requests")
	apiKey := fs.String("api-key", os.Getenv("NOLLAMA_API_KEY"), "require this key (Authorization: Bearer <key>); prefer NOLLAMA_API_KEY, which other users can't see in ps")
	readOnly := fs.Bool("read-only", false, "turn off model management (pull, delete, create, copy, push)")
	noPull := fs.Bool("no-pull", false, "turn off /api/pull")
	noDelete := fs.Bool("no-delete", false, "turn off /api/delete")
	allow := fs.String("models", "", "the only models clients may use, comma-separated (default: any)")
	allowPaths := fs.Bool("allow-paths", false, "let requests load a .gguf by path outside the models folder")
	origins := fs.String("allowed-origins", os.Getenv("NOLLAMA_ORIGINS"), "browser origins allowed besides pages on this machine, comma-separated (*: any)")
	hosts := fs.String("allowed-hosts", os.Getenv("NOLLAMA_ALLOWED_HOSTS"), "host names served without an API key besides IPs and localhost, comma-separated (*: any)")
	maxBodyMB := fs.Int("max-body-mb", 8, "largest request body, in MB")
	maxTokens := fs.Int("max-tokens", 0, "longest answer a request may ask for (0: up to the context window)")
	maxNumCtx := fs.Int("max-num-ctx", 32768, "largest context window a request may ask for")
	perClient := fs.Int("per-client", 8, "requests one client IP may have open at once (negative: no limit)")
	public := fs.Bool("insecure-public", false, "listen on a non-loopback address without an API key (anyone who can reach it can use it)")
	draft := fs.String("draft", os.Getenv("NOLLAMA_DRAFT"), "a small model with the same tokenizer that guesses tokens for the larger one to check, e.g. qwen3-0.6b (speculative decoding, experimental: slower on the CPU today; \"\": off)")
	draftTokens := fs.Int("draft-tokens", 0, "tokens the draft guesses at a time (0: the engine's default)")
	fs.Parse(args)
	if !strings.Contains(addr, ":") {
		addr += ":11434"
	}
	memLimit, err := parseSize(*maxMem)
	if err != nil {
		return fmt.Errorf("--max-memory: %w", err)
	}
	cfg := server.Config{ModelsDir: modelsDir(), Threads: *threads, Parallel: *parallel, KeepAlive: *keepAlive,
		NumCtx: *numCtx, Truncate: *truncate, PromptCache: int64(*cacheMB) << 20,
		APIKey: strings.TrimSpace(*apiKey), ReadOnly: *readOnly, NoPull: *noPull, NoDelete: *noDelete,
		Models: list(*allow), AllowPaths: *allowPaths, AllowedOrigins: list(*origins), AllowedHosts: list(*hosts),
		MaxBodyBytes: int64(*maxBodyMB) << 20, MaxTokens: *maxTokens, MaxNumCtx: *maxNumCtx, PerClient: *perClient,
		MaxMemory: memLimit, LowPriority: *lowPrio, KVCacheType: *kvType, Deterministic: *det, GPU: *gpu,
		Draft: *draft, DraftTokens: *draftTokens, UnloadWhenIdle: *keepAlive == 0}
	if err := checkKVCacheType(*kvType); err != nil {
		return err
	}
	if *draft != "" {
		if _, err := pull(*draft); err != nil {
			return fmt.Errorf("draft model: %w", err)
		}
	}
	if !*quiet {
		cfg.Logf = log.Printf
	}
	if cfg.APIKey == "" && !loopback(addr) {
		if !*public {
			return fmt.Errorf("refusing to listen on %s without an API key: anyone who can reach it could use it and download "+
				"models to this machine. Set NOLLAMA_API_KEY, listen on 127.0.0.1, or pass --insecure-public", addr)
		}
		log.Printf("WARNING: listening on %s with no API key. Anyone who can reach this address can use it.", addr)
	}
	srv := server.New(cfg)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("%w (is Ollama or another nollama already on %s? stop it, or set NOLLAMA_HOST)", err, addr)
	}
	hs := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second, ReadTimeout: time.Minute,
		IdleTimeout: 2 * time.Minute, MaxHeaderBytes: 64 << 10}
	log.Printf("nollama %s listening on http://%s (models in %s)", version, ln.Addr(), cfg.ModelsDir)
	if u := srv.Resources().Usage(); u.MemoryLimit > 0 || *lowPrio {
		log.Printf("%d threads, memory budget %s, low priority %v", u.Threads, sizeString(u.MemoryLimit), u.LowPriority)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errc := make(chan error, 1)
	go func() { errc <- hs.Serve(ln) }()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	log.Print("stopping")
	sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = hs.Shutdown(sctx)
	return srv.Close()
}

// loopback reports whether addr (host:port) listens only on this machine.
func loopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback() // "", 0.0.0.0 and :: listen everywhere
}

// list splits a comma-separated flag, dropping blanks.
func list(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func envInt(name string) int {
	n, _ := strconv.Atoi(os.Getenv(name))
	return n
}

func envBool(name string) bool {
	b, _ := strconv.ParseBool(os.Getenv(name))
	return b
}

// parseSize reads a byte size: a number with an optional unit (B, KB, MB, GB, TB, binary multiples: 1GB is 1024 MB).
// A bare number is megabytes. "" and "0" are 0.
func parseSize(s string) (int64, error) {
	s = strings.ToUpper(strings.TrimSpace(s))
	if s == "" {
		return 0, nil
	}
	mult := float64(1 << 20)
	for _, u := range []struct {
		suffix string
		mult   float64
	}{{"TB", 1 << 40}, {"GB", 1 << 30}, {"MB", 1 << 20}, {"KB", 1 << 10}, {"T", 1 << 40}, {"G", 1 << 30}, {"M", 1 << 20}, {"K", 1 << 10}, {"B", 1}} {
		if rest, ok := strings.CutSuffix(s, u.suffix); ok {
			s, mult = strings.TrimSpace(rest), u.mult
			break
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 {
		return 0, fmt.Errorf("%q isn't a size (try 3GB or 800MB)", s)
	}
	return int64(f * mult), nil
}

func sizeString(n int64) string {
	if n <= 0 {
		return "none"
	}
	return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
}

func pull(spec string) (catalog.Source, error) {
	src, err := catalog.Resolve(spec, modelsDir())
	if err != nil {
		return src, err
	}
	if src.Present() {
		return src, nil
	}
	fmt.Fprintf(os.Stderr, "pulling %s from %s\n", src.Name, src.URL)
	last := time.Time{}
	err = catalog.Fetch(src, nil, func(done, total int64) {
		if time.Since(last) < 200*time.Millisecond && done != total {
			return
		}
		last = time.Now()
		if total > 0 {
			fmt.Fprintf(os.Stderr, "\r%5.1f%%  %s / %s", 100*float64(done)/float64(total), size(done), size(total))
		} else {
			fmt.Fprintf(os.Stderr, "\r%s", size(done))
		}
	})
	fmt.Fprintln(os.Stderr)
	if err == nil && src.SHA256 != "" {
		fmt.Fprintln(os.Stderr, "sha256 checked")
	}
	return src, err
}

func cmdPull(args []string) error {
	if len(args) != 1 {
		return errors.New("usage: nollama pull <model>")
	}
	src, err := pull(args[0])
	if err == nil && src.Inside {
		fmt.Println(src.Name, "is built into this program")
	} else if err == nil {
		fmt.Println(src.Path)
	}
	return err
}

func cmdList(args []string) error {
	local, err := catalog.Available(modelsDir())
	if err != nil {
		return err
	}
	if err := catalog.BundleError(); err != nil {
		fmt.Fprintln(os.Stderr, "nollama:", err)
	}
	if len(local) == 0 {
		fmt.Printf("no models in %s yet; try: nollama pull %s\n", modelsDir(), catalog.Catalog[1].Name)
		return nil
	}
	fmt.Printf("%-28s %10s  %s\n", "NAME", "SIZE", "MODIFIED")
	for _, m := range local {
		where := ""
		if m.Inside {
			where = "  (built in)"
		}
		fmt.Printf("%-28s %10s  %s%s\n", m.Name, size(m.Size), time.Unix(m.ModTime, 0).Format("2006-01-02 15:04"), where)
	}
	return nil
}

func cmdRm(args []string) error {
	if len(args) != 1 {
		return errors.New("usage: nollama rm <model>")
	}
	src, err := catalog.Resolve(args[0], modelsDir())
	if err != nil {
		return err
	}
	if src.Inside {
		return fmt.Errorf("%s is built into this program, not a download; it can't be deleted", src.Name)
	}
	if src.URL == "" {
		return fmt.Errorf("%s is a file you named, not a download; nollama leaves it alone", src.Path)
	}
	if err := os.Remove(src.Path); err != nil {
		return err
	}
	fmt.Println("deleted", src.Name)
	return nil
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	threads := fs.Int("threads", 0, "CPU threads (0: the engine's default)")
	maxTokens := fs.Int("max-tokens", 1024, "longest answer, in tokens")
	temp := fs.Float64("temperature", 0.7, "sampling temperature (0: always the likeliest token)")
	system := fs.String("system", "", "system prompt")
	verbose := fs.Bool("verbose", false, "print token counts and speed after each answer")
	kvType := kvCacheFlag(fs)
	gpu := gpuFlag(fs)
	det := fs.Bool("deterministic", false, "the same logits, bit for bit, on any CPU or GPU (Metal, CUDA)")
	draftName := fs.String("draft", "", "a small model with the same tokenizer that guesses tokens to check, e.g. qwen3-0.6b")
	draftTokens := fs.Int("draft-tokens", 0, "tokens the draft guesses at a time (0: the engine's default)")
	fs.Parse(args)
	if fs.NArg() < 1 {
		return errors.New("usage: nollama run [flags] <model> [prompt]")
	}
	src, err := pull(fs.Arg(0))
	if err != nil {
		return err
	}
	m, err := engine.Load(src.Path, engine.LoadOptions{Threads: *threads, PromptCache: 256 << 20, KVCacheType: *kvType, Deterministic: *det, GPU: *gpu})
	if err != nil {
		return err
	}
	defer m.Close()
	var draft *engine.Model
	if *draftName != "" {
		dsrc, err := pull(*draftName)
		if err != nil {
			return err
		}
		if draft, err = engine.Load(dsrc.Path, engine.LoadOptions{Threads: *threads, PromptCache: 64 << 20, Deterministic: *det, GPU: *gpu}); err != nil {
			return err
		}
		defer draft.Close()
		if err := engine.CheckDraft(m, draft); err != nil {
			return err
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var history []engine.Message
	if *system != "" {
		history = append(history, engine.Message{Role: "system", Content: *system})
	}
	answer := func(prompt string) error {
		history = append(history, engine.Message{Role: "user", Content: prompt})
		res, err := m.Generate(ctx, m.ChatPrompt(history), engine.GenerateOptions{MaxTokens: *maxTokens,
			Temperature: *temp, TopP: 0.9, TopK: 40, Seed: uint64(time.Now().UnixNano()), Draft: draft, DraftTokens: *draftTokens},
			func(piece string) bool { fmt.Print(piece); return true })
		fmt.Println()
		if err != nil {
			return err
		}
		history = append(history, engine.Message{Role: "assistant", Content: res.Text})
		if *verbose {
			fmt.Fprintf(os.Stderr, "[prompt %d tokens (%d cached) in %s, answer %d tokens at %.1f tokens/s]\n",
				res.PromptTokens, res.CachedTokens, res.Prefill.Round(time.Millisecond), res.OutputTokens,
				float64(res.OutputTokens)/max(res.Decode.Seconds(), 1e-9))
			if res.Drafted > 0 {
				fmt.Fprintf(os.Stderr, "[draft: %d of %d guessed tokens kept]\n", res.Accepted, res.Drafted)
			}
		}
		return nil
	}
	if fs.NArg() > 1 {
		return answer(strings.Join(fs.Args()[1:], " "))
	}
	fmt.Fprintf(os.Stderr, "%s ready (%s). Type a message; /bye or Ctrl-D to quit, /clear to forget the chat.\n", src.Name, m.Info().Kernels)
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<20), 1<<20)
	for {
		fmt.Fprint(os.Stderr, ">>> ")
		if !in.Scan() {
			fmt.Fprintln(os.Stderr)
			return in.Err()
		}
		line := strings.TrimSpace(in.Text())
		switch line {
		case "":
			continue
		case "/bye", "/exit", "/quit":
			return nil
		case "/clear":
			history = history[:0]
			if *system != "" {
				history = append(history, engine.Message{Role: "system", Content: *system})
			}
			continue
		}
		if err := answer(line); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
	}
}

func cmdBundle(args []string) error {
	fs := flag.NewFlagSet("bundle", flag.ExitOnError)
	out := fs.String("o", "", "the file to write")
	exe := fs.String("exe", "", "the program to add the models to (default: this nollama); any Go program that uses nollama's models package finds them")
	check := fs.String("check", "", "list the models inside this file and check their SHA-256, instead of writing one")
	fs.Usage = func() {
		fmt.Fprint(os.Stderr, `usage: nollama bundle -o <file> [-exe <program>] <model>... 
       nollama bundle -check <file>

Writes a copy of the program with the models added to its end. The copy finds them by name with nothing to
download: "nollama run qwen3-0.6b", the server's /api/tags, and catalog.Resolve in your own Go program. A model is
<name>, <name>=<spec> to give it another name, or any spec nollama run takes; it's downloaded first if needed.

`)
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if *check != "" {
		return checkBundle(*check)
	}
	if *out == "" || fs.NArg() == 0 {
		fs.Usage()
		os.Exit(2)
	}
	if *exe == "" {
		self, err := os.Executable()
		if err != nil {
			return err
		}
		*exe = self
	}
	var srcs []catalog.Source
	for _, arg := range fs.Args() {
		name, spec, renamed := strings.Cut(arg, "=")
		if !renamed {
			spec = arg
		}
		src, err := pull(spec)
		if err != nil {
			return err
		}
		if renamed {
			src.Name = catalog.Normalize(name)
		}
		srcs = append(srcs, src)
	}
	list, err := catalog.Bundle(*out, *exe, srcs)
	if err != nil {
		return err
	}
	for _, b := range list {
		fmt.Printf("%-28s %10s  sha256 %s\n", b.Name, size(b.Size), b.SHA256)
	}
	fmt.Println("wrote", *out)
	return nil
}

func checkBundle(path string) error {
	_, list, err := catalog.ReadBundle(path)
	if err != nil {
		return err
	}
	if len(list) == 0 {
		return fmt.Errorf("%s has no models inside", path)
	}
	bad := 0
	for _, b := range list {
		err := catalog.Verify(catalog.Source{Name: b.Name, Path: catalog.BundledPrefix + b.Name, File: path, Offset: b.Offset, Size: b.Size, SHA256: b.SHA256})
		status := "sha256 ok"
		if err != nil {
			status, bad = err.Error(), bad+1
		}
		fmt.Printf("%-28s %10s  %s\n", b.Name, size(b.Size), status)
	}
	if bad > 0 {
		return fmt.Errorf("%d of %d models failed their check", bad, len(list))
	}
	return nil
}

func size(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%d KB", n>>10)
}

// gpuFlag adds -gpu, defaulting to NOLLAMA_GPU.
func gpuFlag(fs *flag.FlagSet) *string {
	def := os.Getenv("NOLLAMA_GPU")
	if def == "" {
		def = "auto"
	}
	return fs.String("gpu", def, "auto: run on the GPU when one works (Metal on Apple Silicon, CUDA on NVIDIA); off: CPU only (also NOLLAMA_GPU)")
}

// kvCacheFlag adds -kv-cache-type, defaulting to NOLLAMA_KV_CACHE_TYPE.
func kvCacheFlag(fs *flag.FlagSet) *string {
	def := os.Getenv("NOLLAMA_KV_CACHE_TYPE")
	if def == "" {
		def = engine.KVCacheF32
	}
	return fs.String("kv-cache-type", def, "format of the attention cache: "+strings.Join(engine.KVCacheTypes, ", ")+
		" (smaller saves memory on long contexts; also NOLLAMA_KV_CACHE_TYPE)")
}

func checkKVCacheType(t string) error {
	for _, k := range engine.KVCacheTypes {
		if strings.EqualFold(strings.TrimSpace(t), k) {
			return nil
		}
	}
	return fmt.Errorf("unknown KV cache type %q (want one of %s)", t, strings.Join(engine.KVCacheTypes, ", "))
}
