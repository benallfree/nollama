# nollama

An embeddable, Ollama-compatible LLM engine, in Go.

If your app or tool wants a local model, the usual answer is "first install Ollama", and your users now have two
programs to install, update and keep running. nollama is an inference engine written in Go, plus an HTTP server that
speaks Ollama's API and OpenAI's. Ship it as one binary, or compile it into your own Go program. Your end users install
nothing else: no Ollama, no Python, no cgo.

> nollama is alpha. Today it runs Qwen3 models on the CPU, with hand-tuned assembly for ARM64 and x86-64: good for small
> models doing short, focused jobs, such as labeling a ticket, pulling JSON out of an email or summarizing a record.
> It also calls tools and makes embeddings (with Qwen3-Embedding). GPU support, more model families and images aren't
> there yet.

## Install

macOS or Linux, with Homebrew:

```bash
brew tap benallfree/nollama https://github.com/benallfree/nollama
brew install benallfree/nollama/nollama
```

Windows, with [Scoop](https://scoop.sh):

```powershell
scoop bucket add nollama https://github.com/benallfree/nollama
scoop install nollama
```

macOS or Linux, without Homebrew:

```bash
curl -fsSL https://raw.githubusercontent.com/benallfree/nollama/main/install.sh | sh
```

On Linux the script installs a `.deb`, `.rpm` or `.apk` package that adds a `nollama` systemd service (settings in
`/etc/nollama/nollama.env`); on macOS it puts `nollama` in `/usr/local/bin`. The zips and packages are also on the
[releases page](https://github.com/benallfree/nollama/releases), for Linux, macOS and Windows on x86-64 and ARM64, and
Linux on ARMv7, ppc64le and s390x.

With Go: `go install github.com/benallfree/nollama/cmd/nollama@latest`.

## Use it

Chat in the terminal (the first run downloads the model and checks its SHA-256):

```bash
nollama run qwen3-1.7b
nollama run qwen3-1.7b "Write a title for a post about sourdough starters."
```

Run the server. It listens on `127.0.0.1:11434`, Ollama's address, so tools that talk to Ollama work unchanged:

```bash
nollama serve
```

```bash
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "qwen3:1.7b",
  "messages": [{"role": "user", "content": "Label this ticket: I was charged twice."}],
  "format": {"type": "object", "properties": {"label": {"enum": ["bug", "billing", "question"]}}, "required": ["label"]},
  "stream": false
}'
```

nollama also has an endpoint of its own, `/api/classify`, which returns a probability for each label instead of
generating text (see [Classify](#classify)):

```bash
curl http://127.0.0.1:11434/api/classify -d '{
  "model": "qwen3:1.7b",
  "system": "Label the support ticket: bug, feature request or billing question.",
  "prompt": "I was charged twice this month.",
  "labels": ["bug", "feature request", "billing question"],
  "complete": true
}'
```

It answers `{"label": "billing question", "labels": [{"label": "bug", "probability": 0.000004, "logprob": ..., "tokens": 2}, ...]}`,
each label's probability, log-probability and token count, in the order given. `prompt` is the user's turn in the chat template (with `system`), or `messages` gives
the whole conversation; `raw: true` sends `prompt` as is. `normalize`, `options.num_ctx` and `keep_alive` work too.

Or with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3-1.7b", messages=[{"role": "user", "content": "Say hi in French."}])
print(r.choices[0].message.content)
```

Embeddings, for search and similarity (Ollama's `/api/embed` and OpenAI's `/v1/embeddings`; pull the model first with
`nollama pull qwen3-embedding-0.6b`):

```bash
curl http://127.0.0.1:11434/api/embed -d '{
  "model": "qwen3-embedding:0.6b",
  "input": ["The capital of China is Beijing.", "Gravity is a force that attracts two bodies towards each other."]
}'
```

Vectors are unit length, so a dot product is the cosine similarity. For retrieval, Qwen3-Embedding wants an instruction
on queries (not on documents): `Instruct: Given a web search query, retrieve relevant passages that answer the
query\nQuery:<the query>`. `dimensions` keeps the first n values (32 to 1024).

### Tool calls

Pass `tools` (and `tool_choice`: `auto`, `none`, `required` or a named function) on `/v1/chat/completions` or
`/api/chat`. The model decides between answering and calling a tool; once it starts a call, nollama holds the rest of it
to a tool you listed and to that tool's parameter schema, token by token. A call always parses, and its arguments match
the schema. `required` or a named function makes the answer a call from the first token. Several calls in one answer
come back as several entries in `tool_calls`.

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
r = client.chat.completions.create(model="qwen3-1.7b", messages=messages, tools=tools)
call = r.choices[0].message.tool_calls[0]  # finish_reason is "tool_calls"
print(call.function.name, call.function.arguments)  # get_weather {"city": "Paris"}

messages += [r.choices[0].message, {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 18, "sky": "sunny"}'}]
r = client.chat.completions.create(model="qwen3-1.7b", messages=messages, tools=tools)
print(r.choices[0].message.content)  # It's 18°C and sunny in Paris.
```

Other commands: `nollama pull <model>`, `nollama list`, `nollama rm <model>`, and `nollama bundle` to
[ship a model inside the binary](#ship-the-model-inside-the-binary). `nollama serve --help` lists the server's
flags (threads, answers at once, how long an idle model stays loaded, context window, KV cache type). `NOLLAMA_HOST`
sets the address, `NOLLAMA_MODELS` the models folder and `NOLLAMA_KV_CACHE_TYPE` the KV cache's format.

## Context limits

nollama never cuts a prompt to make it fit. Ollama's default context window quietly drops the start of a long prompt,
and the model then seems to ignore its instructions. Here, a prompt that doesn't fit the context window (`num_ctx`,
default 4096, `--num-ctx` on the server) is an HTTP 400 that gives the token counts and the limit, in each API's error
shape. An answer that reaches `num_predict`/`max_tokens` or the end of the window ends with `done_reason: "length"`
(Ollama) or `finish_reason: "length"` (OpenAI). Token counts are always in the answer: `prompt_eval_count` and
`eval_count` on Ollama, `usage` on OpenAI (when streaming, with `stream_options.include_usage`).

To drop old turns instead, send `"truncate": true` with a chat (`/api/chat` or `/v1/chat/completions`), or start the
server with `--truncate` (a request can still send `"truncate": false`). nollama then keeps the system prompt and the
most recent messages, leaving `max_tokens` (or a quarter of the window, at most 1024) for the answer. It says what it
dropped in the answer, `"truncated": {"messages": 6, "tokens": 1840, "prompt_tokens": 5012}`, and in the server log.

## Embed it in a Go program

Mount the server in your own HTTP server:

```go
import "github.com/benallfree/nollama/server"

srv := server.New(server.Config{ModelsDir: "./models"})
defer srv.Close()
mux.Handle("/llm/", http.StripPrefix("/llm", srv.Handler()))
```

Or call the engine directly, with no HTTP at all:

```go
import (
	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/models"
	"github.com/benallfree/nollama/schema"
)

src, _ := models.Resolve("qwen3-1.7b", models.DefaultDir())
_ = models.Fetch(src, nil, nil) // downloads once, checks the SHA-256
m, _ := engine.Load(src.Path, engine.LoadOptions{}) // a downloaded file, or a model built into the binary
defer m.Close()

label, _ := schema.Compile([]byte(`{"type":"object","properties":{"label":{"enum":["bug","billing","question"]}},"required":["label"]}`))
res, _ := m.Generate(ctx, m.ChatPrompt([]engine.Message{{Role: "user", Content: "I was charged twice."}}),
	engine.GenerateOptions{MaxTokens: 30, Constraint: label.Fit}, nil)
fmt.Println(res.Text) // {"label": "billing"}
```

Embeddings work the same way, with an embedding model:

```go
es, _ := m.Embed(ctx, []string{engine.EmbedQuery("", "What is the capital of China?"), "The capital of China is Beijing."},
	engine.EmbedOptions{})
fmt.Println(engine.Cosine(es[0].Vector, es[1].Vector)) // about 0.77
```

### Classify

To label something, you don't need the model to write an answer and then parse it. `Classify` reads the prompt once
and scores each candidate label as the model's answer, so you get a probability for every label, and the answer is
always one of them:

```go
prompt := m.ChatPrompt([]engine.Message{
	{Role: "system", Content: "Label the support ticket: bug, feature request or billing question."},
	{Role: "user", Content: "I was charged twice this month."},
})
res, _ := m.Classify(ctx, prompt, []string{"bug", "feature request", "billing question"},
	engine.ClassifyOptions{Complete: true})
fmt.Println(res.Best(), res.Probs) // 2 [0.000004 0.000018 0.99998] with qwen3-1.7b
```

Labels can be several tokens long. `Complete` scores each label as the whole answer (it must be followed by the end
of the turn), which matters when one label starts another ("bug", "bug report"); `Normalize` scores per token, so long
labels aren't penalized. Labels are tokenized on their own, so write them as the answer would start. A one-token label
costs nothing beyond reading the prompt; the other labels' tokens are read together in one batched pass; and the prompt
cache skips instructions shared across calls. Labeling a ticket with 5 labels (an 85-token prompt), that's 1.6 to 1.8
times faster than generating a schema-constrained JSON answer (`go test ./engine -bench Classify`), and you get
the model's odds for every label instead of one word.

The engine also keeps the start of recent prompts (`LoadOptions.PromptCache`), so a long system prompt is read once,
and can keep a conversation's state between calls (`GenerateOptions.Thread`).

A prompt that doesn't fit is an error you can check: `errors.Is(err, engine.ErrContextExceeded)`, or `errors.As` with
`*engine.ContextError` for the counts. `engine.TrimMessages` drops a conversation's oldest messages to fit a budget,
and tells you what it dropped.

### Save a conversation and pick it up later

A thread's state (the keys and values of every token the model read and wrote) can be saved wherever your app keeps
things, and loaded back after a restart or on another machine. The next turn then reads only the new message, not the
whole conversation:

```go
var buf bytes.Buffer
_ = m.SaveThread(&buf, "", "chat-42", engine.SnapshotOptions{})   // after a Generate with Thread: "chat-42"
// ... store buf.Bytes() in a file, a database row, object storage ...

info, err := m.LoadThread(bytes.NewReader(saved), "", "chat-42", engine.SnapshotOptions{})
res, _ := m.Generate(ctx, prompt, engine.GenerateOptions{Thread: "chat-42"}, nil) // res.CachedTokens: the saved part
```

`SavePromptCache` and `LoadPromptCache` do the same for the prompt cache, so a long system prompt warmed once (`Warm`)
can ship with your app or survive restarts, pinned so other prompts don't push it out.

A snapshot is tied to the model file it came from (a fingerprint of the GGUF's header and samples of its weights, so a
renamed copy works and a finetune with the same shapes doesn't), versioned and checksummed; a mismatch returns
`engine.ErrSnapshotModel`, damage `engine.ErrSnapshotCorrupt`. It's half precision by default, which moves token
probabilities by about one in a million; `SnapshotOptions{Full: true}` stores exact floats, and the conversation then
continues bit for bit as if it had never stopped. `Compress: true` deflates it: that saves 11 to 17% at full precision,
but only 2 to 6% at half, whose bits are mostly noise to a compressor.

Snapshots are big: every token keeps its keys and values for every layer, about 112 KB per token at half precision for
Qwen3 0.6B. Restoring is still far cheaper than reading again. A 2,149-token conversation on `qwen3:0.6b` (Q4_K_M,
Apple M1 Max, 4 threads, other jobs running, so the reading time is on the slow side):

| | Size | Save | Load |
|---|---|---|---|
| Reading the conversation again | | | 29 to 35 s |
| Half precision | 247 MB | 1.5 to 3 s | 0.5 to 1.0 s |
| Half, compressed | 231 MB | 1.6 s | 1.5 s |
| Full precision | 493 MB | 0.7 s | 0.3 to 0.8 s |
| Full, compressed | 409 MB | 1.7 s | 2.4 s |

(`GenerateOptions.ThreadDir` already keeps threads on disk in a folder the engine manages; snapshots are for when your
app wants to hold the state itself.)

### Keep it from starving your app

Give nollama a budget so it shares the machine with the program it lives in:

```go
res := engine.NewResources(engine.ResourceOptions{
	Threads:     4,       // CPU threads, shared by every model and every answer
	Memory:      3 << 30, // bytes for weights, answers in progress and caches
	LowPriority: true,    // your app's threads win the CPU when it's busy
})
defer res.Close()

srv := server.New(server.Config{ModelsDir: "./models", Resources: res})
m, _ := engine.Load(src.Path, engine.LoadOptions{Resources: res}) // same budget
```

- **Threads.** Every model and every answer shares the same worker threads, taking turns, instead of each model
  starting its own. Each call also works on its caller's goroutine, so cap concurrent calls (the server's `Parallel`)
  to cap the total.
- **Memory.** The budget counts the model files (they're mapped into memory and read in full for every token), each
  answer's keys and values, and the prompt cache and conversation threads kept in memory. When something doesn't fit,
  nollama first drops what it can rebuild (idle buffers, then the least recently used cache entries, across all models
  on the budget). A load or an answer that still doesn't fit fails with an error matching `engine.ErrMemoryLimit`
  (the server answers 503) instead of meeting the OOM killer. Go's own heap overhead isn't counted, so leave some
  headroom, or also set `GOMEMLIMIT`.
- **Low priority.** The OS runs nollama's threads below your app's: nice 10 on Linux (on desktops with autogroup
  scheduling, that's below the rest of your process rather than other programs), the utility QoS class on macOS, and
  `THREAD_PRIORITY_LOWEST` on Windows. Other systems run at normal priority (`engine.LowPrioritySupported`). It's done
  without cgo: the threads doing the work are dedicated to it and end when it does, so a lowered thread never runs
  the rest of your app.

For one model, `LoadOptions` takes `Threads`, `MaxMemory` and `LowPriority` directly. `nollama serve` takes
`--threads`, `--max-memory 3GB` and `--low-priority` (or `NOLLAMA_THREADS`, `NOLLAMA_MAX_MEMORY`,
`NOLLAMA_LOW_PRIORITY`).

### The same numbers on every machine

By default the logits depend on the CPU: the arm64 and amd64 assembly kernels sum in different orders, Go fuses
`x*y+z` into one instruction on arm64 but not on amd64, and a prompt read in batches rounds differently from one read a
token at a time. The differences are in the last bits, but quantizing activations between layers can grow them to
about a logit, enough to change a sampled token. Thread count doesn't matter.

`LoadOptions.Deterministic` (or `nollama serve --deterministic`, `NOLLAMA_DETERMINISTIC=1`, `nollama run
--deterministic`) makes the logits the same bits on arm64 and amd64, at any thread count and prompt batch size, with or
without a prompt-cache hit. Greedy output, logprobs, `Score` (any `Batch`), `Classify` and seeded sampling follow. Use it when two machines must agree
exactly, for example a compressor driven by the model's probabilities. It runs the portable Go kernels instead of the
assembly ones (`Info().Kernels` reports `go (deterministic)`), so it's slower, mostly when reading prompts. It won't
continue a conversation from half-precision state: thread files in a `ThreadDir` are ignored, and `LoadThread` /
`LoadPromptCache` need a snapshot saved with `Full`. It needs the default `f32` KV cache. `TestDeterministic` checks the logits against golden hashes, so CI
on each architecture catches a change that breaks this.

## Test without a model

The `nollamatest` package lets your tests run in CI with no download. A scripted model answers with a function you
write, so you can test what your code does with an answer:

```go
import "github.com/benallfree/nollama/nollamatest"

func TestRouting(t *testing.T) {
	m := nollamatest.Scripted(t, func(p nollamatest.Prompt) string {
		if strings.Contains(p.Last("user"), "charged") {
			return `{"label": "billing"}`
		}
		return `{"label": "question"}`
	})
	// m is an *engine.Model: pass it to your code. Its answers are tokenized, streamed, cut at max tokens and stop
	// strings, and checked against your schema (an answer that doesn't fit is an error).
}
```

`nollamatest.Load(t, nollamatest.TinyConfig{})` is a tiny Qwen3 model with random weights (about 1 MB, built in
memory in a few milliseconds). It runs the whole engine, constraints included, and writes a jumble of English words,
the same every run on a given machine.

For code that talks HTTP, `nollamatest.NewServer(t)` runs nollama's real server on either model. It answers to every
catalog name (`qwen3:1.7b` and the rest), so code that names a real model works unchanged:

```go
ts := nollamatest.NewServer(t, nollamatest.WithResponder(nollamatest.Reply("Bonjour !")))
client := openai.NewClient(option.WithBaseURL(ts.URL+"/v1"), option.WithAPIKey("unused"))
// or an Ollama client pointed at ts.URL
```

## Ship the model inside the binary

A model can travel inside the executable, so the program runs it on a machine with no network and nothing to
download. Two ways:

**Add it to a built binary.** `nollama bundle` writes a copy of a program with GGUF files added to its end:

```bash
nollama bundle -o mytool qwen3-0.6b                 # a nollama that has qwen3-0.6b built in
nollama bundle -o myapp -exe ./myapp tagger=qwen3-0.6b   # your own Go program, model named "tagger"
nollama bundle -check myapp                         # list what's inside and check each SHA-256
```

The copy finds its models by name: `mytool run qwen3-0.6b`, `/api/tags` and `/v1/models` list them, and in your own
program `models.Resolve("tagger", dir)` returns them before any download. Each model sits on a 64 KiB boundary and is
mmapped straight from the executable, so it loads as fast as a separate file (tens of milliseconds). Catalog models are
checked against their SHA-256 as they're copied in, and the bundle records every model's SHA-256 for `-check` and
`models.Verify`. On macOS the result runs with the linker's ad-hoc signature, but `codesign` can't sign a file with
data after its end: bundle before shipping only if you don't need a Developer ID signature, and use `go:embed` (below)
if you do. On Windows, sign after bundling; nollama still finds the models. macOS checks a new executable the
first time it runs, which takes a few seconds for one this size; after that it starts at once.

**Compile it in with `go:embed`.** The model becomes part of the program's own data, signed along with it:

```go
//go:embed qwen3-0.6b.gguf
var qwen []byte

func init() { models.Embed("qwen3-0.6b", qwen) }

src, _ := models.Resolve("qwen3-0.6b", models.DefaultDir()) // the embedded one, no download
m, _ := engine.Load(src.Path, engine.LoadOptions{})          // src.Path is "embedded:qwen3-0.6b"
```

The operating system pages embedded data in from the executable as it's read, so there's no copy and no startup cost.
Either way, a model inside the program resolves like any other: `models.Resolve` returns it with a path of
`bundled:<name>` or `embedded:<name>`, and `engine.Load(src.Path, …)` opens it, so code written for downloaded models
needs no change, and models not built in still download as usual. `engine.LoadBytes` takes any GGUF already in
memory, and `engine.LoadRange` one at an offset inside any file.

## Long contexts: a smaller KV cache

While it answers, the model keeps the attention keys and values of every token so far (the KV cache). At full
precision that's 224 KB per token for Qwen3 0.6B and 1.7B (288 KB for 4B), so a 4,096-token conversation takes about
900 MB on top of the model. `--kv-cache-type` (or `NOLLAMA_KV_CACHE_TYPE`, or `engine.LoadOptions.KVCacheType`) stores
it smaller. The names are Ollama's `OLLAMA_KV_CACHE_TYPE` values:

| Type | Per token (0.6B, 1.7B) | 4,096 tokens | 32,768 tokens | Perplexity vs `f32` (0.6B, 1.7B) |
|---|---|---|---|---|
| `f32` (default) | 224 KB | 896 MB | 7.0 GB | exactly what the model computes |
| `f16` | 112 KB | 448 MB | 3.5 GB | +0.3%, +0.9% (Ollama's default cache) |
| `q8_0` | 60 KB | 238 MB | 1.9 GB | +0.1%, −0.0% |
| `q4_0` | 46 KB | 182 MB | 1.4 GB | −4.1%, +0.1% |

`q4_0` stores values in 4 bits but keeps keys at 8 (`q8_0`): with 4-bit keys Qwen3's perplexity more than doubles,
because attention scores are sensitive to errors in keys and not in values.

Quality was measured on a fixed 361-token text, teacher-forced, with Ollama's Q4_K_M files: perplexity, and how far
each token's log-probability moves from `f32` (on average 0.12 to 0.17 for `f16`, 0.19 to 0.25 for `q8_0`, 0.28 to
0.29 for `q4_0`). Changes within a couple of percent either way are noise: rounding the cache to `f16` alone moves a
small model's log-probabilities that much. `go test ./engine -run TestKVCacheQuality` with `GGUF` set repeats it.

Speed (provisional: measured on a heavily loaded M1 Max, load average 150 to 230, so only the attention kernels'
relative cost is meaningful): attention over a compact cache takes 1.5× (`q8_0`), 2× (`f16`) or 2 to 3.5× (`q4_0`) the
time of `f32` attention, single-threaded over 2,048 positions. From that, writing at a 2K-token context should be about
5% slower with `q8_0` and 10 to 25% slower with `f16` or `q4_0`, and less at shorter contexts; reading a prompt is
dominated by the weights either way. These are estimates until they're measured on an idle machine.
`go test ./engine -run '^$' -bench BenchmarkKVAttention` measures the kernels.

The prompt cache and threads keep their copies at full precision, within their own budgets (`--prompt-cache-mb`,
`LoadOptions.ThreadMemory`); the KV cache type sets the memory of the answers being written.

```bash
nollama serve --kv-cache-type q8_0
```

## Models

| Name | Download | Memory | Good for |
|---|---|---|---|
| `qwen3-0.6b` | 640 MB | about 1 GB | tags, yes/no, short extraction |
| `qwen3-1.7b` | 1.8 GB | about 2 GB | a good default for a small server |
| `qwen3-4b` | 2.5 GB | about 3 GB | better answers, slower |
| `qwen3-embedding-0.6b` | 640 MB | about 1 GB | embeddings (1024 values) for search and similarity |

These come from Qwen's own Hugging Face repository and are checked against their SHA-256. Ollama's spellings work too
(`qwen3:1.7b`, `qwen3-embedding:0.6b`). Any other Qwen3 GGUF works by `hf:<org>/<repo>/<file>.gguf`, by path to a `.gguf` file (in
`nollama run`; the server takes paths only with `--allow-paths`), or, for the server too, by name once the file is in
the models folder (`mymodel.gguf` there is `mymodel`). The engine
runs Qwen3 dense models in Q8_0, Q4_K_M, F16 and BF16; other architectures are not supported yet.

## Benchmarks

Measured on an Apple M1 Max (64 GB), 4 CPU threads, generating 128 tokens; nollama and Ollama 0.23 ran on the same
GGUF files (Ollama's `qwen3:0.6b` and `qwen3:1.7b`, Q4_K_M). "Short" is a 35-token prompt, "long" a 590-token one.
Medians of 3 runs, which vary by 10 to 20% from run to run on a busy laptop.

| Model | Prompt | Engine | Reading the prompt | Writing the answer | First token | Load | Memory |
|---|---|---|---|---|---|---|---|
| 0.6B | short | **nollama, CPU** | 241 tok/s | 76 tok/s | 145 ms | 0.04 s | 667 MB |
| 0.6B | short | Ollama, CPU | 241 tok/s | 85 tok/s | 145 ms | 1.1 s | 875 MB |
| 0.6B | short | Ollama, GPU (Metal) | 731 tok/s | 82 tok/s | 48 ms | 0.9 s | 844 MB |
| 0.6B | long | **nollama, CPU** | 220 tok/s | 51 tok/s | 2.7 s | 0.02 s | 906 MB |
| 0.6B | long | Ollama, CPU | 349 tok/s | 53 tok/s | 1.7 s | 1.0 s | 885 MB |
| 0.6B | long | Ollama, GPU (Metal) | 1485 tok/s | 42 tok/s | 0.4 s | 0.6 s | 862 MB |
| 1.7B | short | **nollama, CPU** | 86 tok/s | 31 tok/s | 409 ms | 0.10 s | 1.4 GB |
| 1.7B | short | Ollama, CPU | 89 tok/s | 39 tok/s | 392 ms | 0.6 s | 1.7 GB |
| 1.7B | short | Ollama, GPU (Metal) | 364 tok/s | 54 tok/s | 96 ms | 0.7 s | 1.7 GB |
| 1.7B | long | **nollama, CPU** | 99 tok/s | 32 tok/s | 5.9 s | 0.02 s | 1.7 GB |
| 1.7B | long | Ollama, CPU | 262 tok/s | 32 tok/s | 2.3 s | 0.6 s | 1.8 GB |
| 1.7B | long | Ollama, GPU (Metal) | 589 tok/s | 29 tok/s | 1.0 s | 1.0 s | 1.7 GB |

What that means:

- **Writing** runs at roughly Ollama's CPU speed, 80 to 100% of it. A word is about 1.3 tokens, so the 1.7B model writes a
  short answer in a second or two.
- **Reading long prompts** is where nollama is slower: 40 to 65% of Ollama's CPU speed, and far behind a GPU. A
  600-token prompt takes a few seconds before the first word. The prompt cache helps when prompts share a start (a fixed
  system prompt, a growing chat): only the new part is read.
- **Loading** is near instant (the file is memory-mapped), and memory use is a little lower.
- On a small server it is slower still. On a 1 vCPU x86 VPS (Skylake, AVX2), `qwen3-0.6b` reads about 35 tokens/s and
  writes about 9: a one-word label takes about a second, a paragraph ten or more. Keep answers short (`num_predict`,
  `max_tokens`) and run long jobs in the background.

Today, if your users have a GPU and are happy to install Ollama, Ollama is faster.

Run them yourself with `go run ./engine/cmd/bench bench` (it needs Ollama running, for the comparison).

## Draft models (experimental, not faster yet)

A small model can guess the next few tokens and let the larger one check them all in one pass (speculative decoding).
On a GPU that check costs about as much as writing one token, so every guess the larger model agrees with is a token for
free. The answer doesn't change: with temperature 0 it is exactly the larger model's own answer, and with sampling it
follows the same distribution.

```bash
nollama serve --draft qwen3-0.6b
nollama run --draft qwen3-0.6b qwen3-4b "Write a Go function that reverses a string."
```

**On nollama's CPU engine today it makes answers slower, so leave it off.** Checking a batch of five tokens costs the
1.7B model three to five times as much as writing one, so even when the larger model keeps most guesses it comes out
behind: 0.2 to 0.9 times the speed without a draft in our runs. It is here, correct and tested, so it speeds answers up
as soon as nollama reads short batches faster.

The guesses themselves are good. With `qwen3-0.6b` guessing up to 4 tokens for `qwen3-1.7b` (it stops early after a
token it isn't sure of), each check yields, at temperature 0: about 4 tokens for code, 2.7 to 2.8 for JSON or a list,
and 1.8 for free-form prose.

The draft must share the larger model's tokenizer (all the Qwen3 sizes do), and is skipped for the draft model itself.
A request can set `draft_tokens` in Ollama's `options` (or at the top level of an OpenAI request): how many tokens the
draft guesses at a time, `0` to turn it off. In Go, load both models and pass `engine.GenerateOptions{Draft: draft}`;
`engine.CheckDraft` says whether a pair fits.

## What's supported

Ollama's API: `/api/chat`, `/api/generate` (with `raw`, `system`, `context`), `/api/tags`, `/api/ps`, `/api/show`,
`/api/pull`, `/api/delete`, `/api/version`, `/api/embed` (`input`, `truncate`, `dimensions`) and the older
`/api/embeddings`, streaming and not, `format` (`"json"` or a JSON schema), `options`
(`temperature`, `top_p`, `top_k`, `num_predict`, `num_ctx`, `seed`, `stop`, and nollama's `draft_tokens`), `keep_alive`.

OpenAI's API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` (text or token ids,
`dimensions`, `encoding_format` `float` or `base64`), streaming (with `stream_options.include_usage`)
and not, `response_format` (`json_object`, `json_schema`), `max_tokens`/`max_completion_tokens`, `temperature`,
`top_p`, `seed`, `stop`.

nollama's own: `/api/classify` (label probabilities, above).

Not yet: GPUs, model families other than Qwen3, tool calls, images, `/api/create`, thinking mode
(Qwen3's thinking is always off), and several models loaded at once (nollama holds one and swaps on demand).

## Security

`nollama serve` listens on `127.0.0.1` with no API key, which is safe on a machine you alone use. It also guards
against the attacks that have hit local LLM servers:

- **Web pages.** A page you visit can send requests to a server on your machine. nollama refuses any request from a
  browser page that isn't served from this machine (its `Origin` header), and any request whose `Host` header names a
  domain rather than an IP address or `localhost`, which stops DNS rebinding. `--allowed-origins` (`NOLLAMA_ORIGINS`)
  and `--allowed-hosts` (`NOLLAMA_ALLOWED_HOSTS`) widen these.
- **Files on disk.** A request names models, not files: catalog names, `hf:` specs, and `.gguf` files in the models
  folder. A path anywhere else (`/etc/passwd`, `../../x.gguf`) is refused unless you pass `--allow-paths`. Downloads
  always land inside the models folder, whatever the `hf:` spec says.
- **Open to the network.** `serve` won't listen on anything but a loopback address without an API key. Set
  `NOLLAMA_API_KEY` (better than `--api-key`, which other users can see in `ps`) and clients send
  `Authorization: Bearer <key>`: OpenAI clients through `api_key`, Ollama's through their `headers` option. `GET /`
  answers without the key, for health checks. `--insecure-public` listens without one anyway, with a warning.
- **Limits.** Request bodies are capped at 8 MB (`--max-body-mb`), the context window a request may ask for at 32768
  tokens (`--max-num-ctx`), and one client IP to 8 requests at once (`--per-client`; behind a reverse proxy, set it
  there). `--max-tokens` caps how long an answer may be.
- **Model management.** `--read-only` turns off pulls, deletes, create, copy and push; `--no-pull` and `--no-delete`
  turn off one each. `--models qwen3-1.7b,qwen3-0.6b` limits the models clients may use, and the listings show only
  those.

Embedding the server, the same settings are fields of `server.Config`. A typical embedded server serves one model and
manages nothing:

```go
srv := server.New(server.Config{ModelsDir: "./models", ReadOnly: true, Models: []string{"qwen3-1.7b"}, APIKey: key})
```

## License

MIT
