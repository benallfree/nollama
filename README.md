# nollama

An embeddable, Ollama-compatible LLM engine, in Go.

If your app or tool wants a local model, the usual answer is "first install Ollama", and your users now have two
programs to install, update and keep running. nollama is an inference engine written in Go, plus an HTTP server that
speaks Ollama's API and OpenAI's. Ship it as one binary, or compile it into your own Go program. Your end users install
nothing else: no Ollama, no Python, no cgo.

> nollama is alpha. Today it runs Qwen3 models on the CPU, with hand-tuned assembly for ARM64 and x86-64: good for small
> models doing short, focused jobs, such as labeling a ticket, pulling JSON out of an email or summarizing a record.
> GPU support, more model families, tool calls, images and embeddings aren't there yet.

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

Or with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3-1.7b", messages=[{"role": "user", "content": "Say hi in French."}])
print(r.choices[0].message.content)
```

Other commands: `nollama pull <model>`, `nollama list`, `nollama rm <model>`. `nollama serve --help` lists the server's
flags (threads, answers at once, how long an idle model stays loaded, context window). `NOLLAMA_HOST` sets the address
and `NOLLAMA_MODELS` the models folder.

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
m, _ := engine.Load(src.Path, engine.LoadOptions{})
defer m.Close()

label, _ := schema.Compile([]byte(`{"type":"object","properties":{"label":{"enum":["bug","billing","question"]}},"required":["label"]}`))
res, _ := m.Generate(ctx, m.ChatPrompt([]engine.Message{{Role: "user", Content: "I was charged twice."}}),
	engine.GenerateOptions{MaxTokens: 30, Constraint: label.Fit}, nil)
fmt.Println(res.Text) // {"label": "billing"}
```

The engine also keeps the start of recent prompts (`LoadOptions.PromptCache`), so a long system prompt is read once,
and can keep a conversation's state between calls (`GenerateOptions.Thread`).

## Models

| Name | Download | Memory | Good for |
|---|---|---|---|
| `qwen3-0.6b` | 640 MB | about 1 GB | tags, yes/no, short extraction |
| `qwen3-1.7b` | 1.8 GB | about 2 GB | a good default for a small server |
| `qwen3-4b` | 2.5 GB | about 3 GB | better answers, slower |

These come from Qwen's own Hugging Face repository and are checked against their SHA-256. Ollama's spellings work too
(`qwen3:1.7b`). Any other Qwen3 GGUF works by `hf:<org>/<repo>/<file>.gguf` or by path to a `.gguf` file. The engine
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

## What's supported

Ollama's API: `/api/chat`, `/api/generate` (with `raw`, `system`, `context`), `/api/tags`, `/api/ps`, `/api/show`,
`/api/pull`, `/api/delete`, `/api/version`, streaming and not, `format` (`"json"` or a JSON schema), `options`
(`temperature`, `top_p`, `top_k`, `num_predict`, `num_ctx`, `seed`, `stop`), `keep_alive`.

OpenAI's API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming (with `stream_options.include_usage`)
and not, `response_format` (`json_object`, `json_schema`), `max_tokens`/`max_completion_tokens`, `temperature`,
`top_p`, `seed`, `stop`.

Not yet: GPUs, model families other than Qwen3, tool calls, images, embeddings, `/api/create`, thinking mode
(Qwen3's thinking is always off), and several models loaded at once (nollama holds one and swaps on demand).
The server has no authentication; keep it on `127.0.0.1` or put it behind something that has.

## License

MIT
