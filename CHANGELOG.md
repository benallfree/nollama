# Changelog

Every user-visible change gets a line under **Unreleased** in the commit that makes it. At a release, `## Unreleased`
becomes `## vX.Y.Z` and its lines are the release notes (`release.sh` reads them).

## Unreleased

- `nollama serve` can require an API key (`NOLLAMA_API_KEY` or `--api-key`; clients send `Authorization: Bearer <key>` or `X-API-Key`), and refuses to listen on a non-loopback address without one unless you pass `--insecure-public`.
- Requests from web pages not served from this machine, and requests whose `Host` is a domain other than `localhost` (DNS rebinding), are refused unless allowed with `--allowed-origins` / `--allowed-hosts` (or an API key is set, for hosts).
- The server no longer loads a model by a path outside its models folder (`"model": "/etc/passwd"`); `--allow-paths` turns that back on. A `.gguf` file in the models folder can be used by its name, and `hf:` downloads can't write outside the folder.
- `--read-only`, `--no-pull` and `--no-delete` turn off model management; `--models` limits which models clients may use.
- Request limits: body size (`--max-body-mb`, default 8), context window (`--max-num-ctx`, default 32768), answer length (`--max-tokens`), and requests open at once per client IP (`--per-client`, default 8).
- The same settings are fields of `server.Config` for embedders.
- `nollamatest` package: test code that uses nollama with no model download. A tiny random-weight Qwen3 model (about 1 MB, built in memory) runs the real engine; a scripted model answers with a function you write; `nollamatest.NewServer(t)` runs either behind the real Ollama/OpenAI HTTP server, answering to every catalog model name.
- `engine.LoadOptions.Respond` makes a model answer from a script instead of its weights, and `server.Config.Load` lets a test choose how model files are opened.
- Save and load a conversation's state, and the prompt cache's, through any `io.Writer`/`io.Reader` (`Model.SaveThread`, `LoadThread`, `SavePromptCache`, `LoadPromptCache`): a long conversation picks up after a restart without being read again. Snapshots are tied to the model file by a fingerprint, versioned and checksummed, half or full precision, optionally compressed.
- Embeddings: `/api/embed` (one text or a list, `truncate`, `dimensions`), `/api/embeddings` and `/v1/embeddings` (text or token ids, `dimensions`, `encoding_format` `float` or `base64`) answer with unit-length vectors from Qwen3-Embedding, which `qwen3-embedding-0.6b` (or `qwen3-embedding:0.6b`) downloads. In Go, `Model.Embed` and `engine.EmbedQuery` for the query instruction. `/api/show` reports an embedding model's capability as `embedding`.
- A prompt that doesn't fit the context window is an HTTP 400 that gives the token counts and the limit, in Ollama's and OpenAI's error shapes; nollama never truncates a prompt silently.
- Opt-in truncation: `"truncate": true` on `/api/chat` and `/v1/chat/completions`, or `nollama serve --truncate`, keeps the system prompt and the most recent messages, and reports what was dropped in a `truncated` field and the server log.
- `prompt_eval_count` and `eval_count` are always in Ollama's final answer, even when 0. `/v1/completions` streams `usage` with `stream_options.include_usage`.
- Engine: context errors are `*engine.ContextError` (matches `engine.ErrContextExceeded`) with the counts; `engine.TrimMessages` trims a conversation to a token budget.
- Models inside the binary: `nollama bundle -o <file> <model>...` adds GGUF files to the end of nollama or any Go program built with it, found by name and mmapped from the executable; `nollama bundle -check` verifies them. Go programs can also compile a model in with `go:embed` and `models.Embed`. A built-in model's path is `bundled:<name>` or `embedded:<name>`, which `engine.Load` opens, so loading by `src.Path` works for every model. New `engine.LoadRange`, `engine.LoadBytes`, `models.Bundle`, `models.Builtin`, `models.Available` and `models.Verify`; `/api/tags`, `/v1/models` and `nollama list` show built-in models.
- Tool calls: `tools` and `tool_choice` (`auto`, `none`, `required`, a named function) on `/v1/chat/completions` and `/api/chat`, streaming or not. Once the model starts a call, the rest of it is held to a listed tool and that tool's parameter schema, so calls always parse. `schema.CompileTools` and `engine.ChatPromptTools` do the same for Go programs.
- Resource budgets for apps that embed nollama: `engine.Resources` (or `server.Config` / `engine.LoadOptions`) shares one set of CPU threads across every model and answer, keeps weights, answers' state and caches inside a memory budget (freeing caches first, then failing with `engine.ErrMemoryLimit`, 503 from the server), and can run the engine's threads at a lower OS priority on Linux, macOS and Windows. `nollama serve` gets `--max-memory` and `--low-priority`, and `--threads` is now shared by all models; `NOLLAMA_THREADS`, `NOLLAMA_MAX_MEMORY` and `NOLLAMA_LOW_PRIORITY` set them too.
- Classify: `POST /api/classify` (nollama's own endpoint) and `Model.Classify` in Go take a prompt and candidate labels and return a probability for each, from one read of the prompt instead of generating and parsing an answer. Multi-token labels, `complete` (the label is the whole answer) and `normalize` (per-token scores); the prompt cache applies.
- Smaller KV cache: `nollama serve --kv-cache-type` (or `NOLLAMA_KV_CACHE_TYPE`, or `engine.LoadOptions.KVCacheType`) stores the attention cache as `f16`, `q8_0` or `q4_0` instead of `f32`, cutting its memory to 1/2, about 1/4 or about 1/5 for long contexts (`q4_0` keeps keys at 8 bits: 4-bit keys more than double Qwen3's perplexity). `/api/show` reports it as `nollama.kv_cache_type`.
- Deterministic mode: `engine.LoadOptions.Deterministic` (`nollama serve --deterministic`, `NOLLAMA_DETERMINISTIC`, `nollama run --deterministic`) gives the same logits, bit for bit, on arm64 and amd64, at any thread count and prompt batch size, with portable Go kernels. Sampling, the RoPE tables and the Go fallback kernels no longer depend on the compiler fusing multiply-adds, so their numbers change very slightly everywhere.
- Speculative decoding (experimental: on the CPU engine today it makes answers slower, so it is off unless you ask for it): `nollama serve --draft qwen3-0.6b` (or `NOLLAMA_DRAFT`, `nollama run --draft`) loads a small model beside the one answering to guess a few tokens at a time, which the larger model checks in one pass. With temperature 0 the answer is exactly the one you'd get without it, and with sampling it follows the same distribution. `draft_tokens` in a request's options (0: off) and `--draft-tokens` set how many tokens it guesses. In Go: `engine.GenerateOptions.Draft` and `engine.CheckDraft`.

## v0.0.1-alpha.1

First release.

- `nollama serve` answers Ollama's API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/show`, `/api/pull`, `/api/delete`) and OpenAI's (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), streaming or not, on `127.0.0.1:11434`.
- Structured output: Ollama's `format` (`"json"` or a JSON schema) and OpenAI's `response_format` (`json_object`, `json_schema`) limit the answer to JSON that matches, token by token.
- `nollama run`, `pull`, `list` and `rm` in the terminal.
- Qwen3 models (0.6B, 1.7B, 4B by name; any Qwen3 GGUF from Hugging Face or a file) in Q8_0, Q4_K_M, F16 or BF16, on the CPU, with assembly kernels for ARM64 (NEON, dot product) and x86-64 (AVX2, FMA).
- Go packages to embed it: `server` (the HTTP handler), `engine` (load a model and generate), `models` (find and download models), `schema` (JSON schema constraints).
- Installs with Homebrew, Scoop, `install.sh`, or `.deb`/`.rpm`/`.apk` packages that add a systemd service.
