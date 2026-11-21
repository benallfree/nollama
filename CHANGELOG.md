# Changelog

Every user-visible change gets a line under **Unreleased** in the commit that makes it. At a release, `## Unreleased`
becomes `## vX.Y.Z` and its lines are the release notes (`release.sh` reads them).

## Unreleased

## v0.0.1-alpha.1

First release.

- `nollama serve` answers Ollama's API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/show`, `/api/pull`, `/api/delete`) and OpenAI's (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), streaming or not, on `127.0.0.1:11434`.
- Structured output: Ollama's `format` (`"json"` or a JSON schema) and OpenAI's `response_format` (`json_object`, `json_schema`) limit the answer to JSON that matches, token by token.
- `nollama run`, `pull`, `list` and `rm` in the terminal.
- Qwen3 models (0.6B, 1.7B, 4B by name; any Qwen3 GGUF from Hugging Face or a file) in Q8_0, Q4_K_M, F16 or BF16, on the CPU, with assembly kernels for ARM64 (NEON, dot product) and x86-64 (AVX2, FMA).
- Go packages to embed it: `server` (the HTTP handler), `engine` (load a model and generate), `models` (find and download models), `schema` (JSON schema constraints).
- Installs with Homebrew, Scoop, `install.sh`, or `.deb`/`.rpm`/`.apk` packages that add a systemd service.
