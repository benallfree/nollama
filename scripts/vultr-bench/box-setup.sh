#!/bin/sh
# cloud-init user-data for a benchmark box (vultr.sh fills in the two placeholders). Runs as root at first boot:
# authorizes the run's SSH key, downloads the models that fit in RAM, optionally installs Ollama, then writes
# /root/bench/ready (or /root/bench/failed). Log: /var/log/nollama-bench-setup.log
# DRY_RUN=1 BENCH_DIR=... SETUP_LOG=... RAM_MB=... runs it anywhere without downloading or installing (for testing).
set -u
B=${BENCH_DIR:-/root/bench}
exec >"${SETUP_LOG:-/var/log/nollama-bench-setup.log}" 2>&1
say() { echo "$(date +%T) $*"; }
# any exit before "ready" (including a shell error) tells the driver at once
trap '[ -e "$B/ready" ] || { say "setup exited before ready"; touch "$B/failed"; }' EXIT
DRY=${DRY_RUN:-0}

if [ "$DRY" != 1 ]; then
	mkdir -p /root/.ssh && chmod 700 /root/.ssh
	echo "__PUBKEY__" >>/root/.ssh/authorized_keys
	chmod 600 /root/.ssh/authorized_keys
fi
WITH_OLLAMA="__OLLAMA__"

mkdir -p "$B/models"
cd "$B/models" || exit 1
ram_mb=${RAM_MB:-$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)}
say "setup start, ${ram_mb} MB RAM"

fail() { say "FAILED: $*"; touch "$B/failed"; exit 1; }

fetch() { # url file
	[ -s "$2" ] && return 0
	[ "$DRY" = 1 ] && { say "would download $1 -> $2"; return 0; }
	say "download $2"
	curl -fsSL --retry 3 --connect-timeout 10 -m 1800 -o "$2.part" "$1" || return 1
	mv "$2.part" "$2"
	say "done $2 ($(du -m "$2" | cut -f1) MB)"
}

ollama_blob_url() { # tag -> blob URL of Ollama's qwen3:<tag> model layer
	curl -fsS -m 20 -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
		"https://registry.ollama.ai/v2/library/qwen3/manifests/$1" |
		python3 -c 'import json,sys; m=json.load(sys.stdin); print([l["digest"] for l in m["layers"] if l["mediaType"].endswith("image.model")][0])' |
		sed 's|^|https://registry.ollama.ai/v2/library/qwen3/blobs/|'
}

HF=https://huggingface.co/Qwen
args=""
add() { # label file size_mb url
	if [ $(($3 + 350)) -gt "$ram_mb" ]; then
		say "skip $1: $3 MB doesn't fit in ${ram_mb} MB"
		return
	fi
	[ -n "$4" ] || fail "no URL for $1"
	fetch "$4" "$2" || fail "download $2"
	args="${args:+$args,}$1=$B/models/$2"
}

add qwen3-0.6b:Q4_K_M qwen3-0.6b-Q4_K_M.gguf 499 "$(ollama_blob_url 0.6b)"
add qwen3-0.6b:Q8_0 qwen3-0.6b-Q8_0.gguf 610 "$HF/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
add qwen3-1.7b:Q4_K_M qwen3-1.7b-Q4_K_M.gguf 1300 "$(ollama_blob_url 1.7b)"
add qwen3-1.7b:Q8_0 qwen3-1.7b-Q8_0.gguf 1750 "$HF/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf"
add qwen3-4b:Q4_K_M qwen3-4b-Q4_K_M.gguf 2382 "$HF/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
add qwen3-4b:Q8_0 qwen3-4b-Q8_0.gguf 4083 "$HF/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf"
echo "$args" >"$B/models.arg"
say "models: $args"

if [ "$WITH_OLLAMA" = 1 ] && [ "$DRY" != 1 ]; then
	say "installing Ollama"
	curl -fsSL -m 600 https://ollama.com/install.sh | sh || fail "ollama install"
	# cloud-init runs while systemd is still booting, so the installer writes the service without starting it
	systemctl daemon-reload || true
	if ! systemctl enable --now ollama; then
		say "systemctl couldn't start ollama; running ollama serve directly"
		setsid nohup ollama serve </dev/null >/var/log/ollama.log 2>&1 &
	fi
	for i in $(seq 1 30); do
		curl -fsS -m 2 http://127.0.0.1:11434/api/version && break
		say "waiting for ollama ($i)"
		sleep 2
	done
	curl -fsS -m 2 http://127.0.0.1:11434/api/version || fail "ollama not serving"
fi

touch "$B/ready"
say "ready"
