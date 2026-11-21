#!/bin/sh
# Run `bench vps` on a fresh Vultr instance and destroy it afterwards.
#
#   VULTR_API_KEY=...  VULTR_BENCH_APPROVED=vc2-1c-1gb,vhp-2c-4gb-amd  scripts/vultr-bench/vultr.sh all vc2-1c-1gb [region] [--ollama]
#
# Steps, each one blocking call that ends in seconds (except `all`, meant to run detached with its output in a log):
#   create PLAN [REGION] [--ollama]  boot an instance (cloud-init downloads the models) -> prints the run dir
#   wait RUN                          until SSH answers and setup finished (bounded, prints every attempt)
#   push RUN                          copy the Linux bench binary (built here, from this checkout)
#   start RUN                         start the benchmark detached on the box
#   status RUN                        tail the box's log; says DONE when results are there
#   fetch RUN                         copy results.json/results.md/logs into the run dir
#   destroy RUN                       delete the instance and check it is gone
#   list                              every instance labelled nollama-bench-* (anything left running)
#   all PLAN [REGION] [--ollama]      every step above, destroying the instance on exit, success or not
#   resume RUN                        rerun on a box that's already set up (fresh binary), fetch, destroy
#
# The API key comes from VULTR_API_KEY (or the macOS Keychain item vultr-api-key) and reaches curl on stdin, never
# on a command line or in a file.
# Plans must be listed in VULTR_BENCH_APPROVED, so nothing is created that wasn't agreed to. Run dirs (instance id,
# address, throwaway SSH key, results) live under $NOLLAMA_BENCH_DIR (default ~/.cache/nollama-vultr-bench).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE=${NOLLAMA_BENCH_DIR:-$HOME/.cache/nollama-vultr-bench}
OS_ID=${VULTR_OS_ID:-2284} # Ubuntu 24.04 LTS x64
say() { echo "$(date +%T) $*"; }
die() { say "error: $*" >&2; exit 1; }

# The key: VULTR_API_KEY, or else (macOS) the login Keychain item "vultr-api-key", read into memory only.
if [ -z "${VULTR_API_KEY:-}" ] && command -v security >/dev/null 2>&1; then
	VULTR_API_KEY=$(security find-generic-password -s vultr-api-key -w 2>/dev/null || true)
fi

api() { # METHOD PATH [JSON]
	[ -n "${VULTR_API_KEY:-}" ] || die "no API key: set VULTR_API_KEY or add Keychain item vultr-api-key"
	if [ $# -ge 3 ]; then
		printf 'header = "Authorization: Bearer %s"\n' "$VULTR_API_KEY" |
			curl -sS -m "${API_TIMEOUT:-30}" -K - -X "$1" -H 'Content-Type: application/json' --data-binary "$3" "https://api.vultr.com/v2$2"
	else
		printf 'header = "Authorization: Bearer %s"\n' "$VULTR_API_KEY" |
			curl -sS -m "${API_TIMEOUT:-30}" -K - -X "$1" "https://api.vultr.com/v2$2"
	fi
}

ids_by_label() { # LABEL -> instance ids with that label, one per line
	api GET '/instances?per_page=500' | json '"\n".join(i["id"] for i in d["instances"] if i["label"]=="'"$1"'")'
}
json() { python3 -I -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

SSHOPTS="-o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
rssh() { # RUN cmd...
	r=$1; shift
	timeout 60 ssh $SSHOPTS -i "$r/key" "root@$(cat "$r/ip")" "$@"
}

cmd_create() {
	plan=$1 region=${2:-ewr} ollama=0
	[ "${3:-}" = --ollama ] && ollama=1
	case ",${VULTR_BENCH_APPROVED:-}," in *",$plan,"*) ;; *) die "plan $plan is not in VULTR_BENCH_APPROVED" ;; esac
	label="nollama-bench-$plan"
	existing=$(ids_by_label "$label") || die "can't list instances"
	[ -z "$existing" ] || die "an instance labelled $label already exists ($existing): destroy it first"
	run=${RUN:-"$BASE/$(date +%Y%m%d-%H%M%S)-$plan"}
	mkdir -p "$run"
	echo "$plan" >"$run/plan"
	ssh-keygen -q -t ed25519 -N '' -C nollama-bench -f "$run/key"
	pub=$(cat "$run/key.pub")
	ud=$(sed -e "s|__PUBKEY__|$pub|" -e "s|__OLLAMA__|$ollama|" "$HERE/box-setup.sh" | base64 | tr -d '\n')
	body=$(printf '{"region":"%s","plan":"%s","os_id":%s,"label":"%s","hostname":"%s","user_data":"%s","backups":"disabled","enable_ipv6":false}' \
		"$region" "$plan" "$OS_ID" "$label" "$label" "$ud")
	say "creating $plan in $region (ollama=$ollama)"
	out=$(API_TIMEOUT=90 api POST /instances "$body" 2>&1) || true
	if ! id=$(echo "$out" | json 'd["instance"]["id"]' 2>/dev/null); then
		# the request may have failed after Vultr made the instance: look for it by label
		say "create didn't answer cleanly ($out); looking for the instance by label"
		sleep 10
		id=$(ids_by_label "$label" | head -1)
		[ -n "$id" ] || die "create failed and no $label instance exists"
		say "found $id: adopting it"
	fi
	echo "$id" >"$run/id"
	say "created instance $id; run dir $run"
	echo "$run"
}

cmd_wait() {
	run=$1 limit=${2:-1500} t0=$(date +%s)
	id=$(cat "$run/id")
	while :; do
		el=$(($(date +%s) - t0))
		[ "$el" -gt "$limit" ] && die "gave up after ${el}s"
		if [ ! -s "$run/ip" ]; then
			st=$(api GET "/instances/$id" | json 'd["instance"]["status"]+" "+d["instance"]["power_status"]+" "+d["instance"]["main_ip"]')
			say "[$el s] instance: ${st% *}"
			ip=${st##* }
			case "$st" in "active running "*) [ "$ip" != 0.0.0.0 ] && echo "$ip" >"$run/ip" ;; esac
		else
			s=$(rssh "$run" 'test -e /root/bench/failed && echo failed || { test -e /root/bench/ready && echo ready || { tail -1 /var/log/nollama-bench-setup.log 2>/dev/null || echo booting; }; }' 2>&1 || echo "ssh not up yet")
			say "[$el s] box: $s"
			case "$s" in ready) say "ready after ${el}s"; return 0 ;; failed) rssh "$run" 'cat /var/log/nollama-bench-setup.log' || true; die "setup failed" ;; esac
		fi
		sleep 10
	done
}

cmd_push() {
	run=$1
	say "building bench for linux/amd64"
	(cd "$REPO" && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$run/bench" ./engine/cmd/bench)
	say "copying bench ($(du -m "$run/bench" | cut -f1) MB)"
	timeout 120 scp $SSHOPTS -i "$run/key" "$run/bench" "root@$(cat "$run/ip"):/root/bench/bench"
	say "pushed"
}

cmd_start() {
	run=$1
	ollama=""
	rssh "$run" 'command -v ollama >/dev/null' && ollama=-ollama
	plan=$(cat "$run/plan")
	# only the setsid'd command goes to the background (a backgrounded `cd && ...` list would keep the SSH channel open)
	rssh "$run" "cd /root/bench || exit 1; setsid nohup ./bench vps -models \"\$(cat models.arg)\" $ollama -label $plan -json results.json </dev/null >results.md 2>bench.log & echo started"
}

cmd_status() {
	run=$1
	# pidof matches the process name only (pgrep -f would match this very shell, whose command line names it)
	rssh "$run" 'cd /root/bench; tail -3 bench.log; if pidof bench >/dev/null; then echo RUNNING; elif [ -s results.md ]; then echo DONE; else echo STOPPED; fi'
}

cmd_fetch() {
	run=$1
	for f in results.json results.md bench.log; do
		timeout 60 scp $SSHOPTS -i "$run/key" "root@$(cat "$run/ip"):/root/bench/$f" "$run/$f" || say "no $f"
	done
	timeout 60 scp $SSHOPTS -i "$run/key" "root@$(cat "$run/ip"):/var/log/nollama-bench-setup.log" "$run/setup.log" || true
	say "fetched into $run"
}

cmd_destroy() {
	run=$1
	id=$(cat "$run/id" 2>/dev/null) || id=""
	if [ -z "$id" ] && [ -s "$run/plan" ]; then
		id=$(ids_by_label "nollama-bench-$(cat "$run/plan")" | head -1) || id=""
		[ -n "$id" ] && echo "$id" >"$run/id"
	fi
	[ -n "$id" ] || { say "no instance to destroy for $run"; return 0; }
	say "destroying $id"
	api DELETE "/instances/$id" >/dev/null || true
	for i in 1 2 3 4 5 6; do
		code=$(printf 'header = "Authorization: Bearer %s"\n' "$VULTR_API_KEY" |
			curl -s -m 15 -K - -o /dev/null -w '%{http_code}' "https://api.vultr.com/v2/instances/$id")
		[ "$code" = 404 ] && { say "instance $id is gone"; mv "$run/id" "$run/id.destroyed"; return 0; }
		say "still there (HTTP $code), retrying delete ($i)"
		api DELETE "/instances/$id" >/dev/null || true
		sleep 5
	done
	die "instance $id may still exist: check the Vultr dashboard"
}

cmd_list() {
	api GET '/instances?per_page=500' | json '"\n".join(i["id"]+" "+i["label"]+" "+i["plan"]+" "+i["status"]+" since "+i["date_created"] for i in d["instances"] if i["label"].startswith("nollama-bench-")) or "no nollama-bench instances"'
}

# poll until the benchmark ends, then fetch
watch_and_fetch() {
	run=$1 t0=$(date +%s)
	while :; do
		sleep 30
		s=$(cmd_status "$run" 2>&1 || echo "status failed")
		say "[$(($(date +%s) - t0)) s] $(echo "$s" | tr '\n' ' ')"
		case "$s" in *DONE*) break ;; *STOPPED*) say "benchmark stopped without results"; break ;; esac
		[ $(($(date +%s) - t0)) -gt "${BENCH_MAX_S:-5400}" ] && { say "benchmark over its time limit"; break; }
	done
	cmd_fetch "$run"
	cat "$run/results.md" 2>/dev/null || true
}

# rerun on a box that's already set up (new binary), then fetch and destroy
cmd_resume() {
	run=$1
	trap 'cmd_destroy "$run"' EXIT INT TERM HUP
	rssh "$run" 'pkill -x bench; cd /root/bench && rm -f results.md results.json bench.log; echo cleared'
	cmd_push "$run"
	cmd_start "$run"
	watch_and_fetch "$run"
}

cmd_all() {
	run="$BASE/$(date +%Y%m%d-%H%M%S)-$1"
	mkdir -p "$run"
	echo "$1" >"$run/plan"
	trap 'cmd_destroy "$run"' EXIT INT TERM HUP
	RUN=$run cmd_create "$@"
	cmd_wait "$run"
	cmd_push "$run"
	cmd_start "$run"
	watch_and_fetch "$run"
}

c=${1:-}
[ -n "$c" ] || { sed -n 2,22p "$0"; exit 2; }
shift
case "$c" in
create | wait | push | start | status | fetch | destroy | list | all | resume) "cmd_$c" "$@" ;;
*) die "unknown step $c" ;;
esac
