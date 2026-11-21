#!/bin/sh
# After the package is installed or upgraded (.deb, .rpm, .apk). Makes the nollama user and /var/lib/nollama, and
# restarts the service if it was running. A new install leaves the service stopped until you enable it.
set -e

if ! id nollama >/dev/null 2>&1; then
	if command -v useradd >/dev/null 2>&1; then
		useradd --system --user-group --home-dir /var/lib/nollama --no-create-home \
			--shell "$(command -v nologin || echo /bin/false)" nollama
	elif command -v adduser >/dev/null 2>&1; then # BusyBox (Alpine)
		addgroup -S nollama 2>/dev/null || true
		adduser -S -D -H -h /var/lib/nollama -s /sbin/nologin -G nollama nollama
	fi
fi
install -d -o nollama -g nollama -m 750 /var/lib/nollama

if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
	systemctl daemon-reload || true
	if systemctl is-enabled --quiet nollama.service 2>/dev/null; then
		systemctl try-restart nollama.service || true
	else
		cat <<'MSG'

nollama is installed. To run it as a service on 127.0.0.1:11434:

  sudo systemctl enable --now nollama
  curl http://127.0.0.1:11434/api/pull -d '{"model":"qwen3-1.7b"}'

Settings: /etc/nollama/nollama.env
MSG
	fi
fi
