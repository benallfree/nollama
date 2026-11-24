#!/bin/sh
# Before the package is removed: stops and disables the service. Not on an upgrade (deb passes "upgrade", rpm 1).
# The nollama user and /var/lib/nollama (downloaded models) are left in place.
set -e

case "$1" in
upgrade | 1) exit 0 ;;
esac
if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
	systemctl disable --now nollama.service >/dev/null 2>&1 || true
fi
