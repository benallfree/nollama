# Ask for the list, never $version.info: asking for a version before its tag exists makes the proxy (and
# sum.golang.org) cache "unknown revision" for about 30 minutes after it is published.
known=$(curl -s -m 10 "https://proxy.golang.org/$module/@v/list" || true)
! grep -qx "$version" <<<"$known" ||#!/usr/bin/env bash
# Cuts a nollama release as a draft on GitHub. Run from a clean checkout of main, in sync with origin:
#
#   1. In CHANGELOG.md, rename "## Unreleased" to "## vX.Y.Z" (and add a new empty Unreleased above it); commit, push.
#   2. ./release.sh vX.Y.Z
#   3. Publish the draft on GitHub (brew and scoop installs fail until you do).
#
# It vets and tests, tags, pushes the tag, and runs goreleaser: zips, Linux packages, checksums, and the Homebrew
# formula (Formula/) and Scoop manifest (bucket/), which goreleaser commits to main in this repo; then it pulls them.
#
# Needs gh (signed in) and goreleaser.
set -euo pipefail

repo=benallfree/nollama
module=github.com/benallfree/nollama
cd "$(dirname "$0")"

step() { echo "$(date +%T) $*" >&2; }
die() { echo "release: $*" >&2; exit 1; }
# macOS has no timeout unless coreutils is installed; then the limits are skipped.
command -v timeout >/dev/null || timeout() { shift; "$@"; }

version=${1:-}
[[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || die "usage: $0 vX.Y.Z"
command -v goreleaser >/dev/null || die "needs goreleaser (brew install goreleaser)"

[ "$(git symbolic-ref --short -q HEAD || true)" = main ] || die "not on main"
[ -z "$(git status --porcelain)" ] || die "the working tree has changes; commit them first"
step "fetching origin"
timeout 30 git fetch -q origin main || die "git fetch origin main failed"
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "HEAD is not origin/main; push or pull first"
sha=$(git rev-parse HEAD)

notes=$(mktemp)
awk -v v="## $version" '$0 == v || index($0, v " ") == 1 {on=1; next} on && /^## / {exit} on' CHANGELOG.md |
  sed -e '/./,$!d' >"$notes"
[ -s "$notes" ] || die "CHANGELOG.md has no \"## $version\" section with notes; write it, commit and push first"

step "vet and test"
go vet ./...
go test ./...

# A tag the Go proxy has seen can never point at another commit, so a version it knows is burned.
step "checking the version is unused"
if git rev-parse -q --verify "refs/tags/$version" >/dev/null; then
  [ "$(git rev-parse "$version^{commit}")" = "$sha" ] || die "tag $version already exists on another commit"
fi
# Ask for the list, never $version.info: asking for a version before its tag exists makes the proxy (and
# sum.golang.org) cache "unknown revision" for about 30 minutes after it's published.
known=$(curl -s -m 10 "https://proxy.golang.org/$module/@v/list" || true)
! grep -qxF "$version" <<<"$known" || [ "$(git rev-parse -q --verify "refs/tags/$version^{commit}" || true)" = "$sha" ] ||
  die "the Go proxy already knows $version; pick the next version"

git rev-parse -q --verify "refs/tags/$version" >/dev/null || git tag -a "$version" -m "nollama $version"
step "pushing tag $version"
timeout 60 git push origin "refs/tags/$version"

step "goreleaser"
export GITHUB_TOKEN="${GITHUB_TOKEN:-$(gh auth token)}"
goreleaser release --clean --release-notes "$notes"

step "pulling the formula and manifest goreleaser committed"
timeout 30 git pull -q --ff-only origin main
step "draft release made for $version: publish it on https://github.com/$repo/releases (brew and scoop installs fail until then)"
