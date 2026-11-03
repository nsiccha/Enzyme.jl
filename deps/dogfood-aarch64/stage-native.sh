#!/usr/bin/env bash
# Run on the desktop via a reviewed command-on-GO. No build runs here.
set -euo pipefail
test "$#" -eq 2
repo=$1
recipe_sha=$2
source_sha=a5b5ce88390b7d978508439f2074f50818a99a59
resolver_repo=/home/niko/github/nsiccha/Claude
resolver_sha=9154fd455cb7391bdfebb3a8af3f34022774e0b1
[[ "$recipe_sha" =~ ^[0-9a-f]{40}$ ]]

# Check the control link before staging anything. A refused connection needs
# the user to restore maclink; never retry it in a loop.
ssh -n -o BatchMode=yes -o ConnectTimeout=15 mac 'test "$(uname -s)" = Darwin'
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/recipe"
git -C "$repo" archive "$source_sha" > "$stage/source.tar"
test "$(git get-tar-commit-id < "$stage/source.tar")" = "$source_sha"
git -C "$repo" archive "$recipe_sha" deps/dogfood-aarch64 | tar -xf - -C "$stage/recipe"
git -C "$resolver_repo" show "$resolver_sha:lib-resolve.sh" > "$stage/lib-resolve.sh"
# Only emit the maintained public resolver program. Supplying its registry
# variables avoids unrelated repository discovery and Bash dependencies.
GITHUB_DIR=/unused LOCAL_PKG_ORDER='' bash -c \
    'set -euo pipefail; source "$1"; dogfood_libenzyme_program' \
    _ "$stage/lib-resolve.sh" > "$stage/resolver.jl"
test -s "$stage/resolver.jl"
printf 'source_commit=%s\nrecipe_commit=%s\nresolver_commit=%s\n' \
    "$source_sha" "$recipe_sha" "$resolver_sha" > "$stage/PROVENANCE"
(cd "$stage" && shasum -a 256 source.tar resolver.jl > SHA256SUMS)

# The payload contains only named repository sources and the emitted program.
# The Mac returns build/provenance/canary logs, never application data.
tar -cf - -C "$stage" source.tar resolver.jl PROVENANCE SHA256SUMS recipe |
    ssh -o BatchMode=yes -o ConnectTimeout=15 mac \
        'set -eu; export PATH="$HOME/homebrew/bin:$HOME/.juliaup/bin:$PATH"; run="$HOME/.local/state/kb-agents/libenzyme-dogfood-runs/'"$recipe_sha"'"; mkdir -p "$(dirname "$run")"; mkdir "$run"; tar -xf - -C "$run"; bash "$run/recipe/deps/dogfood-aarch64/run-native.sh" "$run"'
