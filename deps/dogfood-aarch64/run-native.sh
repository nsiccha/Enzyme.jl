#!/usr/bin/env bash
# Invoked only by the reviewed stage-native.sh payload.
set -euo pipefail
test "$#" -eq 1
run=$1
recipe="$run/recipe/deps/dogfood-aarch64"
source_sha=a5b5ce88390b7d978508439f2074f50818a99a59
backend_sha=1c92729e08a0a8ba05a6f6e0843440467c24981a
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
for tool in julia cmake git shasum; do command -v "$tool"; done
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ENZYME_EXPECT_ARCH=aarch64
unset JULIA_PROJECT KB_LIBENZYME_DOGFOOD_ROOT
julia --startup-file=no --project=@stdlib -e \
    'println("NATIVE_JULIA ", VERSION, " LLVM ", Base.libllvm_version, " ", Sys.MACHINE); Sys.ARCH == :aarch64 || error("Native aarch64 required"); Base.libllvm_version.major == 15 || error("Recipe requires LLVM 15; report this runtime before selecting a new build recipe")'
(cd "$run" && shasum -a 256 -c SHA256SUMS)
cat "$run/PROVENANCE"
test "$(git get-tar-commit-id < "$run/source.tar")" = "$source_sha"
mkdir "$run/source"
tar -xf "$run/source.tar" -C "$run/source"
export KB_JOB_STATE="$run/build-state" KB_JOB_WORKSPACE="$run"
export KB_JOB_CPU_SHARE=4
export KB_LIBENZYME_BUILD_DEST="$HOME/.local/state/kb-agents/libenzyme-dogfood"
out="$KB_LIBENZYME_BUILD_DEST/$backend_sha/llvm15"
# Preserve an existing build until its provenance has been reviewed.
test ! -e "$out"
bash "$recipe/build-llvm15.sh"
julia --startup-file=no --project=@stdlib "$recipe/setup.jl" \
    "$run/env" "$run/source" "$run/resolver.jl" | tee "$run/setup.log"
grep -Fx "$(printf 'KB-DOGFOOD-LIBENZYME\t%s\tllvm15\t%s' "$backend_sha" "$out")" "$run/setup.log"
julia --startup-file=no --project="$run/env" "$recipe/canaries.jl" | tee "$run/canaries.log"
grep -Fx CANARIES_COMPLETE "$run/canaries.log"
(cd "$out" && shasum -a 256 -c SHA256SUMS)
printf 'NATIVE_COMPLETE source=%s backend=%s llvm=15\n' "$source_sha" "$backend_sha" | tee "$run/VALIDATED"
test -s "$run/VALIDATED"
