#!/bin/bash
# Build the dogfood libEnzyme line (deps/dogfood_libenzyme.toml) on a Linux x86_64 build host and
# install it beside its provenance:
#   <O>/<sha>/llvm15/  libEnzyme-15.so + libEnzymeBCLoad-15.so against LLVM_full_jll 15.0.7 (Julia 1.10)
#   <O>/<sha>/llvm18/  libEnzyme-18.so + libEnzymeBCLoad-18.so against LLVM_full_jll 18.1.7 (Julia 1.12)
#   <O>/<sha>/llvm20/  libEnzyme-20.so + libEnzymeBCLoad-20.so against LLVM_full_jll 20.1.8+2 (Julia 1.13;
#                      the build Julia 1.13.1's own libLLVM_jll 20.1.8+2 comes from)
# and run the libEnzyme lit tests on an LLVM 16 opt plugin (upstream's lit CI configuration):
#   <O>/<sha>/lit16-summary.txt           the combined gate for test/Enzyme, test/TypeAnalysis and
#                                         test/ActivityAnalysis. A repair needs 0 failed.
#   <O>/<sha>/lit16-broad-summary.txt     the whole test tree, informational: about 303 tests fail on
#                                         a host without the MLIR, clang and Fortran tools.
#   <O>/<sha>/lit16-broad-failed.txt      the broad run's failed and unresolved tests, sorted
#   <O>/<sha>/lit16-broad-new-failures.txt  those that the nearest built ancestor's list lacks
# lit20 runs test/Enzyme on an LLVM 20 plugin (the LLVM a Julia 1.13 consumer loads), informational.
# Both lit targets also build the helper programs their tests run (H below), where the source
# defines them: the default "all" target leaves them out, and a test whose helper is missing fails.
#
# usage: build.sh [targets...]   (default: lib15 lib18 lib20 lit16)
#   KB_DOGFOOD_ROOT  build root <D> (default ~/.local/state/kb-agents/libenzyme-dogfood): compile
#                    trees go to <D>/build/<sha>-<target>, with their logs beside them
#                    (<sha>-lit16.lit.log is the gate's log, <sha>-lit16.lit-broad.log the broad run's)
#   KB_DOGFOOD_SRC   source checkout (default <D>/src; must be a clean tree, enzyme/ inside)
#   KB_DOGFOOD_OUT   install root <O> (default <D>; a staging root installs nothing the resolver reads)
#   KB_DOGFOOD_JOBS  compiler and lit jobs (default 16)
# exit status: 0 after BUILD_OK, which needs every requested target built and installed and the
# lit16 gate passed; 1 after BUILD_FAIL; 2 when the source checkout is missing or dirty.
# The build host runs an installed copy of this file as <D>/build.sh; this file is its source.
set -o pipefail
D=${KB_DOGFOOD_ROOT:-$HOME/.local/state/kb-agents/libenzyme-dogfood}
S=${KB_DOGFOOD_SRC:-$D/src}
SRC=$S/enzyme
SHA=$(git -C $S rev-parse HEAD) || { echo "refusing: $S is not a git checkout"; exit 2; }
DIRTY=$(git -C $S status --porcelain --untracked-files=no | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "refusing: $S has $DIRTY modified tracked files"; exit 2; }
A=$HOME/.julia/artifacts
LLVM15=$A/bc00f0454c7a931d3b569ad1fec7ede3b8ff9788
LLVM16=$A/e4dcdbd63ddce19e5bc4d33cea396ae0a53344d1
LLVM18=$A/4b10dbd3635fbac231d39dcaf88b394ccf5ebd9c
LLVM20=$A/0f9628f1ffef5adbbde95d9a1e7006b3fc7433d0   # LLVM_full_jll 20.1.8+2, x86_64-linux-gnu-cxx11
J=${KB_DOGFOOD_JOBS:-16}
WHAT=${*:-"lib15 lib18 lib20 lit16"}
O=${KB_DOGFOOD_OUT:-$D}
OUT=$O/$SHA; mkdir -p $OUT $D/build
echo "host=${KB_HOST:-?} branch=$(git -C $S rev-parse --abbrev-ref HEAD) sha=$SHA src=$S targets=[$WHAT] script=$(sha256sum < "$0" | cut -c1-12) $(date -u +%FT%TZ)" | tee -a $OUT/BUILD_INFO

# lit's closing report: the failed test names, when there are any, and the counts. A log without
# one (lit was killed, or never started) gives its last lines instead.
lit_report() {
  awk '/^[A-Z][A-Za-z ]* Tests \([0-9]+\):$/ || /^Testing Time:/ { p = 1 } p' "$1" | grep . || tail -15 "$1"
}
lit_counts() {
  awk '/^Testing Time:/ { p = 1 } p' "$1" | grep . || tail -15 "$1"
}
# Every test lit lists by name in its closing report (failed, unresolved, timed out), sorted.
lit_failed() {
  grep -E '^  Enzyme :: ' "$1" | sed 's/^  //' | LC_ALL=C sort
}

fail=0
for w in $WHAT; do
  B=$D/build/$SHA-$w
  case $w in
    lib15) LL=$LLVM15; FLAGS="-DENZYME_EXTERNAL_SHARED_LIB=ON"; T="Enzyme-15 EnzymeBCLoad-15"; V=15; H=;;
    lib18) LL=$LLVM18; FLAGS="-DENZYME_EXTERNAL_SHARED_LIB=ON"; T="Enzyme-18 EnzymeBCLoad-18"; V=18; H=;;
    lib20) LL=$LLVM20; FLAGS="-DENZYME_EXTERNAL_SHARED_LIB=ON"; T="Enzyme-20 EnzymeBCLoad-20"; V=20; H=;;
    # H: test helpers. test/Enzyme loads the call handler plugin (%loadTestCallHandler);
    # test/ActivityAnalysis runs the re-evaluation program.
    lit16) LL=$LLVM16; FLAGS="-DENZYME_CLANG=OFF -DENZYME_EXTERNAL_SHARED_LIB=OFF -DLLVM_EXTERNAL_LIT=$LLVM16/tools/lit"; T="LLVMEnzyme-16"; V=16; H="EnzymeTestCallHandler-16 EnzymeActivityReEvaluationTest";;
    lit20) LL=$LLVM20; FLAGS="-DENZYME_CLANG=OFF -DENZYME_EXTERNAL_SHARED_LIB=OFF -DLLVM_EXTERNAL_LIT=$LLVM20/tools/lit"; T="LLVMEnzyme-20"; V=20; H="EnzymeTestCallHandler-20";;
    *) echo "unknown target $w"; fail=1; continue;;
  esac
  [ -d $LL/lib/cmake/llvm ] || { echo "missing LLVM for $w: $LL (install the LLVM_full_jll artifact first)"; fail=1; continue; }
  echo "== $w ($T) $(date +%T)"
  cmake -DLLVM_DIR=$LL/lib/cmake/llvm -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENZYME_ENABLE_BENCHMARKS=OFF $FLAGS -B$B -S$SRC > $B.cmake.log 2>&1 || { echo "cmake FAILED $w"; tail -20 $B.cmake.log; fail=1; continue; }
  if [ -n "$H" ]; then  # a source older than a helper's tests does not define that helper
    defined=$(cmake --build $B --target help 2>/dev/null)
    for h in $H; do
      case "$defined" in *" $h"*) T="$T $h";; esac
    done
  fi
  cmake --build $B --parallel $J -t $T > $B.build.log 2>&1 || { echo "build FAILED $w"; grep -m5 -B2 -A5 'error' $B.build.log; fail=1; continue; }
  echo "built $w ($T) $(date +%T)"
  LIT="python3 $LL/tools/lit/lit.py -s -j $J --no-progress-bar"
  case $w in
    lit16)
      gate=(test/Enzyme test/TypeAnalysis test/ActivityAnalysis)
      (cd "$B" && timeout 3600 $LIT "${gate[@]}") > $B.lit.log 2>&1; rc=$?
      { echo "lit16 gate: ${gate[*]} on the LLVM 16 plugin, commit $SHA, lit exit $rc"; lit_report $B.lit.log; } | tee $OUT/lit16-summary.txt
      [ $rc -eq 0 ] || { echo "lit16 gate FAILED (lit exit $rc), log $B.lit.log"; fail=1; }
      # The whole tree also exercises BCLoader, MLIR, clang Integration and Fortran.
      # Missing tools can cause failures here; the broad run is informational.
      timeout 3600 $LIT $B/test > $B.lit-broad.log 2>&1; rc=$?
      { echo "lit16 broad suite: the whole test tree, informational, commit $SHA, lit exit $rc"; lit_counts $B.lit-broad.log; } | tee $OUT/lit16-broad-summary.txt
      lit_failed $B.lit-broad.log > $OUT/lit16-broad-failed.txt
      # Compare with the nearest ancestor that has a list, in this install root or the default one.
      rm -f $OUT/lit16-broad-new-failures.txt
      base=
      for a in $(git -C $S rev-list --first-parent -n 50 HEAD^ 2>/dev/null); do
        for r in $O $D; do
          [ -f $r/$a/lit16-broad-failed.txt ] && { base=$r/$a/lit16-broad-failed.txt; break 2; }
        done
      done
      if [ -n "$base" ]; then
        echo "broad suite, failing here but not on ${a:0:8}:"
        LC_ALL=C comm -13 $base $OUT/lit16-broad-failed.txt | tee $OUT/lit16-broad-new-failures.txt
      fi
      ;;
    lit20)   # test/Enzyme on LLVM 20, informational: the tests' CHECK lines are written for LLVM 15/16
             # output, so tag v0.0.301 itself fails 708 of 1099 here (value names, i8* vs ptr, the removed
             # -opaque-pointers flag). Compare $OUT/lit20-failed.txt with the v0.0.301 baseline list
             # <D>/lit20-baseline-v0.0.301-failed.txt: a test failing here and passing there is a regression.
      timeout 3600 $LIT $B/test/Enzyme > $B.lit.log 2>&1; echo "lit rc=$?"
      tail -6 $B.lit.log | tee $OUT/lit20-summary.txt
      lit_failed $B.lit.log > $OUT/lit20-failed.txt
      [ -f $D/lit20-baseline-v0.0.301-failed.txt ] && { echo "failing here but passing on v0.0.301:"; LC_ALL=C comm -13 $D/lit20-baseline-v0.0.301-failed.txt $OUT/lit20-failed.txt | tee $OUT/lit20-new-failures.txt; }
      ;;
    *)
      mkdir -p $OUT/llvm$V &&
        cp $B/Enzyme/libEnzyme-$V.so $B/BCLoad/libEnzymeBCLoad-$V.so $OUT/llvm$V/ &&
        cp $B.build.log $OUT/llvm$V/build.log &&
        (cd $OUT/llvm$V && sha256sum *.so | tee SHA256SUMS) || { echo "install FAILED $w"; fail=1; }
      ;;
  esac
done
[ $fail -eq 0 ] && echo BUILD_OK || echo BUILD_FAIL
exit $fail
