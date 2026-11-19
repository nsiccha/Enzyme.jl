#!/bin/bash
# Check build.sh's control flow, exit status and lit reports without LLVM and without compiling
# anything: a stub cmake and a stub lit stand in for the real ones, inside a throwaway HOME.
# usage: selftest.sh [path/to/build.sh]     (default: the build.sh beside this file)
# Prints one line per failed check and ends with SELFTEST_OK or SELFTEST_FAIL; exit status 0 or 1.
set -u
here=$(cd "$(dirname "$0")" && pwd)
script=${1:-$here/build.sh}
[ -f "$script" ] || { echo "no such script: $script"; exit 2; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/dogfood-build-selftest.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

home=$tmp/home
D=$home/.local/state/kb-agents/libenzyme-dogfood
mkdir -p "$D" "$tmp/bin"
cp "$script" "$D/build.sh"
git_() { GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 git -c user.name=selftest -c user.email=selftest@example.invalid "$@"; }

# A source checkout with two commits, so the second has a built ancestor to compare with.
git_ init -q "$D/src"
mkdir "$D/src/enzyme"
echo "project(Enzyme)" > "$D/src/enzyme/CMakeLists.txt"
git_ -C "$D/src" add enzyme && git_ -C "$D/src" commit -q -m first
c1=$(git_ -C "$D/src" rev-parse HEAD)
echo "# second" >> "$D/src/enzyme/CMakeLists.txt"
git_ -C "$D/src" commit -q -a -m second
c2=$(git_ -C "$D/src" rev-parse HEAD)

# The LLVM_full_jll artifact directories the script names, each with a stub lit.
cat > "$tmp/lit.py" <<'STUB_LIT'
import argparse, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-s", action="store_true")
parser.add_argument("-j", type=int)
parser.add_argument("--no-progress-bar", action="store_true")
parser.add_argument("roots", nargs="+", type=Path)
roots = [root.resolve() for root in parser.parse_args().roots]
broad = len(roots) == 1 and roots[0].name == "test"
test_dir = roots[0] if broad else roots[0].parent
suites = [root.name for root in roots]
suite_passes = {"Enzyme": 1195, "TypeAnalysis": 96, "ActivityAnalysis": 37}
passes = 1402 if broad else sum(suite_passes[suite] for suite in suites)
helpers_defined = os.environ.get("STUB_HELPERS", "1") == "1"
handler_built = any(f.startswith("EnzymeTestCallHandler-") for f in os.listdir(test_dir))
reeval_built = os.path.exists(os.path.join(test_dir, "ActivityAnalysis", "EnzymeActivityReEvaluationTest"))
if os.environ.get("STUB_LIT_NO_REPORT") == "1":
    print("stub lit: killed before its report")
    sys.exit(2)
regressions = []
if helpers_defined and not handler_built:
    regressions.append("Enzyme/ReverseMode/cache_for_reverse_arg.ll")
if helpers_defined and not reeval_built:
    regressions.append("ActivityAnalysis/reevaluation.cpp")
if os.environ.get("STUB_REGRESSION"):
    regressions.append(os.environ["STUB_REGRESSION"])
failed = ["Enzyme :: " + test for test in regressions if broad or test.split("/")[0] in suites]
passes -= len(failed)
unresolved = []
if broad:
    failed += ["Enzyme :: MLIR/ReverseMode/square.mlir", "Enzyme :: Integration/ReverseMode/sum.c",
               "Enzyme :: Fortran/ForwardMode/square.f90"]
    unresolved = ["Enzyme :: test_find_package/main.c"]
for t in failed:
    print("FAIL: %s (1 of 1)" % t)
for name, tests in (("Unresolved", unresolved), ("Failed", sorted(failed))):
    if tests:
        print("********************\n%s Tests (%d):" % (name, len(tests)))
        print("".join("  %s\n" % t for t in tests))
print("\nTesting Time: 0.01s")
if broad:
    print("  Unsupported      :   23")
print("  Passed           : %4d" % passes)
print("  Expectedly Failed: %4d" % (17 if broad else 10))
if unresolved:
    print("  Unresolved       : %4d" % len(unresolved))
if failed:
    print("  Failed           : %4d" % len(failed))
sys.exit(1 if failed or unresolved else 0)
STUB_LIT
for a in $(grep -oE '\$A/[0-9a-f]{40}' "$script" | sed 's/^\$A\///'); do
  mkdir -p "$home/.julia/artifacts/$a/lib/cmake/llvm" "$home/.julia/artifacts/$a/tools/lit"
  cp "$tmp/lit.py" "$home/.julia/artifacts/$a/tools/lit/lit.py"
done

# The stub cmake: logs its arguments, lists targets, and "builds" by creating the files a real
# build leaves where the script looks for them.
cat > "$tmp/bin/cmake" <<'STUB_CMAKE'
#!/bin/bash
echo "cmake $*" >> "$STUB_LOG"
if [ "$1" != --build ]; then
  for a in "$@"; do case $a in -B*) mkdir -p "${a#-B}";; esac; done
  exit 0
fi
B=$2; name=${B##*-}; V=${name#???}
case " $* " in
  *" --target help "*)
    echo "The following are some of the valid targets for this Makefile:"
    echo "... all (the default if no target is provided)"
    [ "${STUB_HELPERS:-1}" = 1 ] && echo "... EnzymeActivityReEvaluationTest" && echo "... EnzymeTestCallHandler-$V"
    echo "... LLVMEnzyme-$V"
    exit 0;;
esac
[ "$name" = "${STUB_FAIL_BUILD:-}" ] && { echo "stub.cpp:1:1: error: stub compile failure"; exit 1; }
case $name in
  lib*) mkdir -p $B/Enzyme $B/BCLoad; echo "enzyme $V" > $B/Enzyme/libEnzyme-$V.so; echo "bcload $V" > $B/BCLoad/libEnzymeBCLoad-$V.so;;
  lit*) mkdir -p $B/test/{Enzyme,TypeAnalysis,ActivityAnalysis}
        case " $* " in *" EnzymeTestCallHandler-$V "*) : > $B/test/EnzymeTestCallHandler-$V.so;; esac
        case " $* " in *" EnzymeActivityReEvaluationTest "*) : > $B/test/ActivityAnalysis/EnzymeActivityReEvaluationTest;; esac;;
esac
exit 0
STUB_CMAKE
chmod +x "$tmp/bin/cmake"
export STUB_LOG=$tmp/cmake.log

# run <stub settings...> -- <build.sh arguments...>: sets $rc and $out, starts a fresh cmake log.
run() {
  local settings=()
  while [ "$1" != -- ]; do settings+=("$1"); shift; done
  shift
  : > "$STUB_LOG"
  out=$(env -u BASH_ENV -u KB_DOGFOOD_ROOT -u KB_DOGFOOD_SRC -u KB_DOGFOOD_OUT -u STUB_HELPERS \
        -u STUB_REGRESSION -u STUB_FAIL_BUILD -u STUB_LIT_NO_REPORT HOME="$home" PATH="$tmp/bin:$PATH" \
        KB_DOGFOOD_JOBS=2 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 "${settings[@]}" bash "$D/build.sh" "$@" 2>&1)
  rc=$?
}
checks=0; bad=0
ok() {  # ok <description> <command...>
  local what=$1; shift
  checks=$((checks + 1))
  "$@" > /dev/null 2>&1 || { bad=$((bad + 1)); echo "FAILED: $what"; }
}
# The counts a reader of a summary takes: the first "Passed" line, and "Failed" or 0 without one.
passed() { grep -m1 -E '^\s*Passed\s*:\s*[0-9]+' "$1" 2> /dev/null | grep -oE '[0-9]+$'; }
failed() { local f; f=$(grep -m1 -E '^\s*Failed\s*:\s*[0-9]+' "$1" 2> /dev/null | grep -oE '[0-9]+$'); echo "${f:-0}"; }
built_with() { grep -qE -- "--build $D/build/$1-$2 .* -t $3\$" "$STUB_LOG"; }
last_line_is() { [ "$(printf '%s\n' "$out" | tail -1)" = "$1" ]; }

# 1. Every default target on the first commit.
git_ -C "$D/src" checkout -q --detach "$c1"
run --
O=$D/$c1
ok "default targets: exit status 0" [ $rc -eq 0 ]
ok "default targets: ends with BUILD_OK" last_line_is BUILD_OK
ok "lit16 builds both test helpers beside the plugin" built_with $c1 lit16 "LLVMEnzyme-16 EnzymeTestCallHandler-16 EnzymeActivityReEvaluationTest"
ok "lit16-summary.txt opens by naming all gated directories" grep -Fxq "lit16 gate: test/Enzyme test/TypeAnalysis test/ActivityAnalysis on the LLVM 16 plugin, commit $c1, lit exit 0" <(head -1 $O/lit16-summary.txt)
ok "lit16-summary.txt reports the gate's combined passes" [ "$(passed $O/lit16-summary.txt)" = 1328 ]
ok "lit16-summary.txt reports no failure" [ "$(failed $O/lit16-summary.txt)" = 0 ]
ok "lit16-broad-summary.txt reports the broad passes" [ "$(passed $O/lit16-broad-summary.txt)" = 1402 ]
ok "lit16-broad-summary.txt reports the broad failures" [ "$(failed $O/lit16-broad-summary.txt)" = 3 ]
ok "lit16-broad-failed.txt lists the failed and unresolved tests" [ "$(cat $O/lit16-broad-failed.txt 2> /dev/null | wc -l)" -eq 4 ]
ok "lit16-broad-failed.txt is sorted" env LC_ALL=C sort -c $O/lit16-broad-failed.txt
ok "no helper-dependent test is among the broad failures" bash -c "[ -f $O/lit16-broad-failed.txt ] && ! grep -q -e cache_for_reverse_arg -e reevaluation $O/lit16-broad-failed.txt"
ok "no comparison without a built ancestor" [ ! -e $O/lit16-broad-new-failures.txt ]
for v in 15 18 20; do
  ok "llvm$v libraries installed with matching SHA256SUMS" bash -c "cd $O/llvm$v && sha256sum -c SHA256SUMS"
done
ok "BUILD_INFO records the commit" grep -q "sha=$c1" $O/BUILD_INFO

# 2. The next commit adds a failure in a directory outside the gate.
git_ -C "$D/src" checkout -q --detach "$c2"
run STUB_REGRESSION=BCLoader/stub_new.ll -- lit16
O=$D/$c2
ok "new broad failure: exit status stays 0" [ $rc -eq 0 ]
ok "new broad failure: listed against the built ancestor" [ "$(cat $O/lit16-broad-new-failures.txt 2> /dev/null)" = "Enzyme :: BCLoader/stub_new.ll" ]
ok "new broad failure: the output names the ancestor" grep -q "failing here but not on ${c1:0:8}:" <<< "$out"
ok "new broad failure: the gate summary is unchanged" [ "$(passed $O/lit16-summary.txt)/$(failed $O/lit16-summary.txt)" = 1328/0 ]

# 3. A single failure in any gated directory fails the build.
for test in Enzyme/ReverseMode/stub_regression.ll TypeAnalysis/stub_regression.ll ActivityAnalysis/reevaluation.cpp; do
  run STUB_REGRESSION="$test" -- lit16
  ok "$test: exit status 1" [ $rc -eq 1 ]
  ok "$test: ends with BUILD_FAIL" last_line_is BUILD_FAIL
  ok "$test: the summary counts the failure" [ "$(passed $O/lit16-summary.txt)/$(failed $O/lit16-summary.txt)" = 1327/1 ]
  ok "$test: the summary names the test" grep -Fxq "  Enzyme :: $test" $O/lit16-summary.txt
  ok "$test: the summary records lit's exit status" grep -q 'lit exit 1$' <(head -1 $O/lit16-summary.txt)
done

# 4. lit dies before its report.
run STUB_LIT_NO_REPORT=1 -- lit16
ok "lit without a report: exit status 1" [ $rc -eq 1 ]
ok "lit without a report: the summary has no pass count" [ -z "$(passed $O/lit16-summary.txt)" ]
ok "lit without a report: the summary keeps the log's last lines" grep -q 'killed before its report' $O/lit16-summary.txt

# 5. A source that defines no test helper (older than the tests that use them).
run STUB_HELPERS=0 -- lit16
ok "no helper targets: only the plugin is built" built_with $c2 lit16 "LLVMEnzyme-16"
ok "no helper targets: exit status 0" [ $rc -eq 0 ]

# 6. A compile failure in one target: the others still build, the status says so.
run STUB_FAIL_BUILD=lib18 -- lib15 lib18 lit16
ok "compile failure: exit status 1" [ $rc -eq 1 ]
ok "compile failure: ends with BUILD_FAIL" last_line_is BUILD_FAIL
ok "compile failure: the failed target installs nothing" [ ! -e $O/llvm18 ]
ok "compile failure: the other library is installed" [ -f $O/llvm15/SHA256SUMS ]
ok "compile failure: the gate still ran" [ "$(passed $O/lit16-summary.txt)" = 1328 ]

# 7. lit20 builds the helper test/Enzyme needs, and stays informational.
run STUB_REGRESSION=Enzyme/ReverseMode/stub_regression.ll -- lit20
ok "lit20 builds the test call handler and no other helper" built_with $c2 lit20 "LLVMEnzyme-20 EnzymeTestCallHandler-20"
ok "lit20: a failing test does not change the exit status" [ $rc -eq 0 ]
ok "lit20-failed.txt lists the failing test" grep -q stub_regression $O/lit20-failed.txt
ok "lit20 still runs test/Enzyme alone" [ "$(passed $O/lit20-summary.txt)" = 1194 ]

# 8. Refusals.
run -- nosuch
ok "unknown target: exit status 1" [ $rc -eq 1 ]
echo "// edited" >> "$D/src/enzyme/CMakeLists.txt"
run -- lit16
ok "dirty source: exit status 2" [ $rc -eq 2 ]
ok "dirty source: nothing is configured" [ ! -s "$STUB_LOG" ]

echo "$checks checks, $bad failed ($script)"
[ $bad -eq 0 ] && echo SELFTEST_OK || echo SELFTEST_FAIL
[ $bad -eq 0 ]
