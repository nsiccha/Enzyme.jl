# Dogfood repair registry

One file per reviewed repair the dogfood line carries, named `<name>.toml`.
`Enzyme.dogfood_repairs()`, `Enzyme.has_dogfood_repair(name)` and
`Enzyme.require_dogfood_repairs(names...; requester)` read this directory
from the Enzyme.jl source that loaded, so a consumer can refuse a dogfood
commit that lacks a repair it needs before it compiles a derivative.

Fields:

- `summary` (required): one line saying what the repair makes work.
- `libenzyme_build` (only for a repair in the native library): the full
  commit of the first dogfood libEnzyme build that carries it. Such a repair
  counts only when the loaded library is the build
  `deps/dogfood_libenzyme.toml` declares, which carries every repair of the
  builds before it.
- `tests` (optional): the regression test files that check it.

A landing that adds a repair adds its file in the same commit. A landing that
removes one, or moves it out of the declared build, deletes it. Names are
lowercase identifiers that describe the repair; they never name a downstream
project. `test/dogfood_repairs.jl` checks every file.
