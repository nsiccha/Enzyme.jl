# Changelog

## Unreleased

- Support literal constant objects captured beside active data, including a
  dictionary inside a callback that reads active views. Static differentiation
  now gives these objects a separate zero-filled shadow when one is needed,
  instead of raising `EnzymeRuntimeActivityError`.
- Preserve input shadows when a function returns newly initialized storage
  (dogfood libEnzyme build `ef0cd634`). Reverse differentiation through a
  non-inlined array gather could segfault because the compiler treated those
  writes as invisible outside the call and discarded the input's shadow.
  Returning a pointer loaded from a new object also retains the writes.
- Stop copying everything activity analysis knows into every hypothesis
  (dogfood libEnzyme build `c9796794`). Each hypothesis now shares its sets
  with the analyzer that created it, importing one walks only what it added,
  and a search of memory asks alias analysis only about the memory accesses
  whose answer can matter. On a 97,303-instruction function this cuts the time
  in activity analysis 5- to 9-fold and the first gradient by roughly 6%. Its
  verdicts and the gradients are unchanged. The alias queries that remain still
  grow quadratically with the size of the function, and so does most of the rest
  of the first gradient.
- Support custom rules that return an input or its shadow as a cached value
  (dogfood libEnzyme build `ef75bdc7`, upstream EnzymeAD/Enzyme#3206).
  Reverse differentiation through a dynamic `setproperty!` whose return is used
  now computes the gradient instead of aborting in the compiler.
- Type a stack slot that holds only floats, such as an `NTuple{N, Float64}`
  a loop reads at a runtime index. Reverse raised `EnzymeNoTypeError`
  ("Cannot deduce type of phi") where a function merged a float with a reload
  of an element of such a slot that nothing else typed.
- Release integer loop-count caches after Forward over Reverse through nested
  dynamic loops (dogfood libEnzyme build `c1743bf9`). The outer Forward pass
  removed their original deallocation because the pointer needed no shadow,
  so repeated Hessian-vector products retained native memory despite reporting
  no Julia allocations. Consumer code needs no change.
- Fix Forward over Reverse of a function with a large tape on Julia 1.10
  (dogfood libEnzyme build `048872e7`), such as a `filter!` or a few array
  operations under a function that returns several arrays. It segfaulted in
  libEnzyme on an `xor` of a length read from the tape, raised `broken gc
  calling conv fix`, segfaulted in Julia's GC lowering, or raised
  `EnzymeRuntimeActivityError`. Values read from a tape now keep their type,
  a tracked `sret` field is cast to its root slot's type, `returnRoots` stay
  beside their `sret` after the calling convention fixup, and integers and arrays
  of integers copied into a tape are their own shadows. An active `xor` of unknown
  type now raises `EnzymeNoDerivativeError` instead of crashing. This also fixes
  `EnzymeNoDerivativeError` for Forward over Reverse through `push!` in `unique` on
  Julia 1.12.
- Merge upstream through `dd86d5435`, including LLVM lifetime fixes, separate
  readonly and attribute-preserving custom-rule hooks, and the shadow of a
  type-unstable `setfield!` return. Keep dogfood's Enzyme_jll 0.0.301 pin and
  declared compiler build. Dogfood is 0.13.218 and requires its in-tree
  EnzymeCore 0.8.28, which also carries `set_atomic_add`.

- Fix `Illegal cached pointer` on the first Reverse gradient, on Julia 1.10, of a
  function that reads an array element in a loop when both the array and the
  index have to be kept for the reverse pass (dogfood libEnzyme build `fce75191`).
  The choice of what to keep could land on the array's data pointer, which cannot
  be kept. It now keeps the array and reloads the pointer. Build `1c92729e` made
  this reachable in compiled model kernels.
- Do not run LLVM's Attributor on code that has already been differentiated
  (Julia 1.10 and 1.11; later versions never run it). Seeding it walks the code
  that must execute around every pointer and every value passed to a call, so
  its time grows with the square of a straight-line region, and a batch of `W`
  tangents makes the differentiated region `W` times longer. The first call of
  a batched Forward over Reverse derivative therefore compiled in time quadratic
  in `W`. Differentiation has used what the Attributor inferred by then, and a
  nested differentiation runs it again on the code it imports.
  `Enzyme.Compiler.RunAttributorAfterAD[] = true` restores the previous behavior.
- Compute Forward-mode derivatives of a division as `(dx - (dy / y) * x) / y`
  (dogfood libEnzyme build `2b1f1cfe`). The previous rule divided by `y * y`,
  which underflows for divisors below about 1e-154 and overflows above about
  1e154, so nested derivatives through `log` of a tiny value, or through any
  division by one, came out `Inf` or `NaN` although they are representable.
- Reduce redundant reverse caches for fresh array buffers on Julia 1.10.
  Merged empty-Memory paths on later Julia versions remain conservative.
- Keep that reduction through a ReactiveKernels/BRM model kernel, which
  applies `vec` to the call's result and passes its inputs as fields of an
  aggregate argument, and after a broadcast that allocates its output on each
  branch. Writes into objects allocated after the call, writes through a
  header merged from such allocations, and calls that write only memory they
  create no longer mark the call's inputs as overwritten.
- Keep it through later calls that write only arrays created after the call:
  a function that writes the elements of an array it allocates (a broadcast
  or a readout into `similar` that Julia does not inline), a function that
  returns such an array, and a function that writes an argument which is
  such an array, as `sum(A; dims)` does through `reducedim_initarray` and
  `_mapreducedim!`. Splats of tuples or arrays into `tuple` through
  `Core._apply_iterate` count as writing only their result. Through a
  ReactiveKernels/BRM model kernel this cuts the reverse storage of a
  multi-channel field consumed that way from 113-1,003 to 7-29 bytes per
  row-feature on Julia 1.10.
- Keep it across a whole model kernel: a call's inputs no longer count as
  overwritten by later stores into the kernel's own stack slots, by later
  calls that fill only arrays they allocate through calls that write the
  arrays they are given (a matrix product through `gemm_wrapper!` and BLAS,
  `collect` of arrays it computes, `push!` onto a vector it allocates), by
  mutually recursive calls that write nothing existing, or by
  `julia.write_barrier`. A kernel evaluating two or three such fields, or
  grouping one field's result with `collect`, keeps about 5-8 instead of
  48-88 bytes per row-feature on Julia 1.10.
- Default dogfood derivative arithmetic to strict floating-point semantics.
  This prevents reassociation from underflowing representable gradients such
  as `exp(log(t * 1e-190))`. Explicit `Enzyme.API.fast_math!(true)` remains
  available before derivative compilation.
- Compile and log from generated functions without calling a logger that has
  no methods in the running world, such as Test's `TestLogger` when Test loads
  after Enzyme. On Julia 1.10 such a task logger made nested derivatives fail
  for the rest of the process, and on Julia 1.10, 1.12 and 1.13 such a global
  logger broke derivatives. Messages Enzyme logs while compiling are now written to
  `Core.stderr` by `SimpleLogger`, without color.
- Differentiate statically through the tape of an inner derivative that holds
  constant data. Forward over Reverse and Reverse over Forward of a function
  that runs `sort`, `unique` or `unique!` on `Const` data raised
  `EnzymeRuntimeActivityError`. A member of an active call's result now takes
  the shadow the callee's derivative gave it, a tape member no reverse pass
  reads needs none, and a merge of `true` and `false` is known to be a `Bool`.

## 0.13.217

- Add `WithGradient(primal, value_and_gradient)` for explicitly supplied
  derivatives of scalar-valued functions, including opaque native calls.
  Supports Forward and Reverse modes, batching, and reusable gradient buffers.
- Repair constant native function-pointer activity analysis and diagnose
  missing derivatives before handling indirect-call shadows. Avoid an
  out-of-bounds parameter-attribute query when optimizing an indirect call.
  Preserve inactive local buffers and symbol lookup, and report missing reverse
  derivatives outside the augmented primal's exception handlers.
