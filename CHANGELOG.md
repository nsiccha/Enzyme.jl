# Changelog

## Unreleased

- Differentiate Forward over Reverse (a Hessian-vector product) through matrix
  products that reach BLAS `gemm`, `syrk` or `symm` (dogfood libEnzyme build
  `3a825a54`). Their reverse rules scale the output's shadow with LAPACK's
  `lascl`, which had no derivative in any mode, so the outer Forward pass
  raised `No forward mode derivative found for dlascl_64_`, or, for
  `A * transpose(B)`, "The original primal code hits this error condition":
  libEnzyme reported a missing forward derivative only at its first call and
  left later calls a stub ending in `unreachable`. `lascl` now has Forward and
  Reverse rules for its matrix (an active scale factor is refused), every call
  of a function without a forward derivative raises the error, and the Forward
  rules of `gemm`, `symm` and `syrk` raise their refusal of an active `beta`
  instead of crashing the compiler. A constant operand still fails in Forward
  over Reverse (todo `0mcxlsc`).
- Recompute a value of an outer loop iteration once per entry of an inner
  loop's reverse, not again in each of its arms on every inner iteration
  (dogfood libEnzyme build `7f23cb72`). Julia arrays make that value a chain of
  dependent loads (`idx.data`, `idx[j]`, `idx[j].data`), which the same loop in
  a function of its own gets as arguments. This was the remaining 6 to 7% of
  Reverse of a ragged nest over that outlined spelling on Julia 1.12 and 1.13.
- Do not recompute a call on a path the forward pass skipped (dogfood
  libEnzyme build `045d93e1`). When the reverse pass recomputes a value that
  three paths merge, one of which skips a call while the other two branch on
  its result, it evaluated that branch condition on all three paths, from
  values the skipping path never computed. A recomputed `sqrt` then threw a
  `DomainError` the primal never raised, for example in a loop over
  `x <= 0 ? 1.0 : (r = sqrt(x); r >= 2 ? exp(r) : r)`. The condition is now
  computed only on the two paths that ran it, unless it is safe to speculate.
- Avoid saving reader inputs for every term when results are written into a
  fresh output array that the reader cannot access (dogfood libEnzyme build
  `08f0e8d9`). The cache analysis now proves that private output storage stays
  inaccessible to the reader, including across loop iterations. Passing or
  publishing that storage to the reader still retains the required caches.
- Fix a segfault on Julia 1.13 (LLVM 20) while compiling a Reverse derivative
  (dogfood libEnzyme build `2cc9b459`). Deciding whether a split reverse pass
  replays a loop only once asked scalar evolution for the loop's trip count
  after Enzyme had inserted placeholder phis without incoming values, and
  LLVM 20's loop guards read the first incoming block of such a phi. The
  decision now takes the loop limit Enzyme already computed for that loop.
- Decide which arm of a branch a loop iteration took from the recomputed
  branch conditions, not from a cache (dogfood libEnzyme build `f57e12db`).
  When four or more paths met in one block inside a loop, as in
  `LogExpFunctions.log1pexp`, the reverse pass taped the path taken at every
  iteration. In a nested loop whose inner trip count varies, each outer
  iteration allocated its own buffer for that tape, which made Reverse of
  such a nest about 1.35 times slower than the same loop with its inner loop
  outlined. Conditions that cannot be recomputed safely keep the cache.
- Differentiate a read that picks each element from data the function made
  itself or from active data, `iszero(s) ? c[i] : active[s]` with `c` computed
  from a `Const` argument, under static activity. LLVM folds the two loads into
  one load through a merge of the two pointers, usually behind the bounds check
  of the destination, and reaching it raised `EnzymeRuntimeActivityError` on the
  path that takes `c`. Such loads are now made on each side, whichever side turns
  out to hold derivatives, by cloning the blocks up to the load where they sit
  behind a branch, and no copy of `c` is made. This also removes the copy of a
  `Const` array that remained on Julia 1.12 under `--check-bounds=yes`. An array
  a call returns as new memory, such as a `copy`, now gets a zero shadow where
  the merged array itself reaches a call, as an array the function allocates did.
- Differentiate Forward over Reverse through `append!` or `copyto!` from a
  `SubArray` on Julia 1.10. The augmented primal of `copyto!` reads a field of
  `unalias(dest, src)` through a pointer phi of the `src` argument and a copy of
  it; a nested derivative could not keep that phi rooted and threw "Enzyme could
  not determine how to keep a pointer phi in this function rooted" when it was
  reached. Such loads are now made in the phi's predecessors.
- Differentiate again, in a later session, what a package differentiated while
  its image was written: in split mode, in forward-mode `gradient` and
  `jacobian`, and in nested modes. Split-mode thunks (`autodiff_thunk`,
  reverse-mode `jacobian`, and the pullbacks of DifferentiationInterface) and
  the inner derivative of a nested one were compiled while their caller was
  inferred and kept in its code by their addresses, and the one-hot seeds of
  forward mode were built by code that named a type and a runtime function by
  their addresses; an image written with any of them segfaulted in every
  session that loaded it. While an image is written, split-mode code now
  fetches its thunks at run time, the seeds are built in plain Julia, and
  code that still holds an address is valid only in the world it was inferred
  in, which keeps it out of the image.
- Support literal constant objects captured beside active data, including a
  dictionary inside a callback that reads active views. Static differentiation
  now gives these objects a separate zero-filled shadow when one is needed,
  instead of raising `EnzymeRuntimeActivityError`.
- Differentiate a broadcast that copies an operand which may alias its
  destination, on Julia 1.13. LLVM 20 merged the data pointers of the copy's
  `Memory` and of the original `Array` into one load through a phi of their
  field addresses, which Enzyme turned into a phi of the two objects and then
  rejected with `IllegalTypeAnalysisException`. The load now stays on each
  edge. This covers broadcasting a scalar/vector union inside a loop.
- Stop copying the numbers a call read when the caller then stores only
  pointers (dogfood libEnzyme build `ece5ac54`). Any store after a call that
  Julia does not inline made Enzyme assume it may overwrite what the call
  read, so the call's reverse pass kept a copy. Storing a buffer back into
  the `Ref` it came from, as in `slot[] = mul!(slot[], X, b)`, copied all of
  `X` on every gradient, about half again the time of the same product
  without the store. A store of a pointer cannot overwrite numbers, so such
  a call now rereads them. A later store of a number still keeps the copy.
- Cache a value that does not change across the iterations of a loop once
  per entry of that loop, not once per iteration (dogfood libEnzyme build
  `f392eba5`). A load in a loop, from memory no instruction of the loop
  writes, through an address the loop does not change, was taped at every
  iteration when the reverse pass could not reload it. A `sum` over a
  generator inside a loop did so for each captured array and its shadow, at
  every term of the sum: about 150 bytes per term, now about 2.
- Preserve input shadows when a function returns newly initialized storage
  (dogfood libEnzyme build `33709d12`). Reverse differentiation through a
  non-inlined array gather could segfault because the compiler treated those
  writes as invisible outside the call and discarded the input's shadow.
  Returning a pointer loaded from a new object also retains the writes.
- Fix a deadlock when derivatives first compile on several threads at once. A
  derivative compiled at its first call held Enzyme's compile lock while it
  emitted code, which needs Julia's codegen lock (the type-inference lock on
  Julia 1.11 to 1.13). A split-mode or nested derivative compiles while Julia
  infers its caller, holding that lock, and waited for Enzyme's. It now
  compiles beside the holder instead, and what compiles share in globals is
  guarded by locks held without blocking.
- Reuse the zero shadow of constant data stored into a write-only output. When
  combined Reverse mode stores a constant array, such as the parent of a view
  that mixes it with active data, into a `Duplicated` output, a later call now
  reuses the zero copy an earlier call put there, if Enzyme allocated it and it
  is still zero, instead of allocating a new copy as long as the array. A
  reference kept to an earlier output shadow may therefore share that buffer
  with a later call's shadow (FAQ, "Reused zero shadows of constant data").
- Differentiate conditional integer/float denominators and boxed scalar/vector
  unions in ordinary Reverse mode. Uses with a proven branch condition now
  select the appropriate input before casts and loads, avoiding conflicting
  integer, floating-point and pointer types during analysis. The same proofs
  remove unreachable boxing branches before their layouts enter a pointer PHI.
  This also handles PHIs formed by branches with side effects, including
  scalar/vector broadcast preprocessing followed by a generated shape fold.
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
