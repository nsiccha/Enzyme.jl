# Changelog

## Unreleased

- On Julia 1.10, Forward and Reverse derivatives of `copyto!(dest, Hermitian(B))`
  for a `Matrix{ComplexF32}` work. They stopped with `cannot handle unknown
  binary operator: shl` (Forward) or `Cannot deduce adding type (cast)`
  (Reverse). Julia 1.10 copies each `ComplexF32` as one `i64`, unpacking the
  upper float with a shift and packing the halves again with `zext`, `shl` and
  `or`. libEnzyme's type analysis typed the shifted half as unknown, so the
  rules for packed floats did not apply. A shift of an integer holding only
  floats by a whole number of floats, and a disjoint `or` of two such
  integers, now keep the float type. The repair is in libEnzyme (build
  `e5c08438`) and is available as
  `Enzyme.has_dogfood_repair(:packed_float_lanes)`.
- On Julia 1.12, the second derivative of a call such as
  `ccall((:cos, library()), Float64, (Float64,), x)` works on the first call
  when `library()` returns a constant library name. The first and second
  derivatives share the call site's resolved address whichever runs first, so
  taking the second derivative does not evaluate the library expression again.
- Merge upstream support for Julia 1.14 rule lookup and JIT modules, and its
  fixes for deferred derivatives and cached thunk imports. The in-tree
  EnzymeCore is now 0.8.29. Enzyme requires LLVM.jl 10.5.2 or newer on the
  10.x line, and GPUCompiler 2.23 or newer on the 2.x line.
- Reverse gradients make a few native allocations per call where they made one
  `malloc` and one `free` per cached value. The non-inlined calls a combined
  Reverse derivative makes inside loops reserve their loop caches of numbers
  and Bools from one tape arena that the derivative owns and releases when it
  returns, and so do the derivative's own bounded caches of numbers. A
  superposition sum over each subject's events, called once per observation,
  made 1203 mallocs per gradient at 600 observations and makes a handful. The
  arena holds at most one partly used 64 KiB chunk beyond what its caches store
  at scale, and up to twice what a small tape stores. A derivative that another
  derivative differentiates again (a nested derivative, an imported thunk, a
  derivative compiled while another compilation runs) keeps one allocation per
  cache. The native repair is available as
  `Enzyme.has_dogfood_repair(:tape_arena)`; libEnzyme leaves the arena off and
  Enzyme turns on `-enzyme-caller-owned-tapes` and
  `-enzyme-arena-combined-caches` when it loads.
- `Enzyme.derivative_report` names where a call comes from when the compiled
  function holding it is called from several places. The `frames` of a
  finding go up through the call of its function while that function has
  exactly one call site, and used to stop there without naming any. A finding
  now also has `call_sites`: the call sites of the function where its frames
  stop, each an `Enzyme.DerivativeCallSite` with its own frames and, where
  those stop again, its own `call_sites`. A call inside a recursion lists the
  calls that enter the recursion. Only calls the derivative can reach count, a
  call without debug information is a call site named by its compiled
  function, and an instruction without a source line gets the call sites of
  its function. Call sites are listed once per function and shared, so a
  report stays linear in the size of the module. `Enzyme.call_chains(finding)`
  lists one complete chain of source lines per call path.
  `Enzyme.has_dogfood_repair(:derivative_report_call_sites)` reports this
  repair.
- Reverse can differentiate a loop that chooses among four array views and
  bounds-checks its reads. When the pass reconstructed which arm ran, it
  could ask for a bounds-check result it had already discarded and throw
  `EnzymeInternalError`. Reaching the merge establishes that the check passed.
  The native repair is available as
  `Enzyme.has_dogfood_repair(:returning_branch_predicates)`.
- Call the helper that reuses a constant array's zero shadow in a write-only
  output, and the zero-shadow helper of a constant mutable object such as a
  `Ref` stored beside active data, as compiled functions. Reverse called both
  through `jl_apply_generic`, so `Enzyme.derivative_report` showed a dynamic
  dispatch on the returning path of such a store. The registry of zero copies
  the reuse keeps is now guarded by a spin lock. The repair is available as
  `Enzyme.has_dogfood_repair(:static_zero_shadow_calls)`.
- On Julia 1.10, keep `EnzymeRules.overwritten(config)` `false` for an
  argument that holds references when the only writes after the rule's call
  store pointers, and nothing the argument leads to outside its own storage is
  mutable memory that holds a pointer. Storing the rule's returned vector into a
  `Ref` the caller passes, as a prepared pullback keeps its outputs, set the flag,
  and a rule copied its argument's data on every call. A store of a pointer
  cannot change numeric data, the rule as libEnzyme already applies to the
  memory a differentiated call reads. From Julia 1.11 an array refers to its
  memory through a field that Julia code replaces, so there an argument that
  holds an array keeps the flag `true`. The repair is available as
  `Enzyme.has_dogfood_repair(:later_pointer_stores)`.
- On Julia 1.10, keep `EnzymeRules.overwritten(config)` `false` for an
  argument that holds references when a loop calls the rule and stores each
  result into an array the function allocated before the loop, as a plate of
  cells does. The write barrier after each store counted as exposing that array
  to the next call, so the flag was `true` and a rule copied its argument's
  data on every call. On Julia 1.11 and later the flag stays `true` in this
  case.
  The native repair is available as
  `Enzyme.has_dogfood_repair(:unexposed_storage_barriers)`.
- Keep those Reverse gradients finite also when the hoisted arithmetic feeds a
  comparison, such as an interval width `b - a` that a guard tests before it
  uses `a`. The native repair is available as
  `Enzyme.has_dogfood_repair(:compared_operand_adjoints)`.
- Keep Reverse gradients finite when the compiler hoists arithmetic on an
  infinite endpoint ahead of the branch that skips it. The native repair is
  available as `Enzyme.has_dogfood_repair(:conditional_use_adjoints)`.
- On Julia 1.12 and later, reuse the inferred specialization when widening a
  call's signature would require runtime dispatch. This lets Reverse compile
  a triangular matrix times a vector view directly.

- Make `EnzymeRules.overwritten(config)` cover data behind references that a
  custom rule's argument holds. For an immutable struct, tuple, named tuple,
  view or other wrapper around an array, and for a callable that holds one,
  the flag stayed `false` after a later write to that array. A rule that
  reread the array in its reverse pass then returned a wrong derivative
  without an error, as Enzyme's own `det` rule did for a view. The flag is now
  `true` whenever a write the rule cannot see may follow its call, also in
  split mode, and stays `false` otherwise. The repair is available when
  `Enzyme.has_dogfood_repair(:overwritten_reachable_data)` is true.

- Differentiate a broadcast writer in Reverse when a `GC.@preserve` overlap
  check selects between writing directly and copying a temporary result.

- Avoid an array-sized shadow in Forward and Reverse when an in-place broadcast
  reads a disjoint `Const` array, as in `a .+= c .* b`.

- Pack reverse caches of Booleans, fixed-size vectors and pointer-free numeric
  records across ragged loops. A three-level nest now grows one shared buffer
  per level, avoiding one allocation per inner invocation. The repair is
  available when `Enzyme.has_dogfood_repair(:ragged_cache_levels)` is true.

- Preserve the scalar result of a fused broadcast over zero-dimensional arrays
  during differentiation.
- Restore Reverse differentiation when a write-only output can reuse the
  zero shadow of a constant array. Compiling this case raised a `MethodError`.
- Preserve the activity of constant data copied into a fresh trace or passed
  through dynamic dispatch. This fixes plain Reverse through Gen's dynamic
  `update` and through static updates reached via a type-unstable caller.
- Remove native tape buffers that the optimized reverse sweep never reads.
  A guarded reduction on Julia 1.13 could allocate two such buffers, each
  as long as the input, when it took its fallback path.
- Accumulate gradients from ordinary indexing of boxed arrays and memory
  back into their elements on Julia 1.11 and later. Repeated reads of an
  immutable record now add to its existing shadow, including in batched Reverse.
- Differentiate conditionally updated records containing both array references
  and floating-point fields in ordinary Reverse, including across loop iterations.
- Preserve concrete inline field types when copying a persistent vector of
  mixed integer and floating-point records. The dogfood libEnzyme build also
  keeps the types of distinct boxed tree nodes separate across dispatch,
  branches, and heterogeneous storage. Gen's static `Map` uses both layouts.
- Differentiate array reads whose byte offset combines an index with the
  distance between an interior data pointer and its backing Memory. LLVM can
  reorder these terms or split them across two pointer calculations. Reverse
  and Reverse over Forward now recognize the displacement in both forms.

- Resolve the zero-shadow call at compile time when a branch merges a fresh
  constant array with an active array and the compiler knows the array type.
  For example, Reverse can now differentiate
  `sum(abs2, size(x, 2) == 0 ? ones(size(x, 1), 1) : x)` without dispatching
  to `make_zero` at run time.
- Keep strong zero (`Enzyme.set_strong_zero`) in every derivative rule of a
  math function. A zero derivative reaching `log` at 0, `exp` where it
  overflows, or any other function whose partial derivative is infinite there
  gave NaN (`0 / 0`, `0 * Inf`) even with strong zero on, because only the
  rules of `*`, `/`, `^` and `rem` checked for it. A mixture with a constant
  zero-mass component, whose `log(0)` sits in one vector with the active
  component, therefore had NaN Forward derivatives under strong zero; it now
  matches Reverse. Without strong zero Forward still gives NaN there, as IEEE
  arithmetic does. The same check now also guards a call that reaches the
  result only through a select arm in Reverse, as repair `select_strong_zero`
  does for `/` and `*`: `ifelse(x > t, log(x), 0.0)` at `x = 0` has derivative
  0, not NaN. Needs libEnzyme build `adc222b3`;
  `Enzyme.has_dogfood_repair(:strong_zero_rules)` reports the repair and
  `test/strong_zero_rules.jl` checks it.
- Keep nested result-type queries concrete when a higher-order helper is
  reused for a different callable. For example, a loop that seeds a sum with
  `zero(Base.promote_op(f, ...))` can now differentiate the addition
  statically when the cell itself queries another callable's result type.

- Differentiate the derivative of a function that runs part of its work in a
  task (`Threads.@spawn` ... `wait`). Forward over Reverse of such a function,
  a Hessian-vector product, raised `TaskFailedException` around
  `EnzymeNoDerivativeError` ("No derivative available for indirect call
  through an inactive function pointer") when it ran: the derivative of a task
  runs the derivative of the task's body through a closure that reaches it by
  its address, which an outer derivative cannot differentiate. As for a
  `Threads.@threads` loop, the task now runs a callable, and an outer
  derivative differentiates a variant of it that asks for the body's
  derivative by its types. Forward over Reverse, Reverse over Forward,
  Forward over Forward and Reverse over Reverse return the right values.
  Several directions at once (`BatchDuplicated`) through a task are not
  supported, nested or not. `Enzyme.has_dogfood_repair(:nested_threaded_tasks)`
  reports the repair; `test/nested_threads.jl` covers it.
- Compile Reverse over Reverse through a `Threads.@threads` loop on Julia 1.12
  and 1.13, which stopped with `EnzymeInternalError` ("Illegal address space
  propagation"). Before differentiating, libEnzyme moves to the heap a stack
  slot whose value the reverse pass may need and marks the allocation, so that
  it can move the slot back to the stack where the reverse pass turns out not
  to need it. The code Enzyme keeps of a derivative for an outer one to
  differentiate still carried that mark, although in an augmented forward pass
  the allocation can escape through the tape; the outer derivative then tried
  to move it back to the stack, which it cannot do while the loop's array of
  tapes holds its address. The kept code now carries no such marks.
  `Enzyme.has_dogfood_repair(:kept_stack_slot_markers)` reports the repair;
  `test/nested_threads.jl` covers it.
- Fix two defects in the reverse rule for `sizehint!` on a `Float32` or
  `Float64` vector, on Julia 1.11 and later. The rule repeats the call on the
  derivative vector, which allocates the new storage uninitialized, but zeroed
  only the vector's current length. Elements that `push!`, `pushfirst!`,
  `append!` or `resize!` later grew into the reserved storage started with
  whatever the allocator had left there. So a Reverse gradient or a nested
  derivative through, for example, `sizehint!(Float64[], n)` followed by `push!`
  returned `NaN` or huge values in some calls and correct values in others,
  more often after heavy allocation. The rule now zeroes all of the new storage.
  Separately, the rule moved the adjoints of the elements the vector held at
  the call back through the derivative vector's header as the reverse pass
  found it. After `pushfirst!` into storage reserved with `first = true`, or a
  `push!` past the reserved storage that reallocated, that header pointed
  elsewhere and those adjoints were lost: the gradient of the retained elements
  came back zero. The rule now reads them where the call put them.
  `Enzyme.has_dogfood_repair(:sizehint_reserved_shadow)` reports both repairs.
  `test/array_storage.jl` grows into reserved storage after leaving freed
  storage nonzero, and `test/nested_unalias_view.jl` covers the original case.
- Stop Forward over Reverse from copying a `Const` array that a struct holds
  beside active data on Julia 1.12 and later. Since the inner reverse pass
  caches the byte offsets of a `MemoryRef` (the `filter!` and `sizehint!`
  repair), the outer derivative stores integers computed from the `Const`
  array's data pointer into its caches and tape, and the constant-field proof
  took such a store as a possible pointer. It then made a zero-filled copy of
  the whole array per call, 800 KB for 100,000 `Float64` in
  `test/const_element_store.jl`. A difference of two addresses holds no
  address, so the proof now accepts it. The same caching repaired Forward over
  Reverse of that generator `sum` with runtime activity, whose
  `EnzymeNoDerivativeError` ("cannot handle unknown binary operator: sub i64")
  the test no longer marks broken.
  `Enzyme.has_dogfood_repair(:constant_field_cached_offsets)` reports the
  repair.
- Keep the type facts of a union return buffer independent of what compiled
  earlier in the process. A function returning, for example,
  `Union{Tuple{Nothing, T}, Tuple{T, T}}` with an inline floating-point struct
  `T` returns through a buffer whose type Enzyme read back from the `Ptr` type
  of that buffer. Julia can reuse an equivalent `Ptr` type cached earlier, whose
  element type spells the union as the abstract `Tuple{Union{Nothing, T}, T}`,
  and Enzyme then annotated the buffer's `Float64` fields as pointers. Plain
  Reverse raised `IllegalTypeAnalysisException` only after other code had
  compiled: DynamicPPL's `gp_pois_regr` catalogue model failed after
  `diamonds` in one process and passed alone. The buffer is now annotated from
  the inferred return type itself. Strict aliasing and static activity stay on.
  `Enzyme.has_dogfood_repair(:union_return_types)` reports the repair;
  `test/union_return_types.jl` caches the equivalent pointer type first and
  checks Reverse gradients on both return branches.
- `Enzyme.derivative_report` omits unreachable calls and marks calls inside
  recursive error-message helpers as `throw_only` when every path into the
  helper ends in an error. A helper also called on a returning path still
  reports its dispatch and allocations there.
  `Enzyme.has_dogfood_repair(:derivative_report_reachability)` reports this repair.
- Allocate the reverse caches of a ragged loop nest together. In a combined
  Reverse derivative (`autodiff(Reverse, ...)`, `gradient`, and
  DifferentiationInterface's `gradient` with
  `AutoEnzyme(; mode = Enzyme.Reverse)`), the packed caches of repair
  `packed_ragged_cache` that share an element type and a loop nest now share
  one offset table and one growing buffer of records, one slot per cached
  value. Before, every cached value had a table and a buffer of its own, so a
  gradient made a native `malloc` and `free` per cached value and per capacity
  doubling of each. Now the nest makes one table and one growth per doubling,
  however many values it caches. An event walk that caches 13 values made 39,
  91 and 117 allocations per gradient at 2, 20 and 100 groups, and makes 6,
  14 and 18 (Julia 1.10.12). Split mode (`autodiff_thunk`) keeps a table and
  a buffer per cached value. Needs libEnzyme build `15345e27`;
  `Enzyme.has_dogfood_repair(:packed_cache_groups)` reports the repair and
  `test/packed_cache_groups.jl` checks it.
- Give a fresh `Memory` a header the garbage collector can read before the
  next safepoint. Julia 1.12 and later leave part of that header to the code
  that allocates: a `Memory` of a constant length that fits a pool cell
  (`zeros(64)`, `[a, b]`) is an object allocation followed by stores of its
  data pointer and its length, and any other `Memory` comes from
  `jl_alloc_genericmemory_unchecked` without its length. Enzyme allocates the
  shadow, which is a safepoint, next to such an allocation: Forward mode right
  after the original, Reverse mode right before it. A collection that started
  there marked a `Memory` whose header held what its cell held before, and
  followed what it found: a segmentation fault in `gc_try_claim_and_push`
  under `gc_mark_outrefs`, once in many thousand calls, since a cell that last
  held a `Memory` of the same size has a valid header left. Plain Forward mode
  on one thread was affected, and so were Forward over Reverse, where it was
  first seen through a `Threads.@threads` loop, and Reverse over Forward.
  - For the object allocation, Enzyme stores the header of an empty `Memory`
    with inline data, atomically, right after it: in Forward mode into the
    allocation a shadow allocation follows, in Reverse mode into the shadow.
    These are the two commits of upstream pull request #3723 for upstream
    issue #3698, which is open.
  - For `jl_alloc_genericmemory_unchecked`, Enzyme stores a length of zero
    right after the allocation wherever a call that can collect comes before
    the store of the real length, which is the case for the original in
    Forward mode. A `Memory` whose elements hold references, such as the
    `Vector` of tapes of a `Threads.@threads` loop's derivative, had that many
    elements scanned.
  `Enzyme.has_dogfood_repair(:memory_header_init)` reports the repair;
  `test/memory_header_init.jl` checks the stores in the derivative's code and
  takes derivatives while fresh `Memory` objects land on cells without a valid
  header. The same repair was found independently from repeated batched
  Forward through `[a, b]` of structs that hold arrays, which aborted on
  Julia 1.13.1 within seconds; `test/forward_memory_shadow.jl` repeats that
  under automatic collection, and `Enzyme.has_dogfood_repair(:forward_memory_shadow_init)`
  reports the repair under that name too.
- Differentiate a `ccall` whose library Julia does not resolve while compiling
  the method. Julia 1.12 leaves a module-qualified library
  (`ccall((:f, Mod.lib), ...)`, `@ccall Mod.lib.f(...)`) to a run-time
  `getproperty` call although `Mod.lib` is a constant; a non-constant global
  and, through Julia 1.12, any other expression are read the first time the
  call runs. Such a call reached Enzyme as an indirect call through a
  looked-up address: with `Ref` arguments, as in a hand-written BLAS or LAPACK
  wrapper, compiling the derivative aborted the process
  (`Assertion '!gutils->isConstantValue(baseOp)' failed`), and with scalar
  arguments it raised "No augmented forward pass found for
  ijl_lazy_load_and_lookup". A library read from a module global is now
  resolved when the derivative is compiled. Behind any other expression the
  call is named for its symbol, so a derivative rule for a math function
  applies. A native function without a rule raises `EnzymeNoDerivativeError`.
  `Enzyme.has_dogfood_repair(:ccall_runtime_library)` reports the repair;
  `test/ccall_library.jl` covers each spelling of the library for a libm
  function and for LAPACK's `lacpy` in Reverse, Forward and Forward over
  Reverse, calls that carry no derivative, and a function without a rule.
- Differentiate BLAS and LAPACK calls such as `ccall((:dlacpy_64_, lbt()), ...)`
  on Julia 1.10–1.12. The derivative evaluates the library expression the
  first time the call site runs and uses that library for the original routine
  and its rule's companion routines. Check
  `Enzyme.has_dogfood_repair(:ccall_computed_blas_library)`.
- Add `Enzyme.derivative_report(f)`, which runs `f()` and reports the
  derivatives Enzyme used for it: each one compiled, linked from a package
  image or asked for again meanwhile, with the calls its compiled code makes
  at run time. A finding names the kind of call (run-time dispatch, Enzyme's own
  run-time differentiation of a call it could not resolve, a box, an
  allocation on Julia's heap, a native allocation such as the tape, or any
  other call into the runtime or a library), the callee, the compiled function
  and sweep, the source lines, whether the call is in a loop or only on an
  error path, and an allocation's size when it is a constant. Until now the
  code a derivative runs could only be read from the text the `DumpPostOpt`
  debug flag prints to standard error, or by compiling it a second time with
  `enzyme_code_llvm`, which does not see a derivative compiled at run time and
  needs the job's settings repeated by hand. Enzyme now reads the optimized
  module of every derivative once, before its JIT takes it, and keeps what it
  found with the compiled code and with the machine code a package image
  keeps. A report therefore costs no second compile and covers code compiled
  earlier in the session. `ir = true` also returns the LLVM bitcode of what
  the block compiled. Where the host hands a compiled derivative to its
  caller (`autodiff`, `autodiff_thunk`, Enzyme's run-time rules), it reads one
  counter more while no report is running. The code of the call itself is
  unchanged, because kernels on a device share it; so a later call through
  `autodiff_deferred`, or of a split-mode thunk the caller kept, is not seen.
  `Enzyme.has_dogfood_repair(:derivative_report)` reports the API;
  `test/derivative_report.jl` covers the scan and the report, and
  `test/precompile.jl` a derivative linked from an image.
- Differentiate a `ccall` that names no library on Julia 1.13, such as the
  `memchr` call behind `findfirst` of a character in a `String`. Julia 1.13 no
  longer attributes a C library function to libjulia-internal while compiling,
  so the call reached Enzyme as a lookup in no library and was named
  `ejlptr$memchr$0`, which libEnzyme does not know: Reverse mode raised "No
  create nofree of empty function (ejlptr$memchr$0)", and a package image
  either failed to precompile or kept no derivative calling such a function.
  The call is now named by its symbol, which the JIT looks up in the process,
  as Julia does. `Enzyme.has_dogfood_repair(:ccall_without_library)` reports
  the repair; `test/ccall_library.jl` covers `findfirst`, `cmp` and
  `ccall(:cos, ...)`, and `test/precompile.jl` derivatives kept in an image.
- Fix a crash while compiling a static Reverse derivative whose code has a
  load through a select of pointers in a block that flows into a phi of
  pointers, where both merge constant data or data the function made with
  other data. Splitting the select rebuilt the phis after it, and the phi was
  then split from freed memory: the process segfaulted in `merge_load_tree`,
  or compiling raised `UndefRefError` from `merge_predecessors_branch`,
  depending on what reused that memory (a Gaussian-process posterior with a
  Cholesky-factored covariance hit it). Every select is now split before the
  phis are collected.
  `Enzyme.has_dogfood_repair(:constant_merge_order)` reports the repair;
  `test/constant_merge_order.jl` covers the select's block placed before and
  after the phi's block, and two selects flowing into two phis.
- Accept captured BLAS input buffers in the readonly check for an unannotated
  function argument. The proof uses per-argument readonly and nocapture
  attributes only for typed raw data pointers, and ignores uses solely in
  Julia GC-root bundles. Captured output buffers and aliases in writable
  positions still raise `EnzymeMutabilityException`.
  `Enzyme.has_dogfood_repair(:readonly_pointer_operands)` reports the repair.
- Differentiate the derivative of a `Threads.@threads` loop. Forward over
  Reverse of a function with such a loop, a Hessian-vector product, aborted the
  process while compiling: "Call parameter type does not match function
  signature!", then "LLVM ERROR: function failed verification". Two things
  stood in the way. The code Enzyme keeps of a derivative for an outer one to
  differentiate had lost the scheduling flag of its `threading_run` calls,
  which a dead-argument pass folded into the callee, so the outer rule read
  the loop body in its place. And the callable that runs the body's derivative
  on each thread reached it through an address, which an outer derivative
  cannot differentiate ("No derivative available for indirect call through an
  inactive function pointer"). Functions whose calls the thread rules
  differentiate now keep their parameters in that code, and an outer derivative
  differentiates a callable that asks for the body's derivative, as any nested
  derivative does. First-order derivatives call the body's derivative as before.
  While a package image is written, the body's derivative that an outer one
  compiles accumulates atomically, so what the image keeps is right in sessions
  with any number of threads.
  `Enzyme.has_dogfood_repair(:nested_threaded_loops)` reports the repair;
  `test/nested_threads.jl` covers Forward over Reverse (also with a batch of
  directions, runtime activity, the static schedule and an active scalar held
  by the loop body), Reverse over Forward, Forward over Forward and Reverse over
  Reverse against closed forms, with one and with two threads, and a package
  whose image keeps two of them, run with one, two and four threads. Reverse
  over Reverse works on Julia 1.10; on Julia 1.12 and 1.13 it still stops with
  an `EnzymeInternalError` ("Illegal address space propagation"), which the
  tests pin as broken.
- Keep the tangent of a tape that an outer derivative hands from one function to
  another. On Julia 1.11 and later every pointer in a tape had the type
  `Core.LLVMPtr{Cvoid}`, which Enzyme takes for constant data, so Forward over
  a split-mode Reverse returned a second derivative of zero with no error.
  Enzyme differentiates a dynamic call and the body of a `Threads.@threads`
  loop in split mode, so a Hessian-vector product lost everything that flowed
  through a dynamic call (a method with unused trailing arguments, a call
  through `Base.inferencebarrier`, type-unstable code); `autodiff_thunk` with
  the reverse pass in a function Julia does not inline showed it directly.
  On Julia 1.10 a tape named the table of a cache's chunks a pointer to
  pointers to elements, although its later slots can hold offsets, and an outer
  derivative stopped with `IllegalTypeAnalysisException` under
  `--check-bounds=yes`. Both are now pointers to `Enzyme.Compiler.TapeMemory`,
  memory of unknown layout that may carry derivatives.
  `Enzyme.has_dogfood_repair(:tape_memory_pointers)` reports the repair;
  `test/nested_threads.jl` covers Forward over a split-mode Reverse without
  threads, for a loop that reads its input and for one that overwrites it, and
  the Hessian-vector product through a dynamic call in
  `test/readonly_argument.jl` is no longer pinned as broken on Julia 1.12.
- Return zero gradients when a guard rejects a nonfinite product, including
  `isfinite(x) && 0 <= x <= 1 ? x * x : -Inf` at `NaN` and either infinity.
  The native select repair now protects products whose operands are nonfinite
  or overflowed, extending the existing guarded-division repair.
  `Enzyme.has_dogfood_repair(:select_nonfinite_products)` reports the repair;
  `test/select_strong_zero.jl` covers scalar, array, batched and split Reverse,
  Forward over Reverse, and finite inputs whose intermediate products overflow.
- Clear the adjoint of a BLAS output that the call overwrote. A routine that
  computes `out := ... + beta * out` does not read `out` when `beta == 0`, and
  a copy never reads its destination, so the adjoint of what `out` held before
  is exactly zero. The reverse rules of `gemv`, `symv`, `spmv`, `gemm`, `symm`,
  `syrk` and `lacpy` multiplied the output's shadow by `beta` instead: a `NaN`
  or an infinity in the adjoint of the result stayed there (`0 * NaN` is `NaN`),
  and a later reverse pass over the same shadow, as with a work buffer whose
  shadow is kept across calls, returned `NaN` gradients. The reverse rule of
  `copy` did not clear the destination's shadow at all, so two copies into one
  vector gave a wrong gradient within a single reverse pass, and a shadow used
  again kept growing. The forward rule of `copy` with a constant source now
  writes a zero tangent instead of `0 *` the tangent it found.
  `Enzyme.has_dogfood_repair(:blas_overwritten_output)` reports the repair;
  `test/blas_overwritten_output.jl` covers every routine with `NaN` and
  infinite output adjoints, shadows used again, `beta != 0`, the triangle
  `syrk` writes, strided and reversed copies, batches, split mode, runtime
  activity, and Forward over Reverse.
- Compile a function that a Reverse derivative reaches through dynamic dispatch
  (`Base.invokelatest`, a call Julia could not infer, a splat) with the
  caller's `set_atomic_add` setting. Such a callee used to get the
  thread-count default whatever the caller set, so `false` left atomic updates
  in it in a process with several threads, and `true` was ignored in a process
  with one. A callee reached from `false` that itself runs a threaded loop or
  tasks is compiled with the default, as before, and so are the functions it
  calls dynamically; the bodies of threaded loops and tasks keep the default.
  A derivative kept in a package image for such a callee has one variant when
  it runs no work in parallel and two otherwise.
  `Enzyme.has_dogfood_repair(:atomic_add_dynamic_callees)` reports the repair;
  `test/threads.jl` covers generic, `invokelatest` and splatted calls and
  calls of a method without a specialized entry point under each setting,
  nested callees, callees that run threaded loops or tasks, and a shared
  shadow under contention, and `test/precompile.jl` the variants an image
  keeps.
- Compile everything a derivative kept in a package image links in as a
  session with several threads compiles it, in the variant the image keeps
  for such sessions. An image is written by a process with one thread, and
  that variant held the body of a `Threads.@threads` loop, a derivative
  taken with `autodiff_deferred` inside the function and the inner
  derivative of a nested one as compiled for one thread, with plain updates
  of shadow memory. A session with several threads that linked it lost
  updates and returned wrong gradients without an error: of a function with
  a threaded loop under the default setting, under
  `set_atomic_add(mode, true)` and in split mode, and of Forward over
  Reverse called from several threads that share a shadow. A derivative
  whose own setting settles how it accumulates now has a variant for several
  threads as well when it links such a part in. The body of `Enzyme.pmap`
  and `Enzyme.@parallel` lost updates the same way. Its derivative is taken
  with the inline ABI, whose code goes wherever the rule's value goes, so
  the rules now hold a plain and an atomic derivative of the body and take
  the one for the number of threads of the process that runs the loop. Any
  other derivative with the inline ABI is compiled for several threads while
  a package image is written. The body of a threaded loop that an outer
  derivative differentiates again, as in Forward over Reverse, follows the
  same rule in place of accumulating atomically whenever an image is
  written, so a session with one thread runs the plain variant of such a
  kept derivative.
  `Enzyme.has_dogfood_repair(:retained_several_threads)` reports the repair;
  `test/precompile.jl` runs these shapes in sessions with one and with two
  threads, and `test/threads.jl` counts the atomic updates of each variant.
- Leave a `getelementptr` address that names no element of an internal
  constant global alone when loads of the global are replaced by parts of its
  initializer. An address before the first element (index `-1`) with such a
  global as its base stopped the compile of a derivative with
  `InexactError: trunc(UInt32, 18446744073709551615)`, and a smaller index past
  the end crashed the process. With opaque pointers (Julia 1.11 and later) an
  address that indexes the global through another type, such as a struct read
  as an array, replaced the load by a different element with no error.
  `Enzyme.has_dogfood_repair(:constant_global_gep_paths)` reports the repair;
  `test/constant_global_gep.jl` covers addresses before and past arrays,
  vectors and nested fields, addresses through another type, the element loads
  that are still replaced, and a table read behind a `-1` sentinel in Reverse,
  Forward, and Forward over Reverse.
- Differentiate real `logabsbeta` with scalar partials, removing the recursive
  reverse tapes reached through `logbeta` and `beta`. Large positive arguments
  retain the stable log-gamma-ratio derivative, and the sign and nonpositive
  integer branches keep their existing behavior. Regression tests cover
  Forward, Reverse, batching, split and nested modes, and loop allocations.
  `Enzyme.has_dogfood_repair(:logbeta_partials)` reports the repair.
- Retrieve cached custom-rule return values at their reverse use instead of
  reusing a forward load that may not dominate that use. A conditional
  `reshape` followed by a broadcast could otherwise abort LLVM verification
  with "Instruction does not dominate all uses!". The native repair preserves
  the separate caching of rule-return fields and is reported by
  `Enzyme.has_dogfood_repair(:conditional_rule_return)`;
  `test/conditional_rule_return.jl` covers early returns, empty arrays,
  combined and split Reverse, batched Reverse, and Forward over Reverse.
- Sync with EnzymeAD/Enzyme.jl through `1ee9a6f8`. This brings GPUArrays 12 support,
  Julia's newer write barriers and allocation intrinsic, and the LLVM.jl 10 API.
  The dogfood compiler passes, retained derivatives, reports, and regression tests
  use the new API. The minimum LLVM.jl version is 10.4.3. Enzyme_jll stays at
  0.0.301 with the native build declared in `deps/dogfood_libenzyme.toml`.
- Limit the runtime `map` wrapper to long-tuple calls. A custom collection's
  `map` with an imprecise return type could otherwise re-enter the wrapper
  indefinitely and overflow the stack during differentiation.
  `Enzyme.has_dogfood_repair(:runtime_map_collection)` reports the repair;
  `test/tuple_map.jl` checks direct and splatted calls in Reverse, Forward,
  and batched Forward, alongside the existing long-tuple checks.
- Infer the return type of `autodiff_deferred` with the compiler of the calling
  code. The previous host-side query saw only host methods: where device code
  reads an array, such as a CUDA device array or a view of one, it met a
  throwing host stub and inferred `Union{}`. A kernel differentiating such a
  function with an inferred `Active` return then failed to compile, with
  `InvalidIRError` naming a dynamic `deferred_id_codegen`. Reverse and Forward
  deferred calls now see the caller's method table. An explicit reverse return
  annotation such as `Active{Float64}`, which raised an error, is accepted too.
  `Enzyme.has_dogfood_repair(:deferred_return_inference)` reports the repair;
  `test/cuda.jl` differentiates column views of device matrices in Reverse and
  Forward for Float32 and Float64, and `test/core/deferred.jl` covers the
  explicit annotation.
- Retain package-image derivatives that call library functions whose selected
  implementation has no exported name, such as glibc's `memchr` reached through
  `findfirst`. Enzyme uses its known foreign-function names and verifies that
  each resolves to the exact address in the containing library. Fresh processes
  can then link these derivatives instead of recompiling them; unknown pointers
  still prevent retention. `test/precompile.jl` checks fresh-process reuse and
  numerical results in Forward and Reverse.
- Give constant differentiable fields separate zero shadows when runtime tuple
  and struct constructors run under static Reverse. Such fields can now be
  captured alongside active data without a runtime-activity error or writes
  into the constant input.
- Omit nonexistent element-layout facts from array allocations initialized with
  the empty Memory singleton on Julia 1.11 and later. Constructor maps whose
  empty and nonempty arrays
  meet at a return can then differentiate without conflicting pointer and
  floating-point type facts. Array and Memory header facts are preserved.
  `Enzyme.has_dogfood_repair(:empty_array_types)` reports the repair.
- Repair and preserve the `sret`/`returnRoots` calling convention for retained
  helpers under `InlineABI`. First-order Forward through a helper returning
  both an inactive integer array and scalar data could otherwise crash Julia's
  GC lowering. `Enzyme.has_dogfood_repair(:inline_return_roots)` reports the
  repair; `test/inline_return_roots.jl` checks scalar and batched Forward and
  Reverse across several aggregate layouts.
- Use Base's inherited `ndims`, `eltype` and array conversion for `TupleArray`,
  avoiding invalidation of existing abstract-array callers when Enzyme loads.
- Preserve distinct allocation instances when an array is carried from one loop
  iteration to the next. Reverse replay could substitute an uninitialized new
  array header for the previous iteration's header, causing incorrect gradients
  or memory corruption in nested scans on Julia 1.13.
  `Enzyme.has_dogfood_repair(:loop_carried_allocations)` reports the repair;
  `test/loop_carried_allocations.jl` covers changing scan lengths, empty inputs,
  repeated calls, garbage collection, and combined and split Reverse.
- Support caller-owned structured destinations in reverse-mode `gradient!`,
  including tuples and named tuples of arrays and `Ref`s. Repeated calls zero
  and refill the same storage, including through DifferentiationInterface
  without contexts. `Enzyme.has_dogfood_repair(:structured_gradient_inplace)`
  reports the repair.
- Check only file-backed allocated ELF sections for embedded process addresses
  when retaining derivatives. Relocation records and other metadata no longer
  cause layout-dependent refusals; embedded addresses in code and data are still
  checked. `Enzyme.has_dogfood_repair(:retained_object_sections)` reports the repair.
- Keep the Julia type of a pointer reloaded from a stack slot where libEnzyme
  drops its tag, on Julia 1.12. Enzyme.jl tags such reloads before
  differentiation, but libEnzyme differentiates a copy of the function in which
  it has rewritten the slot's loads without the tag, and a slot it moves to the
  heap is declared an array of `Any` there. The mixed-activity handler then
  could not see that a reloaded index vector of a `view`, a `Vector{Int}`,
  holds no derivative, and raised `EnzymeRuntimeActivityError` in two places:
  a custom rule that holds `try`/`catch` or `try`/`finally`, differentiated
  again in Forward mode as a Hessian-vector product does (Julia zero-initializes
  the roots slot of an argument that a handler may read, which keeps the slot
  on the stack); and split-mode Reverse, or a dynamic call, of a vararg method
  whose `view` merges an index vector reloaded from its argument tuple with the
  copy `unalias` makes. `abs_typeof` now reads the type of an untagged reload
  from the slot's writes, by the rule of the tagging pass and in every module
  that pass has typed. `Enzyme.has_dogfood_repair(:untagged_slot_reloads)`
  reports the repair; `test/rule_handler_slot_reloads.jl` checks gradients and
  products through rules with either handler, and the split-mode and
  dynamic-call checks of `test/const_slot_reloads.jl` and
  `test/const_element_store.jl` now run on Julia 1.12 as on 1.10 and 1.13.
- Reduce native Reverse tape allocations when a runtime row count varies by
  outer event but is shared by its inner component loop. libEnzyme groups
  those component caches into one allocation per event and recomputes identical,
  safely speculatable branch arithmetic. Values and bounds that later writes
  can change remain taped. The build also includes upstream's clamp for hoisted
  counts of skipped inner loops (libEnzyme `eea6a012`).
  `Enzyme.has_dogfood_repair(:ragged_loop_cache)` reports the repair;
  `test/broadcast_ragged_cache.jl` checks broadcast shapes, mutation and tape sites.
- Accept singleton subtypes of `AbstractFloat` with no runtime payload in
  type layouts. This fixes type-layout assertions when
  differentiating MeasureTheory `For` measures with captured active values.
  Nonempty unsupported float types still require an explicit type mapping.
  `Enzyme.has_dogfood_repair(:singleton_float_layouts)` reports the repair;
  `test/float_layouts.jl` and `test/integration/MeasureTheory/runtests.jl` cover it.
- Accept `memset` initialization of local argument and GC roots slots in the
  readonly check. This lets Soss differentiate `For` measures over captured
  vector data. Writes through captured pointers remain rejected.
  `Enzyme.has_dogfood_repair(:readonly_local_memset)` reports the repair;
  `test/readonly_memset.jl` and `test/readonly_argument.jl` cover it.
- Preserve constant-data provenance through `jl_reshape_array` and
  `ijl_reshape_array`. On Julia 1.10, storing `vec` of a constant matrix beside
  an active vector now receives independent zero derivative storage instead of
  raising `EnzymeRuntimeActivityError`; the primal keeps its original aliases.
  Enable the existing Array reshape rule on Julia 1.11 and 1.12, whose
  MemoryRef array headers need the same handling as Julia 1.13.
  `Enzyme.has_dogfood_repair(:const_array_reshape)` reports the repair;
  `test/const_reshape.jl` covers reverse, forward, batch, split and nested modes.
- Pack eligible ragged Reverse caches into geometrically growing value buffers
  with saved event offsets, reducing allocations as event counts grow. Values
  and bounds that later writes can change remain taped. Buffer growth supports
  split reverse and higher derivatives; extra capacity and copying can increase
  peak native memory. This applies to scalar numeric caches with two allocation
  levels and known chunk bounds on 64-bit integral host pointers using `malloc`.
  `Enzyme.has_dogfood_repair(:packed_ragged_cache)` reports the repair, supplied
  by libEnzyme `e316a076`. `test/broadcast_ragged_cache.jl` covers split reverse,
  empty chunks, large growth steps and nested differentiation.
- Make Reverse over Forward of a typed nested comprehension over ranges,
  `Float64[ts[i] - d[j] for j in eachindex(d) for i in eachindex(ts)]` in a
  function Julia does not inline, compile. The gradient pass of the Forward
  derivative's split reverse dropped the next outer counter inserted into the
  `Flatten` iterator's state while it still built the cached state tuple's
  shadow from it: `EnzymeInternalError`, "Illegal replace ficticious phi for …
  add i64 %value_phi21, 1". libEnzyme's `is_value_needed_in_reverse` had
  memoized the state's loop-exit phi as never needed, provisionally, inside the
  evaluation of the loop-header phi that is its only user, and kept that answer
  once the header phi turned out needed. Answers read inside a cycle now stay
  provisional and are forgotten when the key they relied on turns out needed
  (libEnzyme build `b20b9440b32133907334d6cc60c21d1880ac5eb6`). `Enzyme.has_dogfood_repair(:use_analysis_cycle_memo)`
  reports it; `test/reverse_over_forward_collect.jl` is the canary.
- Stop deferred activity analysis from repeatedly revisiting the same dependency
  when an old hypothesis refills it. This caused intermittent `StackOverflowError`
  while compiling derivatives on Julia 1.12 with forced bounds checks. A dependency
  runs at most once while the known inactivity facts are unchanged, and remains
  pending for another attempt when new facts are proved (libEnzyme build
  `1204679cc8b68fc458be9bc2359e5132e9c841e1`).
  `Enzyme.has_dogfood_repair(:activity_reevaluation)` reports the repair;
  `test/data_pointer_phi.jl` and `test/inlined_loop_nest.jl` cover the affected cases.
- Make Forward over Reverse compile when the inner reverse caches an element of
  a `Vector{Vector{Int}}` and one of a `Vector{Vector{Float64}}` in a loop it
  may skip. Enzyme.jl seeds every cache slot of a GC-tracked pointer type with
  `jl_nothing` cast to the slot's type, LLVM uniques that constant expression,
  and libEnzyme's type analysis let what it learned from one cache's phi reach
  the seed of every other: `IllegalTypeAnalysisException`, "Illegal
  updateAnalysis" on `bitcast (@ejl_jl_nothing …)` with a cache phi as origin.
  A cast of a global marked `enzyme_ta_norecur` is now as opaque to type
  analysis as the global (libEnzyme build `b20b9440b32133907334d6cc60c21d1880ac5eb6`).
  `Enzyme.has_dogfood_repair(:norecur_cast_type_seed)` reports it;
  `test/forward_over_reverse_cache_seed.jl` is the canary.
- Preserve explicit type layouts beyond libEnzyme's 500-byte inference offset
  limit. A wide tuple of broadcast operands dispatched at run time could silently
  lose derivatives of its last fields. The native repair carries a finite bound
  with each supplied layout; the process-wide inference limit remains 500.
  `test/type_offset_layouts.jl` checks large layouts and Reverse through wide tuple
  tails and broadcasts. `Enzyme.has_dogfood_repair(:explicit_type_layouts)` reports
  the repair, supplied by libEnzyme `ce4dc798`.
- Restore the shadow vector's size and storage when reversing `_deleteend!` and
  `sizehint!` on Julia 1.11 and later. Earlier reverse instructions then read the
  original storage, and contributions through aliases to that storage are added
  correctly. This repairs misplaced adjoints through `filter!` on floating-point
  vectors while keeping paired MemoryRef loads recomputable. Preserve integer
  addresses used in pointer-offset calculations, so Reverse over Forward also
  uses the original offsets under forced bounds checking without caching GC roots.
  `Enzyme.has_dogfood_repair(:array_storage_adjoints)` reports it.
- Differentiate Base's `map` over a tuple of 32 or more elements whose elements are
  of several types. Base's `map(f, t::Any32)` and its two- and many-tuple methods fill
  a `Vector{Any}` at a run-time index and splat it back into a tuple, so one value
  carries every element type, and Julia's union split stores it into boxes laid out
  for each type. Type analysis gives a value one layout, so differentiating such a
  `map` raised `IllegalTypeAnalysisException` (and its result was never typed).
  DifferentiationInterface's `Rewrap`, `make_context_shadows` and
  `translate_prepared!` reach these methods with more than 31 contexts, so a nested
  derivative (`hvp` through `SecondOrder` of two `AutoEnzyme` backends) of a function
  with 40 `Constant` contexts and one `Cache` failed in both compositions on Julia
  1.10.12, while 40 `Constant` contexts alone, or 8 plus the `Cache`, differentiated.
  Enzyme's interpreter now maps a concretely typed tuple that reaches one of those
  methods field by field (`override_map`), in Base's order and over the shortest
  tuple; a more specific user `map` method stays authoritative, and the first-order
  `map` outside the differentiated code is unchanged. `test/tuple_map.jl`;
  `Enzyme.has_dogfood_repair(:long_tuple_map)` reports it.
- Apply the long-tuple `map` repair to runtime-dispatched calls, including tuples
  hidden behind `Base.inferencebarrier` and splatted argument tuples. Runtime
  differentiation previously compiled Base's `map` directly, missing the
  interpreter's call-site rewrite and raising `IllegalTypeAnalysisException` on
  mixed tuples of 32 or more elements. A Julia wrapper exposes the concrete runtime
  types to that rewrite while preserving more specific methods and custom rules.
  `test/tuple_map.jl`; `Enzyme.has_dogfood_repair(:runtime_tuple_map)` reports it.
- Clear every restored shadow element when reversing `jl_array_del_end`, which
  shrinks vectors on Julia 1.10, and use each lane's shadow in batched reverse.
  Previously only one element was cleared, so deleting several elements could
  leave stale adjoints in the restored tail. This also affected derivatives
  through `filter!`, including nested derivatives through `sort!(filter!(...))`.
  `test/array_del_end.jl` checks nonzero shadow seeds, partial and complete
  deletion, batching and runtime activity;
  `Enzyme.has_dogfood_repair(:array_del_end_adjoints)` reports it.
- Keep wide tuple splats typed during differentiation. Raise the interpreter's
  inference and optimization splat limits from 32 to 512, following upstream
  #3994 and matching the runtime helpers' argument budget. On Julia 1.12, calls
  with 32 or 40 constant arguments beside an active `Ref` now differentiate in
  Forward and Reverse instead of reaching an unsupported C-ABI generic call.
  This also enables the long-tuple map tests and nested DifferentiationInterface
  derivatives with 40 `Constant` contexts and one `Cache`. Consumer code needs no
  change. `Enzyme.has_dogfood_repair(:wide_active_calls)` reports the repair.
- Tag a boxed `Bool`, `Char` or integer that Enzyme allocates itself the way Julia tags
  it. From Julia 1.10 such a type has a small type tag (`enum jl_small_typeof_tags`):
  an object of it carries `tag << 4` in its header word, not a pointer to its datatype,
  and the garbage collector aborts the process (`GC error (probable corruption)`,
  printing the type) when it marks a pointer-tagged object of a small-tag type.
  `emit_allocobj!` tagged every box with the datatype pointer. That box is the primal
  a derivative returns when its entry returns a small `Union` of isbits members, such
  as `Union{Missing, Bool}` or `Union{Nothing, Int}`, with `ForwardWithPrimal`,
  `ReverseWithPrimal`, split mode, or any call Enzyme dispatches at run time (whose
  thunk always returns its primal), and a custom rule's union tape. ReactiveKernels met it in ordinary Reverse of a plate cell
  whose 33-operand broadcast dispatched `_cell_array_operand`, returning
  `Union{Missing, Bool}`, at run time: the next collection aborted the process. The new
  `emit_typetag!` emits the small tag as Julia's codegen does (`emit_tagfrom`), and a
  boxed `Bool` is `jl_true` or `jl_false`, as Julia boxes it: Julia compares boxed
  `Bool`s by pointer, so a tagged box of its own was `===` to neither.
  `Enzyme.has_dogfood_repair(:small_typetag_boxes)` reports it.
- Make static Reverse of `sum` over a lazy broadcast that reads `Const` data
  beside active data, `sum(Broadcast.instantiate(Broadcast.broadcasted(f, ds,
  y)))` with `Const(y)`, allocate what runtime activity allocates. On Julia
  1.10 and 1.13 it made a zero-filled copy of `y` on every call, 8 bytes per
  element: at 5000 elements 280,544 bytes per gradient against 240,496 with
  runtime activity (Julia 1.10.12). The `Broadcasted` reaches Base's pairwise
  `mapreduce_impl`, which calls itself with the same struct, and the
  constant-field proof gave up any slot a recursive call passes back to its own
  function. `constant_field_places` now treats a slot passed on at the offset it
  was received at as adding no place; a cycle that moves the offset still gives
  the slot up. `Enzyme.has_dogfood_repair(:recursive_constant_fields)` reports
  it.
- Merge upstream through `0126567` (#4035): differentiate broadcast and `map`
  over arrays of structs with inline fields containing padding. Preserve
  dogfood's cycle-aware offset analysis and readonly argument checks. The
  package versions, Enzyme_jll 0.0.301 pin, and declared compiler build are
  unchanged.
- Merge upstream through `a154f47`, Enzyme 0.13.220: TypeTrees of arrays and
  memory with element type `Union{}`, Julia runtime singleton and exception
  globals marked inactive, mode-specific custom-rule attributes stripped from
  cached bitcode, `jl_ptr_to_array` forward with an active value,
  `MixedDuplicated` task closures, `hypot` derivatives of mixed-precision
  arguments in their argument and return types, multi-element type metadata for
  fixed-size copies, `eps`, `nextfloat` and `prevfloat` read-only through
  pointers, sparse shadows in `@import_rrule`, and `julia.get_pgcstack_or_new`
  recomputable. Keep dogfood's Enzyme_jll 0.0.301 pin, its declared compiler
  build and its in-tree EnzymeCore 0.8.28.
- Make `Enzyme.make_zero(x)` and `Enzyme.make_zero!(x)` of a point that is a
  tuple, named tuple or `Ref` of floating-point numbers, integers, and arrays or
  `Ref`s of floating-point numbers cost what building its shadow by hand costs.
  `Enzyme.gradient` calls `make_zero` on each such argument at every call. On
  Julia 1.10.12, `make_zero` of `([0.1, 0.2, 0.3], Ref(0.5), Ref(0.7))` took 66
  to 68 ns and 448 bytes, against 15 to 17 ns and 112 bytes by hand; it now takes
  15 to 16 ns and 112 bytes, and its `make_zero!` 2.2 ns instead of 83 to 86 ns
  and 400 bytes. With a vector and 11 `Ref`s, `make_zero` took 1.25 to 1.32 µs
  and now takes 34 to 36 ns. Two causes:
  - `make_zero` built an `IdDict` at every call, and `make_zero!` an `IdSet`, so
    that two references to one array get one shadow. Such a value holds no
    references among its parts, so a copy now compares its arrays and `Ref`s with
    `===` instead (up to 32 of them; `===` of values of two different types
    folds away), and zeroing it in place needs no set at all. Any other value
    keeps the table.
  - The `Tuple` method of `make_zero` read each field's type as
    `RT.parameters[i]`, which does not infer when Julia loops over a tuple of
    more than 10 fields, so each field dispatched dynamically. It now unrolls up
    to 32 fields with `fieldtype`. A tuple type with an abstract field, passed to
    the four-argument `make_zero`, now zeroes the tuple as its concrete type
    instead of raising an `AssertionError`.

  `Enzyme.has_dogfood_repair(:make_zero_tuple_points)` reports it.
- Make Reverse of a loop whose carried value is read from an array on every edge
  into the loop, such as a loop with its first step peeled (`yp = y[first]; for
  t in first+1:last; …; yp = y[t]; end`), cost close to what the same loop costs
  without the peel on Julia 1.10: inlined into an outer loop, 1.02 to 1.05 times
  it instead of 1.05 to 1.10, and in a function Julia does not inline, under
  strict derivative arithmetic, 1.04 instead of 1.13 to 1.16. LLVM folds such a
  carry into a load through a phi of element addresses. `nodecayed_phis!`
  rewrote that phi into a phi of array objects and offsets, so the reverse pass
  reloaded the array and its data pointer on every iteration.
  `unfold_field_phi_loads!` now loads in the predecessors instead, which leaves
  a phi of the values. `Enzyme.Compiler.UnfoldDataPhiLoads[] = false` restores
  the old rewrite. `Enzyme.has_dogfood_repair(:data_phi_loads)` reports it.
- Make Reverse of an inner loop that reads a value computed through a branch
  cost what the same loop costs in a function Julia does not inline (dogfood
  libEnzyme build `10018598`). In `w = l < 0.1 ? f(l) : g(l)` followed by an
  inner loop reading `w`, the reverse pass rebuilt `w` on every inner iteration:
  a branch on the same condition and the taken arm's computation, calls
  included. It is now rebuilt once per entry of the inner loop's reverse, when
  its arms and the conditions between them can be read there. An arm may read
  the result of a call that carries derivatives from its cache.
  `-enzyme-reverse-loop-entry-merges=0` restores the old placement.
  `Enzyme.has_dogfood_repair(:reverse_loop_entry_merges)` reports it.
- Let a package check, before it compiles a derivative, that this dogfood
  commit carries the repairs it needs. Every dogfood commit reports the same
  version number as a registered release, so a `[compat]` entry cannot tell
  dogfood commits apart, and a repair in libEnzyme only counts when the library
  that loaded is the build `deps/dogfood_libenzyme.toml` declares.
  `deps/dogfood_repairs/` now holds one file per repair the dogfood line
  carries, named after the repair. `Enzyme.dogfood_repairs()` lists the repairs
  this process carries, `Enzyme.has_dogfood_repair(name)` asks for one, and
  `Enzyme.require_dogfood_repairs(names...; requester)` throws
  `Enzyme.DogfoodRepairMissing`, naming each missing repair and why: this
  source predates it, or the loaded libEnzyme is not the declared build.
  `Enzyme.dogfood_libenzyme()` returns the declared build, the loaded library,
  and whether they match. Dogfood commits before this one lack these
  functions, so check `isdefined(Enzyme, :require_dogfood_repairs)` first.
- Give dogfood derivative arithmetic the fast-math flags `nsz` and `contract` by
  default (dogfood libEnzyme build `3cee1409`). Fully strict arithmetic, the default
  since build `df199913`, kept an adjoint's `+ 0.0` (strict IEEE arithmetic does not
  fold it, since `-0.0 + 0.0` is `+0.0`) and every separately rounded multiply and
  add. On Julia 1.10 that made the Reverse gradient of a plain Normal log-density loop
  2.1 times as slow as registered Enzyme, and a quadratic reduction 2.8 times as slow
  as fast math. Neither flag regroups an expression or assumes anything about NaN or
  infinity, so a representable derivative such as that of `exp(log(t * 1e-190))` still
  does not underflow. The new `Enzyme.API.fast_math_flags!` chooses the flags, and
  `fast_math_flags!()` restores fully strict arithmetic; `fast_math!(true)` still
  allows every flag.
- Make Reverse of an inner loop inlined into an outer loop cost what the same
  loop costs in a function Julia does not inline (dogfood libEnzyme build
  `576e674a`). An outer loop over paths gathers each path's slice, runs a
  recurrence over it into a fresh vector and reads that vector back, as
  ReactiveKernels lowers a scan inside a plate cell. Inlined, its gradient took
  up to twice as long on Julia 1.10, with one more trajectory-sized allocation
  per path. Three repairs remove the difference:
  - A later call that writes only memory it creates, such as the next path's
    gather, no longer makes an earlier load uncacheable, so the gathered
    elements and constant weights are reloaded rather than taped at every step.
  - Numbers read back from a fresh array no longer make it look exposed to a
    call, so storing into the output does not make the gather tape its indices.
  - On Julia 1.10, a load of the length of a vector made by
    `jl_alloc_array_1d(atype, n)` that nothing can resize reads `n`. The reverse
    pass then needs neither the vector nor a rebuilt copy of it (replaying the
    whole recurrence) to bound the loop that reads it.
- Give a guarded division its derivative where the guard is false, in Reverse
  mode (dogfood libEnzyme build `0a3566d8`). LLVM speculates guarded code into
  a select: `x > t ? 1 / x : 0.0` becomes `select(x > t, 1 / x, 0.0)`, with the
  division running on every path, and a repeated guard
  (`x > t ? 1 / (x > t ? x : 1.0) : 0.0`) or a guarded operand
  (`1 / (x > t ? x : Inf)`) compiles to the same select. Where the guard is
  false the reverse pass gives the division a zero adjoint, and its rule then
  computed `0 / 0` and `0 * Inf` at `x = 0`: the Reverse derivative was NaN.
  An unused output of the same function made the whole gradient NaN, as the
  posterior moments do that a Kalman update returns beside its log predictive
  density when its pseudo-inverse guards a zero eigenvalue. The reverse rules of
  a value that reaches the result only through select arms, and is a division,
  a remainder or a call or is computed from one, now treat a zero adjoint as
  exactly zero, as `set_strong_zero` does everywhere. Other rules are
  unchanged; libEnzyme's `-enzyme-select-strong-zero=0` restores the previous
  ones.
- Keep the elements of a fresh array that a function Julia does not inline
  fills, on Julia 1.10 (dogfood libEnzyme build `48fe7ef5`). Such a function
  receives the array's header, which it only reads: it loads the data pointer
  from the header and writes the elements through it, so LLVM marks the header
  operand `readonly`. libEnzyme took that to mean nothing wrote the array,
  recreated it uninitialized in the reverse pass, and the reverse of a later
  reader of the array read that memory. The gradient of everything upstream of
  the array came back as whatever the allocator handed out, beside a right
  primal, as for an RKPPL MvNormal slice prior whose `Matrix{Float64}(undef, k,
  G)` a non-inlined forward substitution fills. Such a call now counts as
  writing the array unless the callee's body shows it writes no storage the
  array owns, so the reverse pass keeps the filled array.
- Stop allocating a tape per call, on Julia 1.12 and 1.13, in Reverse of a
  recursive function Julia does not inline that reads a view or an array, such
  as Base's pairwise `sum` of a view of 16 or more elements: 48 to 64 bytes per
  node of the recursion on every warm call. The reverse pass now recomputes the
  root of a `MemoryRef` read (the array's `Memory`) together with its data
  pointer instead of taping it, and `julia.write_barrier` is declared with the
  attributes Julia's codegen gives it, so alias analysis no longer takes it for
  a write to any memory. Split-mode tapes over arrays lose that root, so
  `tape_type` on Julia 1.11+ now matches Julia 1.10. On Julia 1.12, Forward
  over Reverse of such a `sum` over 1,025 or more elements no longer raises
  `EnzymeNoDerivativeError`.
- Stop corrupting the JIT's symbol table, on Julia 1.10 and 1.12, when
  `autodiff_deferred` is called outside of a differentiated function. That call
  jumps through a lazy trampoline whose module was emitted against a
  materialization responsibility for the trampoline's target alone, while the
  module also defined other global symbols, such as the `libname_<function>_<n>`
  cache Julia 1.10 and 1.12 keep for a `ccall` whose library is a run-time value
  (SpecialFunctions' `erfcx` has one). Julia's object layer resolves such symbols without claiming
  them, and LLVM then read and wrote past the end of the dylib's symbol table:
  the first call segfaulted ("JIT session error: Failed to materialize
  symbols", then `unknown function (ip: (nil))`), or the process aborted later
  with "double free or corruption". The trampoline's module now exports its
  target only.
- Return the primal value from `ReverseWithPrimal` on every path (dogfood
  libEnzyme build `42a2ee9a`). Combined Reverse may run a call's forward pass
  inside its reverse pass, and it then moved the store of the return value
  after that call even when the returned value did not depend on it. A path
  that skipped the call (such as Base's `_copyto_impl!` behind a slice
  `u[2:n]` that is empty at run time) returned whatever the return slot held,
  beside a right gradient. The store now moves only with a returned value that
  depends on the call, and a call is not combined when what would move with it
  lies on a path that may skip it.
- Differentiate a `Const` array read back from a stack slot beside active data
  under static activity. A vararg method stores its `Const` arguments into a
  tuple, and a large function keeps that tuple on the stack, so an overlay of
  `Const` and active data is built from a reload of the tuple, not from the
  argument. Two things raised `EnzymeRuntimeActivityError` there:
  `view(overlay, rows)` reaches `objectid` of the overlay through Base's default
  `dataids`, and the constant-field proof read that call as an escape; and the
  mixed-activity handler did not recognize a reload as `Const`-argument data, so
  any read the proof does not cover raised in a large function what a small one
  answers with a zero-filled copy. The proof now knows `objectid` only reads,
  which needs no copy, and the handler accepts a reload whose field only ever
  holds such data, whether the slot is on the stack or libEnzyme moved it to the
  heap. Julia 1.12 reloads the tuple's pointers from a separate slot, which the
  proof does not reach: there such a read gets a zero-filled copy of the `Const`
  array per call, and split mode still raises where Base's `unalias` merges a
  reloaded index vector with its copy.
- Give each incoming edge of a merged value its own zero-filled shadow. The
  mixed-activity handler reused the copy it had made at the end of one incoming
  block on the merge's other edges, which that block need not dominate, and LLVM
  aborted with `Instruction does not dominate all uses!`.
- Keep the stores into a stack slot that only `objectid` reads. Enzyme declared
  the runtime call to read no memory, so LLVM could delete them, and the runtime
  then followed whatever pointers the uninitialized slot held: a segfault in any
  mode for `iseven(objectid(p))` as the last read of a struct `p`. A reverse pass
  also recomputed the call from the slot as the last loop iteration left it, so a
  derivative that depends on such a hash was silently wrong, with runtime activity
  too. It no longer assumes it can.
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
- Fix an LLVM verifier abort ("Call parameter type does not match function
  signature!") on Julia 1.10 when a function differentiated as a callee, or in
  split mode, takes `view` of a `SubArray` of a custom array type it holds on
  the stack (dogfood libEnzyme build `979b679a`). `view` calls `objectid` of
  the parent through Base's aliasing check, and Julia passes that stack slot to
  the runtime as an `i8` pointer. When Enzyme moved the slot to the heap, the
  call kept the slot's own pointer type; it now gets a bitcast to `i8`.
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

## 0.13.221

- Let independent Reverse adjoint sums inherit explicit reassociation permission
  from an eligible primal additive reduction, such as a Julia `@simd` sum. This
  enables vectorized coefficient gradients for row-wise residual sums without
  a mode opt-in or global fast math. Enzyme adds `reassoc` only to proven additive
  adjoint updates; the products and divisions it emits retain their configured
  flags. Strict primal reductions, dependent recurrences, aliased writes and
  atomic accumulation stay conservative. `fast_math_flags!()` clears global
  permissions; explicit primal reduction permission is handled independently.
  `Enzyme.has_dogfood_repair(:simd_adjoint_reduction)` reports the repair, and
  `test/simd_adjoint_reduction.jl` covers it.

## 0.13.217

- Add `WithGradient(primal, value_and_gradient)` for explicitly supplied
  derivatives of scalar-valued functions, including opaque native calls.
  Supports Forward and Reverse modes, batching, and reusable gradient buffers.
- Repair constant native function-pointer activity analysis and diagnose
  missing derivatives before handling indirect-call shadows. Avoid an
  out-of-bounds parameter-attribute query when optimizing an indirect call.
  Preserve inactive local buffers and symbol lookup, and report missing reverse
  derivatives outside the augmented primal's exception handlers.
