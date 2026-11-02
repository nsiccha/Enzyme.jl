# Changelog

## 0.13.217

- Add `WithGradient(primal, value_and_gradient)` for explicitly supplied
  derivatives of scalar-valued functions, including opaque native calls.
  Supports Forward and Reverse modes, batching, and reusable gradient buffers.
- Repair constant native function-pointer activity analysis and diagnose
  missing derivatives before handling indirect-call shadows. Avoid an
  out-of-bounds parameter-attribute query when optimizing an indirect call.
