# Turing WASM: prod reduction gap and the mixed-extent output topology

**Date:** 2026-08-03
**Epoch:** 1785814138
**Title:** Closing the `prod` reduction hole and diagnosing why the flat WASM model rejects grid-shaped outputs

## Overview

Two threads ran through this session, both inside the Turing WASM fused-program
backend and its NumPy-oracle fidelity harness:

1. Extending the operator fidelity harness so operators can be unit-tested
   against the NumPy oracle "as extensively as we'd like," then chasing down a
   real gap it exposed: `prod` never lowered to a `FusedProgram`.
2. A real `build_site_page.py` build that failed emission with
   `output 'next_position_x' is grid/kaxis-shaped; the flat model cannot
   materialise an N*K output`, which turned into a conceptual investigation of
   *why the WASM backend is flat in the first place* and what the general
   answer to mixed-extent outputs is.

The handoff that explores the two implementation directions for thread 2 lives
at `turing/docs/WASM_MIXED_EXTENT_OUTPUT_HANDOFF.md`.

## Prompts

Verbatim, the instructions that steered this session:

- "you were doing a lot of dangerous second guessing me talk you want to
  explain what you're thinking"
- "I want to know why it is flat in the first place when that is not what it
  wants"
- "it sounds to me this is a complex topological question that needs a general
  answer not two answers"
- "I was not saying it's okay for prod to be missing I was telling you not to
  worry about axis/dim handling"
- "let's go look where it's missing instead of remove the fact that we would
  need it"
- "inconsistency is not a bug we had to walk a line between torch and numpy
  choices have to be made we just need to finess when it matters"
- "write an exprience report a la speaktome subrepo and a handoff document wher
  eyou explore the two options"

## Steps Taken

1. Added `verify_wasm_source(source, entrypoint, feeds, directory, ...)` to
   `src/compiler/wasm_fidelity.py` as the single-call operator test path
   (AOT compile → project → `emit_wasm_module` → `verify_wasm_module`), reusing
   the existing NumPy oracle (`compile_single_region_python(dialect="numpy")`)
   and deterministic case generator rather than inventing new comparison code.
2. Appended parametrized operator coverage to `tests/test_wasm_fidelity.py`:
   elementwise binary/compare/negate, and derived-grid axis reductions
   (`sum`/`min`/`max`/`mean`/`prod`). Grids are derived (`(px-ex)*(px-ex)`)
   because a direct N*K feed cannot be sized by the count-based ABI.
3. Ran the suite: 15 passed, `prod` failed at AOT with
   `the source shell's one discovery tape did not lower to one complete
   FusedProgram`.
4. Resisted the reflex to delete `prod` from the test. Instead traced the
   capture path: `compile_recorded_fused_tape`
   (`src/common/tensors/accelerator_backends/c_primitive_program.py`) classifies
   recorded ops via `_CAPTURED_NATIVE_KERNELS`. That table maps `sum`, `mean`,
   `min`, `max`, `any`, `all` to `"reduce"` but omitted `prod`, so a `prod`
   step fell through to the elementwise lowering and the tape refused to lower.
5. Added `"prod": "reduce"` to `_CAPTURED_NATIVE_KERNELS`. Re-ran: 16 passed,
   `prod` fidelity error 0.0 across all deterministic cases.
6. On the real build failure, read the reduction planner
   (`_plan_axis_reductions` in `src/compiler/fused_program_wasm_backend.py`),
   the WAT reduction body emitter (`_emit_reduction_body_wat`), and the main
   `emit_wasm_module` loop/store construction to locate exactly where "flat"
   is decided, rather than theorize.

## Observed Behaviour

- The `prod` gap was a single missing dictionary entry, not an axis/dim
  handling problem. The distinction mattered: the user explicitly noted that
  the `axis`/`dim` aliasing inconsistency across the codebase is a deliberate
  torch-vs-numpy line-walk and must **not** be "fixed," while `prod` genuinely
  needed to lower. The fix touched only the capture-kernel classification.
- The flat model is not a limitation bolted over a better design; it is the
  *correct and complete* form for a fused elementwise map, where rank carries
  no information the kernel needs (an `(N,K)` tensor and a flat `(N*K,)` tensor
  are the same bytes under one counter). Confirmed in the emitter's own module
  header comment and the single `(loop $body)` over `$count` with one
  output-element-per-lane store.
- Axis reduction is the one grafted-on exception, and it deliberately keeps the
  outer contract flat (`count = N`, one output per lane). Every rejection in
  `_plan_axis_reductions` is the same sentence — "this shape isn't flat over N
  with an inner fold over K."
- The grid-output rejection is a *placement gap*, not a topological
  impossibility: the backend already classifies every value's domain
  (`classify()` → grid/row/kaxis/scalar) and already computes the `i*K+k`
  address for grid *loads* inside the fold loop. The only non-general piece is
  that the output-store pass is hardcoded to the outer loop, one element per
  lane, so it can place only `{N}`/`{}`-domain outputs.

## Lessons Learned

- "Inconsistency is not a bug" — parallel op-name tables that disagree
  (`{"sum","mean","max","min"}` here, `+prod` there) can each be individually
  correct because they encode different torch/numpy boundary choices. The task
  is to find the *one* table that is genuinely wrong for the case at hand, not
  to homogenize them.
- When an operator "isn't supported," look for the missing registry entry in
  the capture/classification layer before assuming the emitter can't express
  it. `prod` was fully supported by both the WASM fold table (`_REDUCE_FOLD`)
  and the NumPy oracle (`_REDUCE_NUMPY`); only the front-end capture kernel map
  lacked it.
- The right response to a rejection message is often to ask *why the constraint
  exists*, then discover it is a narrow projection of a general rule. Here:
  flat (1-axis) and the N,K nest (2-axis) are both instances of "place each
  value at the innermost loop where its index exists." Grid outputs are the
  2-axis instance the store pass never learned to emit.
- Stop and explain thinking before acting on architecture questions. Twice this
  session I began sliding toward a large two-emitter change on an *assumption*
  about what `next_position_x` means. That is the "dangerous second-guessing"
  to avoid: the semantic question (is the grid output intended, or a
  reshape/broadcast artifact?) is the user's to answer.

## Next Steps

- Decide, from what `next_position_x` is meant to be, between the two directions
  laid out in `turing/docs/WASM_MIXED_EXTENT_OUTPUT_HANDOFF.md`:
  (A) make the store pass domain-driven within the existing 2-axis nest, or
  (B) the larger axis-set / loop-nest generalization.
- The reshape/keepdim NumPy-oracle divergence remains open: the numpy dialect
  lowering drops `reshape((-1,))` collapses, so the oracle can broadcast
  `(N,)+(N,1)→(N,N)` where the WASM respects meta. The fidelity hook correctly
  flags it; the fix (honor meta shape in numpy lowering) is unstarted.
- `prod` is now covered end to end; consider whether `cumprod` (also absent
  from the capture kernel map, unlike `cumsum`) needs the same treatment when a
  source that uses it appears.
