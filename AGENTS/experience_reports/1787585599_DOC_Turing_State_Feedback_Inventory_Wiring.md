# Turing state-feedback inventory wiring

**Date:** 2026-08-24
**Title:** Name and alias state feeds before class inventory construction

## Overview

Connected a page contract's `state_feedback` mapping to the segmented Wasm
class graph. An originless external feed can now recover its declared state
input name before `build_class_inventory` consumes the manifest, and the input
field is redirected onto the corresponding output field's resident storage.

## Steps Taken

- Read the workspace and Turing-specific guidance and test hazards.
- Confirmed that `build_embedded_class_graph` already owned the general
  `storage_redirects` seam, but `site_bundle` did not pass `state_feedback` to
  it.
- Added a conservative fallback for originless external feeds: it is used only
  when every unnamed feed is accounted for by a feedback input, ordered by the
  authored function parameters.
- Added a focused regression that builds the graph and then the actual class
  inventory, proving `phase` and `phase_next` share one field slot.
- Recorded unrelated failures exposed by broader focused test files in
  `turing/TEST_BASELINE_AND_HAZARDS.md` instead of descending into them.

## Observed Behaviour

`python -m pytest tests/test_wasm_class_modules.py -q --tb=short -k
"embedded_graph_can_redirect or state_feedback_names"` passed 2 tests.

The full `tests/test_wasm_class_modules.py` run passed 27 tests and exposed one
unrelated deprecated-AOT test failure. `tests/test_wasm_class_coordinator.py`
passed 9 tests and exposed one unrelated vector-broadcast failure. Both are
recorded in the baseline manifest.

The repository-local and shared virtual environments both point to a removed
Python 3.10 installation. The available Python 3.11 environment already had
the required test dependencies, so it was used without installing anything.

## Lessons Learned

State feedback is a storage identity decision and must be present before the
inventory canonicalizes field keys. Renaming the public manifest afterward
cannot change the coordinator's already-built slot assignments.

An ABI name is safe as fallback evidence only when the mapping is complete.
Partial positional guessing can attach persistent state to the wrong feed, so
ambiguous inputs remain synthetic and fail the existing redirect validation.

## Next Steps

The page was subsequently published to the workspace gallery as
`pointwise-probe` version `v1-68f4dc9f6621f25a`. Browser verification showed
the actual contiguous Wasm running 1,024 elements with completed frames around
0.0--0.1 ms and no `input_4` error. The retained control regions omit the
declared `next_phase` feedback output, so deployment selection now withholds
that incomplete staged variant and exposes the complete contiguous program.

The black full-screen surface was subsequently repaired and published as
`v18.001-20260824-d567dbe409d19f2e`. The page had promoted the program's
four-output compute shaders (`next_phase` plus RGB) directly to presentation:
the WebGPU shell allocated only the first output buffer, while the WebGL path
could not present four draw targets to the default framebuffer. When the
compiler's RGB passthrough is incomplete, the publisher now selects its
Wasm-backed Canvas 2D presentation instead. That path gained an explicit
display-only channel scale and nearest-neighbour stretching from the 32x32
logical frame to the opaque full-screen canvas. Three focused regression tests
passed, the bundle's Wasm fidelity proof passed all three cases, and both the
local and GitHub Pages copies were visually verified in Chrome as nonblank
full-screen output. Root-site commit: `3cbc611`.

The two unrelated current-tree test failures remain documented in the baseline
rather than promoted into this task's scope.

The `pointwise-probe` page was a pipeline test, not the intended demonstration.
The actual compiled Kuramoto WebGPU field was then built with
`tools/build_kuramoto_webgpu.py` and published at
`site/demos/kuramoto-field/` in root-site commit `d7d686f`. The public GitHub
Pages copy was visually verified in Chrome: the 96x96 field advanced through
step 31, all 9,216 low limbs remained live, the field stayed finite at
approximately `[-5.91, 5.66]`, and coherence reached `0.6646`. Build stamp:
`9d5d73af`.

## Prompt History

> "The remaining blocker: `wasm_class_modules` names a state-carried feed after
> the *value* (`input_4`) while the API names it after the *input* (`phase`).
> `state_feedback` needs threading into `build_embedded_class_graph` — patching
> the manifest afterwards fails because `build_class_inventory` has already
> consumed it."

> "Rule: every third blocker, state the goal out loud and ask whether the chain
> still serves it. \"The next error\" is not a plan."

> "`.turing-cache` is 10 GB of AOT checkpoints and is load-bearing for build
> times. Don't touch it."
