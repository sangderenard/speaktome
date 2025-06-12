# Batched ASCII Classification in draw_diff

**Date/Version:** 1749680315 v1
**Title:** Batched_Draw_Diff

## Overview
Implemented batch processing for ASCII conversion in `draw_diff` and
optimized `AsciiKernelClassifier.classify_batch` to explicitly repeat
interleave inputs and references before computing losses. This allows
each frame update to classify all changed subunits in one tensor
operation.

## Prompts
- System: "look for parallelization optimization in the clock_demo, particularly where we are judging similarities to a token set, I want to repeat interleave and compare every input to every output at once in one operation that is as efficient as its backend is when we insist it work in parallel as torch would"

## Steps Taken
1. Added `default_subunit_batch_to_chars` in `time_sync/draw.py` to
   classify batches of subunits.
2. Modified `draw_diff` to use the batch classifier when the default
   kernel is supplied.
3. Updated `AsciiKernelClassifier.classify_batch` to build explicit
   expanded tensors using `repeat_interleave` for clarity.
4. Ran `python -m py_compile` on updated modules.
5. Attempted `pytest -q` which failed due to environment setup
   restrictions.

## Observed Behaviour
- Syntax check succeeded with no errors.
- Pytest refused to run because the environment was not configured.

## Lessons Learned
Batching improved the clarity of the rendering loop and should enable
GPU acceleration when using the torch backend. Environment setup is
required to run the full test suite.

## Next Steps
Investigate vectorized resizing of subunits and ensure the demo still
runs correctly with large batches.
