# CPU Demo Fix

**Date/Version:** 2025-06-19 v2
**Title:** CPU fallback restoration

## Overview
Running `run.sh` failed when PyTorch was missing. The CPU demo imported modules that unconditionally required `torch` and tensor operations lacked implementations required by the demo.

## Prompts
```
log in pay careful attention to entire script output then identify repo greatest need and fix it
```

## Steps Taken
1. Ran `bash run.sh` and observed `ModuleNotFoundError: No module named 'torch'`.
2. Made `beam_search_instruction` and `lookahead_controller` import `torch` and `BeamSearchInstruction` only when available.
3. Implemented missing tensor operations (`tolist`, `less`, `index_select`) for all backends.
4. Verified `bash run.sh -s "Hello" -m 5 -c -a 3` now runs the NumPy demo successfully.
5. Added this experience report and validated the guestbook.

## Observed Behaviour
After fixes the script prints top sequences instead of crashing:
```
PyTorch is not installed. Running CPU-only demo mode.
Faculty level: NUMPY
Top sequences:
ellocgcxt (score=-1.58)
...
```

## Lessons Learned
The CPU path exercised functions that weren't fully implemented for non-PyTorch backends. Optional imports and a more complete tensor abstraction keep the demo operational without heavy dependencies.

## Next Steps
Additional tensor ops may be needed as features expand, but the demo works for basic exploration.
