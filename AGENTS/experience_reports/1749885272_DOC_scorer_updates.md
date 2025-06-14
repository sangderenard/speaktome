# Scorer Tensor Usage Refresh

**Date:** 1749885272
**Title:** Scorer Tensor Usage Refresh

## Overview
Adjusted scorer functions to treat `AbstractTensor` instances as full tensors rather than relying on a separate ops object.

## Steps Taken
- Revisited `scorer.py` after receiving guidance about the new abstraction style.
- Removed obsolete `ops` variables and calls to `ensure_tensor`.
- Updated tensor creation and indexing to use the tensors themselves.
- Ran `python testing/test_hub.py` *(fails: Environment not initialized)*.

## Observed Behaviour
Code compiles and tests still skip due to missing environment setup.

## Lessons Learned
The abstraction now mirrors PyTorch so operations originate from the tensor itself, simplifying the API.

## Next Steps
Continue porting remaining utilities to the updated style and verify tests once environment setup succeeds.

## Prompt History
- "draw job and perform task"
- "always check the files in the repo ecosystem for your benefit..."
- "you're just slightly out of date, we don't get ops and use them anymore..."
