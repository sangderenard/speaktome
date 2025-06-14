# Tensor Abstraction Update

**Date:** 1749884297
**Title:** Tensor Abstraction Update

## Overview
Converted parts of `speaktome.core.scorer` to rely on `AbstractTensor` instead of direct PyTorch calls. Added a cross-backend test for `next_token_logprob_score`.

## Steps Taken
- Ran `python -m AGENTS.tools.dispense_job` and opened `convert_to_tensor_abstraction_job.md`.
- Updated scorer initialization to use `AbstractTensor.get_tensor()`.
- Rewrote `next_token_logprob_score` and portions of diversity metrics using the tensor abstraction.
- Added new `tests/test_next_token_logprob.py` validating behaviour across available backends.

## Observed Behaviour
Unit tests pass locally with the Pure Python backend.

## Lessons Learned
Some scorer utilities still depend on PyTorch-only ops such as `torch.unique`. Incremental migration is possible by wrapping simpler tensor creation calls first.

## Next Steps
Gradually replace remaining direct PyTorch calls and extend tests for the other scoring functions.

## Prompt History
- "draw job and perform task"
- "always check the files in the repo ecosystem for your benefit..."
