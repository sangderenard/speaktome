# Tensor Backend Audit

**Date/Version:** 1749446805 v1
**Title:** Audit tensor backends and clarify operator policy

## Overview
Remove stale scalar operator helpers from the JAX backend and add guidance for backend implementations.

## Prompts
put an agents file in the tensors folder that highlights the fact that no backend should carry any function for operators other than the one required by the base abstract, since magic functions are now being used. this applies to your proposed edit to the jax backend, we don't want those functions and though your version of the repo right now might not reflect it sub scalar and div scalar have already been removed

## Steps Taken
1. Reviewed root guidance and tensor directory.
2. Removed `sub_scalar` and `div_scalar` from `jax_backend.py`.
3. Ensured `c_backend.py` ends with a newline.
4. Created `speaktome/tensors/AGENTS.md` documenting the operator policy.
5. Wrote this experience report and ran `AGENTS/validate_guestbook.py`.

## Observed Behaviour
No issues encountered. Repository tests pass.

## Lessons Learned
Scalar operator helpers are no longer needed thanks to Python's magic methods. Keeping backends minimal avoids redundancy.

## Next Steps
Monitor for further backend API changes and ensure documentation remains accurate.
