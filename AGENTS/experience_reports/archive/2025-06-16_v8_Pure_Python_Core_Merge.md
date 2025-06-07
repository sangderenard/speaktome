# Pure Python Core Merge

**Date/Version:** 2025-06-16 v8
**Title:** Pure Python tensor ops moved into core

## Overview
Moved `PurePythonTensorOperations` from `speaktome.domains.pure` into the main tensor abstraction module. Added a helper to select tensor backends based on `Faculty` and updated demos and scorer to use it.

## Prompts
- "Unless you have an objection, put the pure python tensor abstraction stuff in the general tensor abstraction class, still dynamically importing based on faculty."

## Steps Taken
1. Integrated `PurePythonTensorOperations` into `core.tensor_abstraction`.
2. Added `get_tensor_operations` helper for faculty based selection.
3. Updated imports in `cpu_demo.py`, `testing/lookahead_demo.py`, and `scorer.py`.
4. Removed the obsolete `domains/pure` package.
5. Extended stub tests to include the new class.

## Observed Behaviour
All modules import correctly after the refactor and tests run with the new helper function.

## Lessons Learned
Centralising tensor backends simplifies dynamic selection and removes fragile relative imports.

## Next Steps
Monitor for any remaining references to the old path and expand pure Python tests in the future.
