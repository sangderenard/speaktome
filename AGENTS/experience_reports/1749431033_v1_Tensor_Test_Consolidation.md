# Tensor Test Consolidation

**Date/Version:** 1749431033 v1

## Overview
Merged individual tensor operation tests into a single parameterized suite and relocated the `Faculty` module under `speaktome.tensors`.

## Prompts
- "look for any tests that work individually on abstract tensor ops and replace and consolidate them along with total tensor abstraction tests into only one test with a flag for interactive or not, running through all available backends after first running a test on Faculty to determine what's available. move Faculty and any references to it to the tensors folder, use it as the gateway so that the abstract tensor will return either the selected ops or the best ops available."

## Steps Taken
1. Moved `faculty.py` to `speaktome/tensors/faculty.py` and updated imports across the codebase.
2. Added `Faculty` exports to `speaktome/tensors/__init__.py`.
3. Introduced new `test_tensor_backends.py` that iterates through available backends with an interactive timing flag.
4. Updated `tests/conftest.py` to provide `--interactive-tensor` option.
5. Removed the old `test_pure_python_tensor_ops.py` file and adjusted audit lists.

## Next Steps
Monitor future backend additions to ensure they integrate with the unified test.
