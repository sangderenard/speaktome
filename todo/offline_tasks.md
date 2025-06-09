# Offline Tasks

This document enumerates work items suitable for agents operating without network access.

## 1. Convert To Tensor Abstraction
- **Reference:** `AGENTS/job_descriptions/convert_to_tensor_abstraction_job.md`
- Ensure all tensor operations use the abstraction layer from `speaktome/tensors/abstraction.py`.
- Replace direct PyTorch calls in modules under `speaktome/core/` with backend-agnostic calls.
- Add tests verifying the pure Python backend produces identical outputs.

## 2. Implement Failed Parent Retirement
- **File:** `speaktome/core/beam_search.py`
- Follow the STUB block starting at line 449.
- Remove failed parent beams from `active_leaf_indices` when lookahead pruning retires all children.
- Update retirement manager tests to cover this logic.

## 3. Expand Header Validation Helper
- **File:** `AGENTS/tools/validate_headers.py`
- Flesh out the stubbed helper to scan each module for the required header block ending with `# --- END HEADER ---`.
- Provide CLI options for interactive fix-ups and integration with `pre-commit`.

## 4. Recursive Test Runner Enhancements
- **File:** `AGENTS/tools/test_all_headers.py`
- Improve the stubbed script so it can locate and execute `Class.test()` methods in subprocesses.
- Capture results in JSON for `AGENTS/tools/format_test_digest.py`.

## 5. Pure Python Backend Initialization
- **File:** `speaktome/tensors/pure_backend.py`
- Determine if initialization needs configuration parameters for benchmarking or datatype handling.
- Document decisions in the module header.

