# Audit Report: Modular Import Attempt

**Date:** 1749795823
**Title:** Attempting to use packages as modules without full environment setup

## Scope
This audit explores whether the various projects in the repository can be imported as Python modules out of the box.

## Methodology
1. Checked repository guidelines and coding standards under `AGENTS.md`.
2. Attempted to import `speaktome`, `tensors`, `fontmapper`, `laplace`, and `time_sync` inside Python.
3. Investigated error messages and environment requirements when imports failed.
4. Reviewed environment setup scripts and header utilities.

## Detailed Observations
- Importing `speaktome` or `tensors` immediately prints the `ENV_SETUP_BOX` message and exits because the `ENV_SETUP_BOX` environment variable must be defined and the required packages are missing.
- `laplace` imports succeed until a submodule requires `torch`, which is not installed in the default container.
- `python testing/test_hub.py` fails because the automated environment setup cannot complete due to missing dependencies and network restrictions.
- Stub implementations exist within `tensors/accelerator_backends/c_backend.py` for many tensor operations. The simple `to_dtype_` stub returned the original tensor without casting.
- Implemented a basic casting behaviour for `to_dtype_` so tests that rely on it can run once dependencies are installed.

## Analysis
The design intentionally aborts if the environment is not fully configured, guiding users to run the provided setup scripts. This ensures optional heavy dependencies like PyTorch are installed only when requested. However, for lightweight module exploration the strict check on `ENV_SETUP_BOX` and immediate `sys.exit(1)` prevents partial usage. While this encourages consistent setup, it reduces flexibility for read-only auditing or minimal dependency scenarios.

## Recommendations
- Document a minimal environment variable configuration for read-only imports, e.g. setting `ENV_SETUP_BOX` manually so modules do not exit, even if features requiring Torch are unavailable.
- Provide clearer instructions in `ENV_SETUP_OPTIONS.md` on how to run tests with optional groups skipped.
- Continue refining stub implementations so that basic operations succeed in a pure Python environment.

## Prompt History
- "perform an audit experience report and any trouble ticket reports necessary in the process of attempting to use projects. the intention is for them to be used as modules so try not to make any proactive edits that would support starting the environment in a way to use the files directly in their own root (unless that can be done while also achieving the modular status correctly?)"
- "always check the files in the repo ecosystem for your benefit... Any time you notice an error in a test investigate, you notice a stub you can implement implement it."
