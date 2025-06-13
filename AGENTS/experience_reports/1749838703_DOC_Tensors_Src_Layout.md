# Documentation Report

**Date:** 1749838703
**Title:** Implement src layout for tensors

## Overview
Poetry failed to build the `tensors` package because it expected a package directory inside `speaktome/tensors`. The project kept its modules directly next to `pyproject.toml`, so the build backend could not locate them.

## Steps Taken
- Created `tensors/src/tensors/` and moved all Python modules and subpackages into this directory.
- Added a simple wrapper `tensors/__init__.py` that re-exports everything from `src.tensors` for compatibility.
- Updated `tensors/pyproject.toml` with `packages = [{include = "tensors", from = "src"}]`.
- Confirmed `pip wheel --no-cache-dir --use-pep517 --editable "./tensors"` now builds successfully.

## Lessons Learned
Aligning with the standard `src/` layout avoids ambiguous package resolution. Editable installs succeed once Poetry knows where to find the modules.

## Prompt History
User demanded the agent locate and correct references to the obsolete path `/workspace/speaktome/tensors` and restore experience reports.
