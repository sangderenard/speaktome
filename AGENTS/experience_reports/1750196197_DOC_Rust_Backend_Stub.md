# Rust Backend Stub

## Overview
Added a new Rust backend skeleton under `tensors/accelerator_backends`. A Cargo project provides the Rust library and a Python wrapper exposes the stub class.

## Steps Taken
- Created `rust_backend` folder with Cargo project files.
- Added optional `rust` dependency group in `tensors/pyproject.toml` and `AGENTS/codebase_map.json`.
- Updated accelerator backend package to import `RustTensorOperations`.
- Added stub test for the new backend.

## Observed Behaviour
Tests skipped due to missing environment setup as expected.

## Lessons Learned
The repository uses group-based dependency installation. New backends require updates to both `pyproject.toml` and `codebase_map.json`.

## Next Steps
- Implement real Rust tensor operations and async coordination.

## Prompt History
- "set up a rust backend in the tensors project in the accelerated backends please..."
