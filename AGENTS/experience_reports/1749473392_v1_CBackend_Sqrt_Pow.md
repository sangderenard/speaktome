# C Backend Pow/Sqrt C Implementation

## Overview
Implemented `pow` and `sqrt` in `CTensorOperations` using compiled C routines to avoid Python arithmetic. Added `sqrt_double` to the C backend and updated FFI declarations.

## Prompts
- "look into the accuracy and avoidance of translating to non c types or running non c operations in the c backend, correcting and mistakes you find in isolating the operation to occur on buffers through c not through python values"

## Steps Taken
1. Added `sqrt_double` function to the C source and FFI declarations.
2. Updated `pow` and `sqrt` methods to call C functions directly.
3. Ran `python testing/test_hub.py`.

## Observed Behaviour
Tests failed during collection due to missing optional `torch` dependency.

## Lessons Learned
Using C functions removes Python loops and keeps buffer operations native.

## Next Steps
Consider adding more C implementations for remaining operations and ensure optional dependencies are installed for full test coverage.
