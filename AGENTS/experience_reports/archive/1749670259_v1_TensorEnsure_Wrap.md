# Ensure Tensor returns wrapper

## Overview
Adjusted `ensure_tensor` to return an `AbstractTensor` instance instead of raw backend primitives. Updated all callers and backends to access `.data` explicitly.

## Prompts
- "You should never be returning a bare data. If torch had a test of it's a tensor and if not make it one function it wouldn't return the internal primitive"

## Steps Taken
1. Modified `ensure_tensor` signature and behavior in `abstraction.py`.
2. Updated backend implementations and methods to use `.data` when passing primitives.
3. Adjusted `AsciiKernelClassifier` to handle the new return type.
4. Attempted to run `testing/test_hub.py` but pytest and other dependencies were missing.

## Observed Behaviour
The code compiles successfully. Tests could not run due to missing dependencies.

## Next Steps
Install requirements via the dev setup script so the test suite can run.
