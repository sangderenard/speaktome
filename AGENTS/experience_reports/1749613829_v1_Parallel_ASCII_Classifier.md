# Parallel ASCII Classifier

## Overview
Updated `AsciiKernelClassifier.classify_batch` to use the tensor abstraction for
batch computations across backends.

## Prompts
- User: "parallelize through abstract tensor from tensors the classify_batch function in ascii kernel classifier from clock_demo in time_sync. obtaining for functions that will require a specific data type the raw data from the backend that provides that type."

## Steps Taken
1. Added repository header to `ascii_kernel_classifier.py` and integrated tensor backend detection.
2. Implemented vectorized loss calculation using PyTorch or NumPy depending on the active backend.
3. Ran `python testing/test_hub.py` (failed due to missing packages).

## Observed Behaviour
The classifier now stacks reference glyphs and processes the entire batch with broadcasting. Tests could not execute because dependencies were unavailable.

## Lessons Learned
Tensor backends simplify dtype handling and device placement, but optional dependencies must be installed for the full test suite.

## Next Steps
Install the recommended extras and rerun the tests once all packages are available.
