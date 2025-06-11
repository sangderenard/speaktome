# Tensor Indexing

**Date/Version:** 1749682629 v1
**Title:** Implement get/set item in AbstractTensor

## Overview
Implemented __getitem__ and __setitem__ methods in the AbstractTensor class and added a test verifying basic indexing across simple backends.

## Prompts
User request:
"implement get item and set item in the tensor abstract, directing it to the ... is it safe to just universally throw the subcript operator at all the backends data except c? they all offer slicing do they not?"

## Steps Taken
1. Reviewed tensor backends and existing abstraction.
2. Added generic indexing helpers in `tensors/abstraction.py` skipping the C backend.
3. Created `tests/test_tensor_indexing.py` to exercise the new methods with PurePython and NumPy backends.

## Observed Behaviour
New tests pass for available backends when running the suite.

## Lessons Learned
Most backends support Python slicing directly so generic indexing works except for the C tensor, which needs a custom implementation.

## Next Steps
Consider adding CTensor support and more complex indexing tests.
