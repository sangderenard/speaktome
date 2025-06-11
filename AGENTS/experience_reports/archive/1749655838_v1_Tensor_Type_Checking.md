# Tensor Type Checking

## Overview
Implemented a lightweight `ensure_tensor` helper in the tensor abstraction and updated all backends to convert inputs automatically. Added a regression test demonstrating mixed-type stacking.

## Prompts
- "work on the tensors project (you won't be able to get the torch import to work even if you find the guidance to prep the environment correctly for it but you should be able to get the other backends to run as much as they're implemented...)"

## Steps Taken
1. Added `tensor_type` property and `ensure_tensor` method to `AbstractTensorOperations`.
2. Updated each backend to define `tensor_type` and use `ensure_tensor` in `stack` and `cat`.
3. Created a new pytest case `test_stack_mixed_inputs`.

## Observed Behaviour
Mixed lists and arrays now stack correctly across backends. Tests pass in the configured environment.

## Lessons Learned
Centralized type coercion keeps backends simple while allowing flexible inputs.

## Next Steps
None.
