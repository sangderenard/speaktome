# to_backend backwards compatibility removal

## Overview
Removed the optional tensor argument from `AbstractTensor.to_backend` to enforce
a single invocation style: `tensor.to_backend(target_ops)`. Updated tests and
modules accordingly.

## Prompts
- "Remove backwards compatibility there is a strict methodology and it is not to offer any general functions not related to an existing abstract tensor if we have need for class methods they will go in tensors/F just like torch"

## Steps Taken
1. Simplified `AbstractTensor.to_backend` to accept only the destination backend instance.
2. Adjusted `ensure_tensor` to use the new method when converting PyTorch tensors.
3. Fixed `AsciiKernelClassifier.ssim_loss` to convert `candidate` and `reference` tensors correctly.
4. Updated `test_tensor_backends` to use the new invocation style.

## Observed Behaviour
The API now requires calling `tensor.to_backend(target)` and rejects the previous `ops.to_backend(tensor, target)` form.

## Next Steps
Run the test suite once dependencies are installed to confirm consistent behaviour across backends.
