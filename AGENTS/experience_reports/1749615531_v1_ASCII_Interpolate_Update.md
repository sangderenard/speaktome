# ASCII Kernel Interpolation Update

## Prompt History
- User: "finish generalizing the algorithm in ascii_kernel_classifier by using the abstract tensor class's argmin and interpolate instead of np.argmin or pillow's resize"

## Notes
Implemented tensor-based interpolation for character resizing and removed PIL dependencies in `AsciiKernelClassifier`. Updated `_prepare_reference_bitmasks` to store tensors directly.
