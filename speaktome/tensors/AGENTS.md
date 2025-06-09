# Tensor Backends

This directory hosts the implementations of tensor operations for different numerical libraries (NumPy, PyTorch, JAX, etc.).

**Important:** Each backend must implement `_apply_operator` from `AbstractTensorOperations`. This single method handles all arithmetic primitives. Avoid creating additional operator helpers like `sub_scalar` or `div_scalar` – Python's magic methods already route standard arithmetic through `_apply_operator`.

Follow the repository coding standards and remember to run the test suite after modifications.
