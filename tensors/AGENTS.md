# Tensor Backends

## Quick Setup

Refer to `../ENV_SETUP_OPTIONS.md` for setup instructions.

This directory hosts the implementations of tensor operations for different numerical libraries (NumPy, PyTorch, JAX, etc.).

**Important:** Each backend must implement `_apply_operator` from `AbstractTensor`. This single method handles all arithmetic primitives. Avoid creating additional bespoke operator helpers – Python's magic methods already route standard arithmetic through `_apply_operator`.

Follow the repository coding standards and remember to run the test suite after modifications.

## Development Ethos

`AbstractTensor` mirrors the PyTorch API while accommodating idioms from NumPy,
JAX, and plain Python lists.  Operators are overloaded to hide backend
specifics, letting contributors work with whichever library they know best.
When behaviours diverge between libraries, PyTorch is the authoritative
reference.

Implementation priority follows this order:
1. Get the abstract interface fully specified.
2. Maintain feature parity across Torch, NumPy, JAX, and pure Python backends.
3. Address backend-specific gaps.
4. Expand the C backend only after the above are complete unless a simple stub
   can be filled quickly.
