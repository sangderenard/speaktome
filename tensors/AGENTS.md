# Tensor Backends

## Quick Setup

Run `setup_env_dev.sh` to install this codebase. Extras for specific backends can be added via pip, e.g. `pip install -e .[jax]`.

This directory hosts the implementations of tensor operations for different numerical libraries (NumPy, PyTorch, JAX, etc.).

**Important:** Each backend must implement `_apply_operator` from `AbstractTensor`. This single method handles all arithmetic primitives. Avoid creating additional bespoke operator helpers – Python's magic methods already route standard arithmetic through `_apply_operator`.

Follow the repository coding standards and remember to run the test suite after modifications.
