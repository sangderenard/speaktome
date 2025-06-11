# Tensor Backends

## Quick Setup

```bash
python AGENTS/tools/dev_group_menu.py --install --codebases tensors
python AGENTS/tools/dev_group_menu.py --install --codebases tensors --groups tensors:jax,ctensor,torch,numpy,dev
```

This directory hosts the implementations of tensor operations for different numerical libraries (NumPy, PyTorch, JAX, etc.).

**Important:** Each backend must implement `_apply_operator` from `AbstractTensor`. This single method handles all arithmetic primitives. Avoid creating additional bespoke operator helpers – Python's magic methods already route standard arithmetic through `_apply_operator`.

Follow the repository coding standards and remember to run the test suite after modifications.
