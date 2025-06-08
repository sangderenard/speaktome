# Convert Torch Code to Tensor Abstraction Job

Some modules still call PyTorch APIs directly. To keep the project backend
agnostic, migrate these operations to use
`AbstractTensorOperations` (see `speaktome/tensors/abstraction.py`).

Steps:
1. Identify direct uses of `torch` or `torch.nn.functional`.
2. Replace them with methods from `AbstractTensorOperations` passed into the
   module or accessed via `config.tensor_ops`.
3. Update tests to run on multiple backends.
4. Note any behavior changes in an experience report.
