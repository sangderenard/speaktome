# Tensor Persistence Audit

## Overview
This report documents an audit of persistent storage features within the tensor abstraction layer.
The repository originally stored all tensor backends in `speaktome/core/tensor_abstraction.py`.
Persistent storage (saving/loading tensors) was not available for any backend.

To improve organization and enable persistence, the abstraction and backends were
moved into a dedicated `speaktome.tensors` package. Simple `save` and `load`
methods were added to the pure Python, NumPy and PyTorch implementations.

## Prompts
- "audit how the speaktome code base is using persistent storage when utilizing
  the abstract tensor class and prepare a report in markdown explaining
  discrepancies where they exist and the feasibility of adding persistent storage
  to each backend's ops. implement any that are trivial, such as pure and numpy."

## Findings
- Only `HumanScorerPolicyManager` and `BeamGraphOperator` used JSON for
  persistence; no tensor backend supported saving or loading arrays directly.
- `CTensorOperations` loads `libm` via `cffi`, making persistence more involved.
- `JAXTensorOperations` remains a stub.

## Feasibility
- **PurePythonTensorOperations**: trivial using `json.dump` and `json.load`.
- **NumPyTensorOperations**: trivial using `numpy.save`/`numpy.load`.
- **PyTorchTensorOperations**: straightforward with `torch.save`/`torch.load`.
- **CTensorOperations**: would need conversion to lists or a custom binary
  format; deferred.
- **JAXTensorOperations**: not implemented.

## Next Steps
- Consider binary formats for `CTensorOperations`.
- When implementing the JAX backend, mirror persistence helpers using
  `jax.numpy.save` equivalents.
