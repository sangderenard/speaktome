# Tensor Abstraction BeamTreeNode

**Date/Version:** 1749414311 v1
**Title:** Tensor Abstraction BeamTreeNode

## Overview
Applied the convert_to_tensor_abstraction_job to BeamTreeNode and CompressedBeamTree. Added float_dtype helper to tensor abstraction interface.

## Prompts
- "draw job and perform task"
- `python -m AGENTS.tools.dispense_job` -> `convert_to_tensor_abstraction_job.md`

## Steps Taken
1. Added `tensor_ops` parameter to `BeamTreeNode` and updated initialization to use the tensor abstraction layer.
2. Extended `AbstractTensorOperations` with a new `float_dtype` property and implemented in all backends.
3. Updated `CompressedBeamTree` and `BeamGraphOperator` to pass tensor operations when constructing nodes.

## Observed Behaviour
`pytest -q` passes with all tests.

## Lessons Learned
Gradual replacement of torch code is possible by enhancing the abstraction layer with missing dtypes.

## Next Steps
Migrate remaining direct torch calls throughout `CompressedBeamTree` and `beam_graph_operator`.
