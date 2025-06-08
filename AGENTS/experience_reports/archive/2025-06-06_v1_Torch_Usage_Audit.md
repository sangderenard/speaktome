# Torch Usage Audit

**Date/Version:** 2025-06-06 v1
**Title:** Audit of PyTorch usage and wrapper feasibility

## Overview
This report documents an audit of the `speaktome` code base to locate
explicit uses of PyTorch. The goal was to determine whether a lighter
alternative such as NumPy could be used and what refactoring effort
would be required.

## Steps Taken
1. Read `AGENTS.md` in the repository root and in `todo/`.
2. Searched the code base for `import torch` and direct calls to
   `torch.*` functions.
3. Examined major modules including `beam_search.py`,
   `compressed_beam_tree.py`, `meta_beam_manager.py`,
   `beam_graph_operator.py`, `pyg_graph_controller.py`, and
   `pygeo_mind.py`.
4. Identified tensor operations (creation, concatenation, slicing) and
   neural network layers that rely on PyTorch.

## Observed Behaviour
- PyTorch tensors are central to almost every data structure. Examples
  include token and score storage in `BeamTreeNode` and
  `CompressedBeamTree`.
- Many algorithms depend on `torch.topk`, tensor concatenation, masking,
  and moving tensors between CPU and GPU.
- `pygeo_mind.py` constructs a neural network using `torch.nn` modules
  and `torch_geometric` layers, which require PyTorch.
- The project relies on GPU support for efficiency. All major classes
  assume tensors live on a PyTorch `device`.

## Lessons Learned
Replacing PyTorch entirely would require re‑implementing tensor
operations and neural network layers in another library. While basic
array storage could be swapped for NumPy arrays, the GNN and scoring
components cannot run without PyTorch and `torch_geometric`.
A wrapper class could abstract some tensor creation and slicing, but the
underlying dependency on PyTorch would remain for model inference and
GPU utilisation.

## Next Steps
- Retain PyTorch for the core algorithms.
- Optionally design a thin wrapper for tensor creation so that simple
  data containers can fall back to NumPy when GPU support is not
  required. This would not remove the PyTorch requirement for the full
  application but might reduce the burden for partial functionality.
- No batch-processing algorithms were modified during this audit.
