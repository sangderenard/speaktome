# Torch Bypass Feasibility Review

**Date/Version:** 2025-06-06 v2
**Title:** Exploring CPU-only extend/search with NumPy fallback

## Overview
Follow-up to the previous Torch usage audit. This pass investigates how minimal functionality (running `extend_once` and basic search) might work without downloading PyTorch. The aim is to let new users inspect the CPU-only path with NumPy arrays while keeping heavy NN dependencies optional.

## Steps Taken
1. Reviewed `beam_search.py`, `compressed_beam_tree.py`, and `meta_beam_manager.py` for direct calls to PyTorch functions. Key operations include tensor creation, concatenation, slicing, and calls to `torch.topk`.
2. Confirmed that `pygeo_mind.py` and GNN components depend on `torch.nn` and `torch_geometric`; these cannot run without PyTorch.
3. Examined lazy-install logic in `lazy_loader.py` for optional packages.
4. Considered how a light wrapper around tensors could delegate to NumPy when no GPU or NN is required.

## Observed Behaviour
- Core algorithms rely on PyTorch tensors for scoring and tree storage. `torch.topk` appears in `beam_search` and `meta_beam_manager` to select beam candidates.
- Extension routines (`extend_leaves_batch`, `extend_leaves_batch_lookahead`) manipulate tensors on GPU but convert to CPU/NumPy for Python loops.
- No batch-processing algorithms were changed during this investigation.

## Lessons Learned
- A minimal wrapper class could present an array-like interface with `to(device)` and slicing behaviour. For CPU-only inspection we could map it to NumPy arrays and implement our own `topk` using `np.argpartition`.
- `extend_once` and search loops could operate on these wrapper arrays as long as neural network calls are skipped. Guided transformer output would not be available without the language model, but users could still explore beam logic on sample data.
- PyTorch and PyGeoMind remain necessary for full functionality, yet a NumPy mode may streamline the initial setup.

## Next Steps
- Prototype a small wrapper (e.g., `SimpleTensor`) supporting `.to()`, slicing, concatenation, and a project-wide `topk` helper.
- Refactor functions that currently depend on `torch.topk` to use the helper so NumPy fallback becomes possible.
- Keep GPU and batch logic intact; the wrapper should only replace tensor storage and simple operations in CPU mode.
