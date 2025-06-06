# User Experience Report

**Date/Version:** 2025-06-08 v1
**Title:** MetaBeamManager Agnostic Conversion Study

## Overview
Explored feasibility of converting `MetaBeamManager` to use the `AbstractTensorOperations` wrapper so it can operate on torch or numpy tensors. Investigated how scoring functions are accessed and where `MetaBeamManager` is referenced across the repository.

## Steps Taken
1. Reviewed `meta_beam_manager.py` to catalogue direct torch dependencies.
2. Searched the repository for instantiations and usages of `MetaBeamManager` and `call_score_fn`.
3. Compared operations with existing methods in `tensor_abstraction.py`.
4. Considered alternatives for `F.pad`, concatenation, and top‑k selection using the wrapper interface.

## Observed Behaviour
`MetaBeamManager` heavily relies on torch for tensor creation, concatenation, top‑k selection and padding. It is instantiated inside `BeamSearch.apply_instruction` and accessed nowhere else. `call_score_fn` simply forwards arguments to user supplied scoring functions.

## Lessons Learned
Most tensor operations used by `MetaBeamManager` are already represented in `AbstractTensorOperations` except for padding and concatenation. The wrapper could be extended with `pad` and `cat` methods or these steps could be reimplemented using existing primitives. Because only `BeamSearch` instantiates `MetaBeamManager`, converting it will minimally impact other modules. Scoring functions could remain pure functions; `MetaBeamManager` merely orchestrates bins, so keeping `call_score_fn` here is reasonable although it might move to `Scorer` for simplicity.

## Next Steps
- Prototype a wrapper‑based version of `MetaBeamManager` adding missing operations (`pad`, `cat`).
- Evaluate performance with the PyTorch backend and implement the NumPy equivalent.
- Document the role of `MetaBeamManager` as an isolated scoring bin manager within `BeamSearch`.
- Consider relocating scoring policies to a dedicated module once the abstraction matures.

