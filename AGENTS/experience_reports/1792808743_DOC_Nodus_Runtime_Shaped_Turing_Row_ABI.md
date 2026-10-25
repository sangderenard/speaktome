# Nodus runtime-shaped Turing row ABI

**Date:** 2026-10-23

**Repository:** `C:\dev\Powershell\nodus`

## Overview

Extended only the Nodus side of the Turing row-tool boundary. No Turing files
were changed. Nodus now preserves scalar, fixed tensor, and rank-known/runtime-
sized tensor contracts; generated C/LLVM wrappers derive extents from live
Nodus `AbstractTensorHandle` descriptors and allocate dynamic outputs from an
explicit input shape source.

The in-process canonical `abstract_tensor.*` tools were also moved from raw
`AbstractTensor*` stack values to the same `VT_ABSTRACT_TENSOR` handle currency
already used by generated vocabulary plugins and Turing artifact wrappers.

## Implemented contract

- `shape: []`: rank-zero scalar.
- Concrete `shape`: fixed tensor.
- `rank: N, shape: null`: runtime-sized tensor.
- `shape_from: <input>`: axis-for-axis allocation authority for dynamic
  outputs and related dynamic inputs.
- `NODUS BUFFER ABI V2` spells unresolved axes as `?` and uses
  `SHAPE <target> <axis> FROM <input> <axis>` records.
- V1 fixed-shape descriptors remain accepted.
- Dynamic Fortran adoption is refused explicitly until the legacy compiled-
  program API importer can carry output shape sources.

Turing artifacts continue to receive only `void** buffers` plus integer
extents. They do not see or depend on Nodus handles.

## Validation

- `tests/test_turing_row_tool_compiler.py`: 12 passed, 3 opt-in real-backend
  tests skipped.
- `graph_ir_test`: passed, including handle-based add, tanh, and sum.
- `artifact_buffer_import_test`: passed; one generated V2 plugin executed the
  same backend over two different vector lengths, with dynamic workspace and
  two dynamic outputs.
- `python_row_workflow_test`, `turing_product_adopter_test`,
  `artifact_buffer_import_test`, and `graph_ir_test`: 4/4 passed together.
- `git diff --check`: clean apart from existing line-ending warnings.

## Remaining boundary

The Turing workers still own rank-known tensor lowering and emitted dynamic
extent provenance. Nodus is ready to consume that C/LLVM product. Dynamic
Fortran output allocation remains an explicit Nodus shortfall.

## Prompt History

> there is a specific compile stage that injects shapes which can be skipped i think

> any turing changes i need to hand to the turing agents, can we check maybe if you're looking at maybe... did we compile and integrate today the tensor operators you were trying to use? i'm unclear on how the process of using the turing compiler for a tool would know it needs an abstract tensor handle, which is a nodus specific abstract tensor

> turing workers are securing your requests, if you don't need verification or api to do the nodus work feel free, it might be some time before the github main reflects your desires
