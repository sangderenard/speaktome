# Graph-Owned Adjoint Aggregate LLVM Execution

**Date:** 2026-08-14
**Title:** Complete aggregate backward calls through repository SSA and LLVM

## Overview

The existing ProcessGraph autograd path now carries aggregate backward-rule
results through source-call linking, isolated numeric regions, repository SSA,
and AOT LLVM without exposing intermediate or record-scratch buffers in the
public ABI. Both independent and combined backward requests retain the same
first-class `AdjointBindingGraph`; packaging selects execution form only.

## Steps Taken

- Audited the unresolved `bw_matmul -> matmul_vjp` call and the root buffer ABI.
- Refreshed caller ProcessGraph lookup inside the fixed-point call linker.
- Recognized one bound aggregate result with graph/indexed projection members.
- Removed unused conservatively allocated `returned_record_storage` arguments.
- Anchored aggregate producers on both their aggregate identity and projections.
- Made LLVM aggregate ABI specialization positional across call boundaries.
- Preserved caller-owned output descriptors for empty-`Ret` numeric regions.
- Materialized aggregate member buffers when the aggregate is consumed whole.
- Extended the real AbstractNN Linear + MSE test through an eight-step native
  SGD wrapper invocation.

## Observed Behaviour

- Root training-motion arguments are exactly the four authored inputs.
- `bw_matmul` is a native call producing aggregate 32 with projections 33/34.
- No source-call records remain unresolved in the demonstrated linear motion.
- Hand-authored ProcessGraph Linear + MSE passes exact native loss and gradient
  parity and a second parameter-updated invocation.
- Real `abstract_nn.Linear` + `MSELoss` passes exact native loss/gradient parity.
- The real AbstractNN artifact completes eight native forward/loss/backward/SGD
  steps, mutates weight and bias, and lowers loss without Python callbacks.
- Focused regression result: 22 passed, 1 deprecation warning.

## Lessons Learned

- A source call may bind one aggregate SSA value even when its physical native
  ABI has multiple output buffers; this is not a scalar return.
- Aggregate identity, not only projection IDs, is a real scheduling dependency
  when a numeric isolator consumes the aggregate whole.
- Source wrappers with real `Ret` values and isolated numeric regions with
  caller-owned empty-`Ret` outputs are distinct valid ABI cases.
- Output identity changes across calls; projection position is the stable
  correlation. Reusing caller IDs as callee return identities left outputs
  unwritten and produced NaN gradients.

## Next Steps

- Run the complete real XOR compilation fixture when the concurrent long-running
  capture is available and retain its growth census.
- Commit the linker work only after the large pre-existing dirty
  `fortran_c_shell.py` lane can be separated without staging unrelated changes.
- Continue the live compiler-event visualization and emergency clamp work after
  the native model loop remains green on a larger real model.

## Prompt History

> "think of it like, if the callign code asked for abackward, fused or independent - if it was in any way curious - it may as well receive by default in all cases this graph structure to make that seamless"

> "please try not to make up reverse operators when the abstract tensor autograd already contains backward mappings"

> "When we are done with this goal it is my intention that we can fuse down to llvm the complete motion of forward loss backward in one fused motion, leaving the optimizer as the only piece outside the hot loop for now"

