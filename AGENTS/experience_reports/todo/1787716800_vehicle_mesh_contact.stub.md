# Vehicle arbitrary-mesh contact follow-up

For the current contact-kernel/GPU-resident wrench transition state and the
required finish sequence, first read
`../1787759061_DOC_Turing_Vehicle_Tensor_WebGPU_Transition_Handoff.md`.

Compile packed-BVH traversal, closest triangle feature/contact generation, and
physics-material lookup behind the existing general contact-surface ABI. Emit
both four-lane WebGPU and batched Wasm paths from shared repository SSA. Keep
the fixed-step worker as the sole state owner and benchmark GPU-resident
force/torque reduction before adding scheduling concurrency. Replace the
circular chassis side proxy with an oriented multi-contact solid manifold.
