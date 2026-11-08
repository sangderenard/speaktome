# Nodus and Meowdoku service backbone: audit follow-up

The full read-only audit is `../1794080719_AUDIT_Nodus_Meowdoku_Service_Backbone.md`. It is also saved at the top workspace level, in `nodus/docs/`, `turing/docs/`, and `turing/examples/meowdoku/`.

## First engineering gates

1. In Nodus, reproduce or disprove the source-level partial-cycle free-spinning wait with a bounded test graph `A -> B -> C -> B`. Keep scheduling in `ThreadManager`; do not add another scheduler.
2. Prove no-token Input and full lossless Output behavior. For the service transaction, absence and backpressure must be explicit and must not produce a successful token. Keep `EdgeTensorFifo` as the sequencing authority.
3. Prove an embedded CPython row exception through the ordinary Nodus graph. The generated wrapper must report refusal so later Output rows publish nothing; retain the interpreter's restored inputs for inspection.
4. Specify one session action transaction and checkpoint boundary using the existing Nodus graph definition, native FIFO/transaction authority, and Meowdoku dt-system save/restore.
5. Prove one mixed Nodus graph for a Meowdoku action against current server behavior before moving more duties. Keep CPython/native deployment mode explicit and visible from retained product metadata.

Later: measure server Vulkan execution for a real fountain product, browser/server numerical divergence, thread creation per tick, and embedded CPython contention. A hosted Python process needs its own proved runtime protocol; the current build-time `python_executable` is not one.

The audited Nodus tree was at `3c793cc`; Turing was actively changing. Recheck current source and existing worker changes before implementation.
