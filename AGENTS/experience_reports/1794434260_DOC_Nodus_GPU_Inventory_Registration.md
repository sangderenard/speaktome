# Nodus native GPU inventory registration

## Scope and result

Nodus now has a source/CPU-verified path from supplied physical-device UUID
snapshots to ordinary per-device tool factories, the existing ContractNegotiator
Offer and native result FIFO. It does not query hardware or launch work during
registration. The implementation is `nodus/include/gpu_inventory_rows.h` and
`nodus/src/gpu_inventory_rows.cpp`; its detailed boundary is documented in
`nodus/docs/NATIVE_GPU_INVENTORY_ROWS_2026-11-11.md`.

Factories retain full UUID identity while Vulkan ordinals may change. A missing
provider, Windows identity or explicit node mask produces unavailable telemetry.
Structural provider availability and harness occupancy are not physical GPU
utilization. Exact approved operation identity remains the protected provider's
responsibility.

The existing module-library format saves UUID plugin IDs. Explicit restoration
on a stopped graph checks native FIFO types and actual factory-instance ownership,
then restores the typed stack and Sequential execution requirements. It does not
add a GPU-specific persistence format or claim automatic production reload.

## Verification

`gpu_inventory_rows_test.cpp` uses real native registry factories, contract
market/worker tools, result FIFO, module-library serializer and graph binding.
It covers unavailable/busy exclusion, generation retention, no replay on callback
removal, cancellation/commit, UUID rebinding after ordinal/order changes, swapped
instance rejection and close refusal while GPU instances remain bound.

`contract_registry_transaction_test.cpp` sweeps all allocation failures during
fresh registry pair creation, including after publication of the first factory.
Canonical ownership now precedes factory publication. Failed creation leaves no
pair or dangling factory; release uses preallocated keys and no allocations.
Existing foreign factory collisions are preserved. Native registry diagnostics
cannot interrupt completed insertion or rollback.

Verification receipts and direct build response files are under
`nodus/out/gpu-inventory-row/`. Tests use CPU callbacks only, with zero provider
link calls, GL calls, dispatches, hardware queries or child launches. No live
application or DLL was replaced, and no CMake regeneration was run.

## Next steps

Production integration still needs same-image factory bootstrap, general native
execution/stack metadata persistence, shared named automation port bindings and
monitoring display wiring. Persistent scene-buffer/stage execution ownership and
full producer-to-consumer validation are separate from this link-only routing
step. See `todo/nodus_gpu_inventory_production_binding.stub.md`.

## Prompt History

User: “we want this development work in native source, bespoke, because it's going to be a vital kernel in nodus”

User: “we are going to nodus now we are making native source that's going to use the tool row api, but it doesn't necessarily need to care about inputs or outputs except that it's getting a shader program, place it into the first best open gpu. gpu's will have their harness built into the gpu row. nodus routes the programs and they enter fifo and get cranked out by gpus that will have ports in that can recieve automation and deliver telemetry, the pieces are all in the canvas shell it's actually quite advanced”
