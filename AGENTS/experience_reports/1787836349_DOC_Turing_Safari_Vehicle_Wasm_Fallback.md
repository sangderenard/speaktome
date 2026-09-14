# Turing Safari vehicle Wasm fallback

**Date:** 2026-08-27
**Title:** Re-enabled fixed-tick vehicle fallback when worker WebGPU is unavailable

## Overview

Reconnected the compiler-emitted wheel-contact and complete vehicle WebAssembly
artifacts to the Living Data Map physics worker. WebGPU remains preferred, but
the worker now starts and continues vehicle simulation through Wasm when WebGPU
is unavailable or faults, including iPhone/Safari configurations without worker
WebGPU.

## Steps Taken

- Packaged the existing vehicle/contact Wasm plugins in the world model.
- Passed both binaries and published ABIs from the page into the sole physics
  worker.
- Restored scalar contact dispatch, wrench reduction, cage contact, and compiled
  vehicle transition within the existing 120 Hz tick.
- Preserved pose, wheel omega, compression, filtered slip/utilization, and
  engine-speed state across WebGPU/Wasm backend changes.
- Kept the existing 71-value transferable snapshot ABI and presentation path.
- Added worker contract and model packaging assertions.

## Observed Behaviour

- Five state-loop/fallback tests passed.
- Generated worker JavaScript passed Node syntax validation.
- The emitted full vehicle Wasm execution test passed and produced nonzero wheel
  angular velocity from throttle.
- The corrected scalar contact-Wasm execution test passed and produced chassis
  suspension load.
- The full projection built with both fallback plugins; its existing unrelated
  four-versus-28-lane reducer assertion still fails before later assertions.

## Lessons Learned

The fallback mathematics had not been deleted; only its runtime adapter had been
disabled. Keeping fallback work inside the same worker/tick avoids a second state
owner and lets WebGPU upgrade or failure occur without changing presentation or
snapshot contracts.

## Next Steps

- Confirm backend reporting and driving behavior on a physical iPhone both with
  WebGPU available and with it unavailable/disabled.

## Prompt History

> When you are done with your present task please investigate an iphone/saphari incompatibility causing no wheel motion

> okay, reenable the fallback it was disabled for testing isolation
