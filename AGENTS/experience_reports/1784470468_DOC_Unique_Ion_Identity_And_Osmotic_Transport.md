# Unique Ion Identity And Osmotic Transport

**Date:** 1784470468
**Title:** Preserve network-side ion names and add pipe diffusion

## Overview

Each heart reservoir now reports its complete network-and-side ion identity in the Flux Radar HUD. Named ions also diffuse through traversal pipes down their own concentration gradients in addition to being carried proportionally by bulk fluid current.

## Steps Taken

- Traced reservoir identity from `IonReservoir.ion_name` through the server snapshot and heart HUD.
- Added `ion_name` to each reservoir's live snapshot telemetry.
- Replaced the HUD's hardcoded `ion` label with the full stored identity, such as `main:forward` or `net:42:backward`.
- Added a pipe-level osmotic ion transfer pass for every named soluble.
- Planned both directions from one concentration snapshot, allowing different ions to counterflow without update-order oscillation.
- Kept solvent on the existing bulk-current path; advective current continues carrying solvent and every dissolved ion proportionally.
- Recorded osmotic component flow independently from net bulk flow for frontend color and direction animation.
- Added regressions for full reservoir names and opposing ion flows with zero bulk current.
- Ran Python compilation, JavaScript parsing, focused transport/HUD regressions, and `git diff --check`.
- Restarted the Flux Radar backend.

## Observed Behaviour

- Reservoir telemetry contains `ion_name`.
- Heart storage values display the complete unique ion name rather than the generic word `ion`.
- With equal pressure and equal total volume, `main:forward` can diffuse one way while `main:backward` diffuses the opposite way.
- In that counterflow test, both ion concentrations equalize while net bulk edge flow remains zero.
- Existing bulk transport, reservoir gating, and soil transport regressions continue to pass.

## Lessons Learned

Bulk solution movement and species-specific concentration transfer are distinct simultaneous mechanisms. Edge telemetry must retain signed flow per species because different ions can move in opposite directions through the same pipe at the same time.

## Next Steps

None.

## Prompt History

> "it just says "ion" in the heart data but there is supposed to be a unique ion for each side of each network"

> "correct, and all ions need to transport according to osmotic transfer in addition to riding current"
