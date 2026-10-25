# Documentation Report

**Date:** 2026-08-25
**Title:** Turing AbstractUI state-loop deployment

## Overview

Converted Living Data Map world physics from a synchronous call inside the
graphics animation frame into a compiler-described fixed-step state loop on a
dedicated JavaScript worker. The worker wraps the existing SymPy → SSA → WASM
physics artifact and publishes acknowledged latest-only snapshots.

## Changes

- Added `state_loop_deployment.py`: neutral loop records, single-writer
  validation, effect-aware host placement, snapshot channel planning,
  annotation discovery without importing user code, and worker emission.
- Declared physics at 120 Hz on a worker, browser graphics at animation-frame
  cadence on the main thread, and actions as event driven.
- Moved player and projectile physics ownership to the emitted worker while
  retaining the former inline WASM execution as a compatibility fallback.
- Replaced allocating object snapshots with three preallocated transferable
  `ArrayBuffer` slabs. Stable slots and generation counters provide lock-free
  exclusive ownership and safe reuse without steady-state backing allocations.
- Added page emission provenance separating compiler products, AbstractUI
  model authority, and browser backend templates still awaiting generalization.
- Regenerated `docs/generated/abstract_ui_object_map.html`.

## Verification

- `tests/test_state_loop_deployment.py` plus
  `tests/test_abstract_ui_div_map.py`: 45 passed.
- Node parsed both the complete generated page script and embedded worker.
- A Node worker-thread compatibility harness instantiated the page's real WASM
  binary through the emitted worker, advanced a body, and observed gravity and
  horizontal motion. A second probe recycled and reused the snapshot boundary
  across successive publications. The boundary is now 10,240 bytes because it
  also carries control generations; an executable probe confirmed generation
  7 and its requested X/Z coordinates returned together.

## Navigation handoff correction

The first threaded version exposed an ownership race: auto-navigation or WASD
wrote a new X/Z position, then the main thread immediately applied an older
worker snapshot. Positions appeared to oscillate and manual direction felt
unresponsive. Each body now carries submitted and returned control generations;
the presenter preserves fresh control until physics has processed it.

The HTML overlay also mixed accumulated offsets/page coordinates with viewport
coordinates. Rendered room frames, map clicks, route lines, and entity dots now
use one bounding-client-rectangle transform relative to the map root, including
CSS scale and local scroll compensation.

The first HTML route overlay still jumped among room/building/courtyard charts
because it chose the smallest containing frame independently for every sample.
Forward projection now blends nested chart corrections to zero at their walls,
keeping the path and entity dot continuous. The certified traversal samples
also emit a narrow non-colliding world-space ribbon into the shared scene mesh,
so WebGL and software first-person views display the route without changing the
pathfinder or physics inputs.

The in-app browser refused direct `file:` navigation under its URL security
policy, so visual browser verification was not used. No attempt was made to
circumvent that boundary.

## Boundary

This does not claim the whole page is compiler generated. Physics WASM, worker
source, document model, and palette are model/compiler owned. DOM construction,
WebGL/Canvas presentation, browser input, inspector layout, and higher-order
follower integration remain backend-template code and are named as such in the
page model for subsequent migration.
