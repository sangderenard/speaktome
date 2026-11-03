# Three.js WebGL2 Vascular Renderer

**Date:** 1784575565
**Title:** Beginning the radar transition from SVG to a GPU anatomical renderer

## Overview

Introduced a production-bundled Three.js/WebGL2 renderer for the FluxGraph
radar. D3 remains the layout engine and SVG remains the interaction, guide,
label, heart, and HUD layer, but the primary anatomical geometry now renders
on the GPU.

## Implementation

- Added a strict TypeScript frontend module built by Vite.
- Pinned Three.js 0.185.1, Vite 8.1.5, and TypeScript 5.9.3.
- Added a WebGL2 canvas beneath the existing transparent SVG overlay.
- Batched physical edge hulls into one indexed quad mesh.
- Batched every individual audited traversal lumen into a second indexed
  mesh. Forward and reverse path tubes are separate strands, colored by their
  actual dominant fluid and animated by pressure/direction shader attributes.
- Bounded dense lumen lanes inside their hull instead of allowing the
  combinatorial traversal count to expand edge width without limit.
- Rendered node bodies and inner fluid fills with two `InstancedMesh` draw
  calls.
- Reused stable GPU buffers when topology size is unchanged rather than
  reallocating geometry on every D3 layout frame.
- Fed client-owned live relaxation tube compartments into `currentTubeState`
  during each visible substep, so the shader view follows the computation.
- Kept invisible SVG node hit targets for the existing tooltip and interaction
  behavior, giving a safe incremental renderer migration.
- Retained automatic SVG fallback if WebGL2 context creation fails.
- Added a production ES-module bundle under `flux_radar/webgl/`, served by the
  existing Python static handler.

## Verification

- TypeScript strict typecheck passed.
- Vite production build passed.
- `npm audit` reports zero vulnerabilities.
- 220 directly callable FluxGraph/radar/persistence tests passed; 23
  fixture-dependent tests were skipped by the direct runner.
- Python compilation, classic JavaScript parsing, and `git diff --check`
  passed.
- A headless Chrome WebGL2 smoke test confirmed:
  - the integrated page creates a WebGL2 context and activates the GPU layer;
  - the production bundle loads without console or page errors;
  - the isolated anatomical scene renders six draw calls and 300 triangles;
  - colored instanced nodes, edge hulls, and separately animated lumen lanes
    are visible.

## Lessons Learned

The simulation/rendering boundary was already suitable for GPU migration.
Resolved D3 positions and persistent per-segment tube state are enough to
populate GPU attributes without changing backend simulation contracts.

An orthographic camera with a downward-positive screen coordinate system
reverses face winding. Node circles therefore require double-sided materials;
this was caught by the real WebGL smoke render rather than syntax checks.

The combinatorial number of audited tubes cannot map to fixed-width lanes.
Compressing lane spacing into a bounded hull span preserves individual strands
while keeping the anatomy spatially coherent.

## Next Steps

Add a GPU color-ID picking pass for selecting an individual lumen strand, then
migrate pressure halos and optional guide geometry. Text labels, controls, and
the heart HUD should remain DOM/SVG because they benefit more from accessibility
and typography than from GPU rendering.

## Prompt History

> "to transition visualization we need to start using something serious, something that gives us opengl on the web interface"

> "proceed please"
