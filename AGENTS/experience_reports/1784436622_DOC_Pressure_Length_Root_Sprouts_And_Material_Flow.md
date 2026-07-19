# Documentation Report

**Date:** 1784436622
**Title:** Pressure-expanded pipes, root-like sprouts, and separate material flow

## Overview

Extended FluxGraph's hydraulic frontend and transport telemetry so pipe
housing length responds to pressure, local nutrient shortage drives
opposite-direction sprouts in time for growth selection, and water plus
each dissolved ion have distinct animated channels.

## Steps Taken

- Added signed per-material flow accounting to every edge.
- Exposed `component_flows` in radar snapshots.
- Made D3 link rest length a function of mean endpoint pressure.
- Added a pressure-length control.
- Rendered water as blue centerline dashes and ions as offset, stable-color
  animated lanes with an active-ion key.
- Read nutrient growth interest before expansion and reconciled it after
  transport without double-counting shortage.
- Added focused core and frontend regressions.

## Observed Behaviour

Focused core regressions pass directly. Python compilation, inline
JavaScript syntax validation, frontend source regression, and
`git diff --check` pass. The full pytest entry point remains blocked by
the repository's pre-existing environment gate.

## Lessons Learned

Aggregate pipe flow is insufficient for material visualization; composition
must be recorded at the moment a mixture is drained. Root demand already
existed as local directional state, but its placement after growth selection
introduced avoidable latency and could cause one extra sprout after supply
arrived.

## Next Steps

None required for this change.

## Prompt History

- "I would like if pressure expanded edges, if there was just a length factor owing to pressure so not only do edges straighten their joints on pumping but they should also elongate their housing. also make sure there is the same drive to make sprouts when needed, you know, so sprouts act like roots, they grow when necessary and wher eneeded like roots? also let's make sure we can see distinctly both water transport and individual ion transport all with their own colors and animations in the frontend please?"
