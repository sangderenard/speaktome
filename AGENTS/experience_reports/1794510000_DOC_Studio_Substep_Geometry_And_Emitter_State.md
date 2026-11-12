# Studio substep geometry and emitter state

**Date:** 2026-11-12

## Overview

The existing `recordStudioExposureFrames` is the state recorder for Studio
exposures. It samples every contracted tick, coalesces identical states, saves
PBR emission and named emitter profiles in each changed frame, and sends
vertices/material IDs only when geometry changes. The Python photo contract
accepts those optional geometry slices and keeps them in the job config.

The detached REST Studio product had a narrower `lightTimeline` implementation:
it rejected any fixture geometry change. That rejection was removed, and the
product now emits the same optional geometry fields when a later slice differs
from the initial fixture. The source scene builder still constructs every
authored state; it does not manufacture triangles or relax the backend.

## Verification

- `node tests/studio_agent_product_check.mjs` passed with a new moving fixture
  case and a material-only case that omits redundant geometry.
- `python -m pytest tests/test_meowdoku_marble_photo.py tests/test_meowdoku_studio_agent.py -q`:
  33 passed.
- Removed an attempted static-geometry standalone replay seam after the owner
  corrected the model. It was not deployed.

## Next Steps

The routed stage-light program still needs to be recorded through the existing
Studio substep clock and saved as an account-owned compressed state sequence.
Its Studio viewer should replay geometry updates, PBR emission, and the
matching Phong light uniforms. The authored REST graph and browser mixer must
drive the same sequence; do not add a fixed demonstration program. Render and
show an actual Phong result only after that path works. The current detached
REST product produces ray payloads, not a Studio Phong replay artifact.

## Prompt History

> hold on look at studio, not marble studio, fountain recordings, compressed geometry is our preferred method of saving, so we'd save just that, anyone viewing needs to run their phong shader of it in studio

> right, it will work because we can encode materials and that means we encode state slice by state slice the light power in the material's emissive properties

> do not lean on keep geometry once, use the existing code for substepping with geometry updates

> you would just use that to save state, see? it detects changes and then makes updates
