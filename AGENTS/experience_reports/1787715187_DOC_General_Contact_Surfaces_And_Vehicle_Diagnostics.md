# General contact surfaces and vehicle diagnostics

**Date/Version:** 2026-08-25 v1

## Overview

Removed ramp-specific physics semantics from Turing's generated AbstractUI
vehicle demo. Gradient geometry now publishes the same solid contact-surface ABI
as flat geometry. Added four-corner tire-patch and chassis-spring diagnostics,
surface-tangent tire bases, and a JSON-configured chassis solid-contact layer.

## Prompt History

> okay, it's unclear to me if the ramps as a special thing are a benefit to us at all, it seems like we should just trust the physics of gradient traversal by weight, contact patch, gravity, and the mrere physics of climbing. I want to see when in a vehicle a little color patch for each contact patch conveying in color the condition of the contact patch, then we would want to see the spring structure of the chassis and it's status from the spring physics. we also might need to tweak to have an extra mode or add a layer on top for vehicles - a more vehicle ready solid body contact with friction for the contact patches

## Steps Taken

1. Audited both browser-host and worker support sampling and found that the
   worker manufactured floor support even outside suspension reach.
2. Replaced ramp/support terminology with a general contact-surface contract;
   the wedge remains only a height-field-prism presentation/test shape.
3. Made wheel forward/right bases tangent to the sampled contact normal so
   gravity, spring load, pressure, and friction determine slope climbing.
4. Added JSON chassis-impact parameters and a worker-side penetration,
   restitution, static-friction, and kinetic-friction resolution layer.
5. Expanded the recycled snapshot ABI with patch area, friction demand/mode,
   and compression, then generated a four-corner mounted-vehicle HUD.

## Observed Behaviour

- The focused vehicle/state-loop/physics/page suite passed 67 tests.
- The regenerated worker and page JavaScript passed syntax checks.
- The in-app browser security policy blocked automated reload/inspection of the
  local `file://` page; the generated artifact itself was updated successfully.

## Lessons Learned

A gradient is geometry data used to construct a contact normal, not a movement
mode. Constraint projection may produce vertical velocity while a body climbs,
but that is the resolved normal constraint and must use the same path for every
surface. Tire and chassis contact should remain separate layers because their
material laws, shapes, and expected telemetry differ.

## Next Steps

- Compile triangle/BVH contact candidate generation so arbitrary meshes publish
  the same contact-surface records as analytic height fields and flat tops.
- Replace the current circular chassis-versus-AABB impact approximation with an
  oriented chassis manifold that can publish multiple simultaneous contacts.

