# Documentation Report

**Date:** 2026-08-25
**Title:** Living Data Map probabilistic portal tube graph

## Overview

The initially conventional portal pair was corrected into a mesh-native,
many-to-many manifold graph. Primary/left action deploys an IN splat and
secondary/right action deploys an OUT splat on the exact rendered triangle.
Every IN owns a normalized spatial-Gaussian distribution over every OUT, with
visible relaxed-quaternion tubes as the intermediary traversable manifolds.

## Changes

- Replaced wall-opening portals with radial mesh splats that record the target
  triangle, barycentric center, neighboring coplanar triangle membership, and
  a local tangent frame.
- Published a `probabilistic-tube-graph` contract with many-to-many set size,
  normalized spatial-Gaussian distributions, directed tube edges, and a
  `relaxed-quaternion-cubic` path model.
- Builds every IN→OUT graph edge and normalizes weights separately per IN.
- Renders each edge as a real seven-sided tube over a relaxed cubic centerline.
  Its cross-section frame uses incremental shortest-arc quaternion parallel
  transport, avoiding up-vector snapping and exposing a continuous orientation
  field for later neural-network experiences.
- Entering an IN samples its edge distribution and moves players or physics
  balls through the chosen tube over time. Emergence maps local splat
  coordinates, velocity, and facing into the OUT frame.
- Added non-colliding blue IN and orange OUT splat geometry.
- Reserved those two palette colors as exact material entries in the compiled
  Pluck Phong adapter. No see-through portal shader was added.
- Persists splats and graph parameters. Saved legacy rectangular portals are
  migrated into graph splats and removed from wall-opening CSG.
- Increased portal-splat stock to twelve, documented the graph contract, and
  regenerated the living data map page.

## Verification

- Placement and living-map regressions: `55 passed`.
- Python source compilation succeeded.
- The regenerated page's complete inline JavaScript parses with Node.

Direct automated browser interaction with the generated `file:` page remains
unavailable under the browser URL policy; reload the existing user-owned tab
for manual interaction.

## Prompt History

> perfect work, let's work on the portal tool, make sure left is an in port, right is an out port, later we'll want to be able to back port sets with graphs, right now it's okay to make it conventional, don't worry too much about the portal see through shader unless you just desperately know how to add it into the source shader that was compiled into the pluck phong shader - just... just doing the portaling and having some color distinction of one side and the other side of each pair deployed

> what's supposed to be the placement tool, it's on randomly seeming, and when I have the portal tool it's on, and that's good because you're wanting to place a portal, but it doesn't put a portal in the tool for you to place, it's just that tool on whatever you are facing

> right or left click, I don't see a portal show up

> a portal such as you describe is not as useful to us as it is to mark and divide triangles around a target abstractly in the mesh and connect their spacial manifold to the spacial manifold of the other splat, you know? I think that might be faster easier and more intuitive

> I would prefer a probabalistic distribution from ins to outs through tubes as the intermediary manifold, but that's extravagant, but i want it - hah - just a graph and you travel the graph in the tubes according to your path and later we can simulate the experience of being in a neural network with spatial gausseans through the network layers

> if you asked me tubes should be relaxed quaturneon paths

## Next Steps

- Generalize the explicit portal graph into authored neural layers and spatial
  Gaussian activation fields without changing deployed splat identities.
- Expose graph sigma, traversal speed, and tube relaxation as inspectable
  controls when the interaction design is ready.
- Add a portal-camera/render-target pass only when see-through presentation is
  explicitly in scope.
