# Asciioscilliscope Radial Intersection Algorithm Proposal

**Date:** 1750628065
**Title:** Recursively refined radial intersection search

## Overview
This document proposes an isosurface-style algorithm for detecting beam intersections in the asciioscilliscope. It borrows ideas from marching cubes and iso-shell sampling to refine emissive intersection sets without evaluating every segment endpoint. The goal is to evaluate beam density only as a function of angle and distance from the gun, using broadcast operations and delayed computation.

## Algorithm Outline
1. **Initial Beam Rays**
   - Treat the electron beam as a collection of rays radiating from a common origin.
   - For a given frame, derive the initial angular resolution and energy density kernel from the gun parameters.

2. **Base-n Segment Subdivision**
   - Instead of marching endpoint to endpoint, subdivide each candidate segment into `n` subsegments.
   - Evaluate intersection functions on the subsegment midpoints. If an intersection is detected, recursively subdivide that subsegment only.
   - Continue subdividing until either a maximum depth is reached or the energy contribution falls below a cutoff.

3. **Concave Solid Mapping**
   - Assume intersections occur within concave solids representing phosphor wells or masks.
   - For each ray, maintain a bitmap of intersecting shells. Concave shapes are implicitly handled by clipping against previously found shells.

4. **Delayed Function Chains**
   - Represent intensity falloff and diffusion as chains of simple functions (scaling, radial weighting, temporal decay). Store these chains rather than computed values.
   - Evaluate the chain lazily once all relevant intersections for a beamlet are known. This reduces memory pressure and allows broadcasting across batches of rays.

5. **Broadcasted Intersection Batches**
   - Process rays in angle-sorted batches so that identical distance checks can be vectorized.
   - Use matrix operations (via Eigen or custom kernels) to update many bitmaps at once.

6. **Kernel Projection**
   - After intersections are resolved, project the beam density kernel into angular bins. The resulting distribution drives phosphor excitation via the `SampleSiteGrid`.

## Advantages
- Reduces the total number of intersection tests by focusing refinement only where intersections occur.
- Uses broadcasting to handle many rays simultaneously, improving cache locality.
- Delayed computation keeps the pipeline flexible and GPU-friendly if needed.

## Next Steps
- Prototype a C++ helper that performs the base-`n` subdivision search given a list of implicit surfaces.
- Explore mapping from intersection bitmaps to the existing `IsoShell` sampling code.
- Assess numeric stability when evaluating many small segments near concave boundaries.

## Prompt History
```
please propose for the asciioscilliscope an algorithm like marching cubes that matches emissive radial functions to emanate from a common point and recursively refine intersection bitmaps by not checking every segment's endpoints but performing base-n searches with n subdivisions inside each intersecting segment, for simplicity always mapping intersections inside concave solids, using delayed computation by deriving function chains and evaluating at last moment and practicing broadcasting wherever possible, such as intersection batches, all tuned to evaluate only the angular extent of the beam density kernel as an expression of the energy density as a function of angle and distance
```

## Conceptual Flag Extension
This proposal ties into a broader conceptual flag titled **Hash Beam Scatter Framework** (`AGENTS/conceptual_flags/Hash_Beam_Scatter_Framework.md`). That document formalizes how hashed steering indices and beam spillover kernels conserve energy and allow sub-pixel diffusion. The radial intersection search described above fits as a local refinement stage feeding into the hash-based scatter pass.

## Next Steps (Updated)
- Integrate the base-`n` subdivision search with the hash beam scatter framework.
- Investigate batching strategies for kernel spillover aggregation.
- Document misfire handling in the nanoscale simulator.
