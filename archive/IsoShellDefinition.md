# IsoShell Design Concept

This note outlines a geometric helper for the `asciioscilliscope` project. An **IsoShell** represents the overlapping region between the trapezoidal CRT tube volume and the electron beam's conic spread. It acts as a simplified boundary for evaluating intensity falloff.

## Definition
* **Trapezoidal Pyramid** – interior CRT shape described by `TrapezoidalPyramid` in `Geometry3D.h`.
* **Conic Projection** – three‑dimensional volume returned by `ConicProjector3D::projectCone`.
* **IsoShell** – the set of points that lie inside both shapes at a given distance `d` from the gun. Mathematically this is `IsoShell(d) = TrapezoidalPyramid \cap Cone(d)`.

The shell can be sampled along the depth axis to produce weighting curves for beam diffusion or reflection models. When the beam cone is perfectly centered, the intersection resembles a truncated frustum. Any off-axis steering skews the intersection toward one side of the trapezoidal walls.

## Sampling Strategy
To approximate energy falloff, segment the IsoShell into thin slices along the beam path:

1. For each distance `d` from the gun, compute the cross-section of the trapezoidal pyramid.
2. Intersect that polygon with the cone section at the same depth.
3. Measure the resulting area to derive a weight for that slice.

Accumulating these areas yields a radial weighting profile that can drive `SampleSiteGrid` aggregation or later diffusion models.

## Complexity Notes
The first implementation should keep the intersection check simple (axis‑aligned trapezoid, symmetric cone). Later versions may account for magnetic field curvature by offsetting the cone center per sample.

