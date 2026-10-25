# Fluid route audits and strap slip

Migrate the 257 legacy fluid edges marked
`route_requires_obstacle_audit=True`, beginning with station service trunks
and gun coolant runs.  Explicitly choose `rigid-orthogonal`, `flexible-hose`,
or `trivial`; only obstacle-aware hard-line routes may clear the audit marker.

Implement the nonlinear release of `strap-clamped-contact` after its declared
friction capacity is exceeded, retaining the strap plate and both bolted
tensioners.  Test locked-below-capacity and slip-above-capacity behavior.

After those bounded mechanics tasks, resume whole-module full-native lowering
of `EngineCycleSim.step` through engine_toy's ExtractionContract.  Do not treat
the new construction-time external fuel graph as proof of native lowering.
