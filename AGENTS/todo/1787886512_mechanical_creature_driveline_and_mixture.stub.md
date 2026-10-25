# MechanicalCreature coupled driveline and mixture follow-up

- Integrate differential-brake rotor polar inertia through a coupled axle
  driveline mass matrix; do not approximate it as independent wheel inertia.
- Make reservoir sequencing and fuel/oxidizer mixing authoritative only after
  carrier chemistry, flow, air charge, spark timing, knock, failure, and live
  storage mass are represented.
- Promote wheel-bearing, knuckle, rotor-mount, and caliper-mount stiffness,
  reaction, inertia, plasticity, and failure metadata into the structural
  solver while preserving exactly one free wheel-spin degree of freedom.
- Add a bounded LVL pose bank and pose controller. Presets should publish
  chassis roll/pitch/height and four hub targets, then solve attainable
  per-corner actuator motion subject to linkage geometry, travel, rate, load,
  damage, and ground-contact constraints. Include user-saved poses.
- Add a topology-preserving wheelbase morphology operation: select front/rear
  chassis cross-sectional cut planes, move the rear axle-local subgraph,
  lengthen only tagged rails/driveshaft/body spans, re-solve routed throttle,
  steering and energy guides, and recompute rigid mass/center/inertia. Never
  scale the complete graph or detach joint identities.
- On live shock-parameter edits, recompute each coilover's display-only preload
  collar, wire radius, and active-turn count from the new rest length, stiffness,
  corner load and linkage motion ratio. Preserve fixed A-arm lengths and the
  existing live WASM input path; no SymPy rebuild is required.
