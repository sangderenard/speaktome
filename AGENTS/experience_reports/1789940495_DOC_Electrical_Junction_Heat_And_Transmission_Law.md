# Electrical junction heat and transmission law

**Date:** 2026-09-20
**Title:** Electrical installation limits, thermal coupling, and distributed-line ABI

## Overview

Extended the cross-repository electrical work in `spectral-analyzer` and
`engine_toy`. Installed cable conductors and their two terminations now retain
separate electrical identities, resistance laws, safety metadata, and thermal
targets. The managed-dt electrical engine evaluates observable installation
conditions and emits real branch loss as heat into the existing thermal
assembly. A four-terminal RLGC law preserves propagation and resonance through
the existing complex multiport ABI.

## Steps Taken

- Kept cable conductors as arbitrary nodal-graph edges rather than introducing
  another scheduler or circuit runtime.
- Added 60/75 C copper ampacity declarations, small-conductor overcurrent caps,
  breaker continuous-load thresholds, listed-connector identity, and verified
  termination-torque state to `engine_toy/electrical_distribution.py`.
- Projected routed electrical conductors into copper path thermal domains in
  `engine_toy/thermal_domains.py`.
- Split each installed conductor into endpoint junction branches and a
  temperature-sensitive copper branch in
  `spectral-analyzer/machine_electrical_graph.py`.
- Added branch loss, heat-source publication, rollback state, and safety-event
  publication to `spectral-analyzer/electrical_dt_engine.py`.
- Added an exact uniform two-conductor RLGC multiport law to
  `spectral-analyzer/electrical_tensor_network.py`.
- Connected the dewar site's real 48 V battery, compressor motor, fan,
  protective earth, terminations, and routed conductors to that same graph.
  The compressor service uses six parallel 1/0 copper sets in separate
  raceways behind one 800 A breaker; the fan has its own AWG 14, 15 A branch.
- Registered electrical alongside atmosphere, machine, engine-cycle, fluid,
  and thermal as top-level participants in the existing dt round in
  `turing/examples/chamber_raincloud_live.py`. The existing realtime contract
  leaves each participant at its own causal ceiling and records the unadvanced
  interval in its time-slip error channel.
- Ran the focused electrical, station-powerplant, and thermal-recirculator
  tests.

## Observed Behaviour

- `spectral-analyzer`: 10 focused electrical tests passed. The tests observe
  conductor loss heating the routed copper path state, termination loss heating
  both endpoint-box volume states, overload conditions, unverified torque, a
  hot termination, and current conservation through the distributed-line law.
- `engine_toy`: 15 focused distribution, station, and recuperator tests passed
  with one pre-existing cffi deprecation warning.
- Station thermal assembly construction produced 40 routed conductor thermal
  states among 49 total states in a direct construction check.
- The dewar scene's electrical registry contains 67 branches. A direct
  one-voxel step measured 535.10 A compressor current, 26.302 kW compressor
  electrical input, 3.272 kW of electrical heat delivered to thermal targets,
  and no electrical safety events.
- A direct causal-ceiling check requested another 0.020 s while constraining
  only thermal to 0.005 s. Engine-cycle, fluid, and electrical clocks reached
  0.040 s; thermal reached 0.025 s and recorded 0.015 s of time slip. This
  demonstrates independent slipping in the actual dewar graph without an
  outer alignment scheduler.
- The final focused `engine_toy` dewar, distribution, and recuperator run
  passed 22 tests. The focused `spectral-analyzer` electrical run passed 10.
- A later correction removed the live scene's imposed 77 K cold-end initial
  condition and removed the unused aggregate `Liquefier`/`feed_dewar` state
  machine. The cold tip now starts at ambient temperature.
- The existing graph-discovered fluid engine retains per-node pressure and
  temperature for declared compressor/expander process loops. The production
  graph now routes working gas through eleven real pipe edges, including
  `expander -> cold_tip -> recuperator_return`, which had previously bypassed
  the cold tip entirely. Electrical motor power sets available compressor
  shaft work; compressor stages, recuperator, expander, cold-tip exchanger,
  radiator, and fan declarations determine fluid and thermal transactions.
- In the actual one-voxel dt scene, six 0.020 s rounds started the cold tip at
  293.15 K and brought it to 292.963 K from 23.146 kW measured compressor
  shaft input, 50 g/s through all eleven working-gas pipe edges, 7.007 kW
  expander shaft output, and 6.984 kW removed at the tip. An unpowered graph
  produces zero flow and zero cooling. The focused graph/fluid/thermal suite
  passed 24 tests after this correction.

## Lessons Learned

- Electrical topology is already arbitrary: a physical conductor is a
  two-terminal edge, while shared nodes, loops, parallel paths, and multiport
  stamps make the complete circuit graph.
- Fire-code installation rules do not supply one universal contact-resistance
  value. Junction resistance remains empirical; listing, torque, ampacity, and
  terminal-temperature constraints are represented separately.
- Spectral lanes must remain complex. Converting early to scalar watts would
  discard the phase and frequency information needed for resonance and
  transmission behavior.

## Next Steps

None required for this bounded change. Breaker trip curves and authored cable
RLGC matrices can later register through the same empirical device and
multiport interfaces when their physical declarations are available.

## Prompt History

> "apply standard fire code expectations of resistance at junctions and the transmission of heat energy to the wire mesh object state"

> "is the wire engine only point to point or can it handle realistic graphs"

> "okay as you work be sure to enrich to those ends, future proof things like resonances/transmission"

> "now make sure we can run our dewer site with all these advanced electrical capabilities and parts in our dt system with all the engines we've managed to register. also I want to be clear, at the current time, though tao can involve some accounting between slipping stages, I want us to be using a fully slipping system in time so each engine performs as fast as it can"

> "stop making cycles/engines for basic graphs that run on our existing physics"

> "liquifier.step might not belong existent in the code at all and might poison agent work suggesting it's okay to make any \"system\" or \"cycle\" you feel like, now please take out there fixed cold head and put in the real physics parts"

> "if you aren't using pipes and the fluid system go fuck yourself"
