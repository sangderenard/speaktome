# Dewar dt-system damage-port viewer

Work on 2026-09-20 connected the existing engine-toy damage ABI to the live Turing chamber without introducing another damage or flow solver.

## Implemented

- `MachineSim` now accepts the same `record_penetration -> HoleEmitterField` path as `EngineCycleSim` and can bind the graph-discovered `FluidCircuit` objects. Its dt rollback excludes that external circuit reference so restoring the machine cannot clone and split the shared fluid ledger.
- `engine_toy/voxel_ports.py` projects one circular physical aperture onto axis-aligned voxel boundary faces. Face areas are normalized to the physical aperture area; a port spanning several voxels remains one port.
- Dewar wall graph nodes declare which atmosphere volume a through puncture opens and which jacket volume it crosses.
- The live chamber registers the existing `FluidCircuitSystem` beside thermal, atmosphere, and machine advances in the dt graph. The chamber voxel volume is registered as an atmosphere-owned fluid residence domain.
- Through punctures on declared chamber walls become ordinary pressure-driven atmosphere ports. Blind damage does not change the boundary. Inflow adds the existing ambient dry-air/water stream; outflow removes local voxel mixture. The atmosphere publishes damage-port count/area, pressure-drop rate, and sudden-decompression status.
- `chamber_raincloud_view.py` embeds the existing `EngineGLView`, weapon selector, ray picker, puncture SDF cutouts, absent-part state, thermal colouring, and emitter particles beside the atmosphere view. Right click applies the selected real cartridge through `RayMesh`; left click identifies the part.

## Verification

- Engine/dewar/fluid focused tests: 16 passed.
- Chamber damage-port plus chemistry packing/deployment tests: 11 passed.
- A direct `.50 AP` ray through the dewar produced punctures in both jacket sides and intervening parts; the through wall created one chamber port, and the next accepted 0.01 s dt window reported decompression and pressure-driven outflow.
- A live OpenGL snapshot completed at `t=0.01 s` and showed the atmosphere room, rotating machine inset, weapon controls, and all four dt participants. Artifact: `turing/artifacts/compiler_evidence/dewar_dt_damage_view.png` (ignored build output).
- A 0.01 s one-cell window advanced atmosphere, fluid, thermal, and machine clocks to exactly 0.01 s in about 0.26 s after startup.

## Known boundary

The cached b64/4^3 atmosphere piece currently reports the known invalid large velocity and refuses its step. This work did not compile or alter compiler code. The live command therefore defaults to the runnable one-cell artifact. Multi-voxel aperture projection is independently covered with a four-face conservation test and is ready for the b64 runtime once that artifact is corrected.

The chemistry HUD remains honest: packing and chamber phase laws are present, while the master inorganic chemistry solve is still pending and is not listed as a running dt participant.
