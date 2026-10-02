# Starter garage corrected to real game Machines

## Result

Removed the parallel mini-engine introduced in the first starter-garage pass.
There is no longer a `StarterGarage`, `ExhaustPassage`, `LintDeposit`,
`MaterialReceiver`, `TextileLoad`, or `GarageDoorState` class, nor a garage
snapshot/restore path, custom pressure-loss calculation, fan-curve solver, or
zero-mass `inventory-object` scene graph.

`starter_garage.py` is now authoring/composition only. It returns existing
`Machine` objects and the existing `AirVolume`. Every loose hand tool,
measuring tool, power/cleanup item, spare, stock item, and safety/reference
item is a Machine. Assemblies such as the overhead door, workbench, electrical
distribution, washer, dryer, and building exhaust are Machines containing
their real component parts and lines. Unknown product masses and dimensions
remain marked unresolved rather than gaining invented physics.

The washer and dryer remain cabinet production graphs. Dryer service parts,
fixed-model interlocks, and the process-air circuit were added directly to
`appliance_production.py`. Seeded faults annotate the actual belt, idler,
roller, switch, latch, connector, thermal fuse, or heater component. The
building exhaust is a separate Machine with one circuit discoverable by the
existing fluid-circuit engine. Its obstruction is a part using `fouling.py`'s
`core-debris` vocabulary; the numerical blockage and material calibration are
explicitly unresolved because the authoring brief supplied no calibration.

Electrical panels and receptacles are built through the existing electrical
hardware builders. Runtime state, damage, and snapshots belong to `MachineSim`
and the managed engine systems. The temporary water-vapour extension to
`AirVolume` was removed rather than retaining an extra laundry-specific
transport implementation.

## Verification

- 88 tests passed across starter garage, cabinet/appliance, autoclave,
  electrical hardware/distribution, machine package/options, fluid routing
  and circuits, thermal recuperation, and frame vocabulary.
- The exporter produced 162 catalogue entries, 40 constituent records, and
  245 ordinary Machine graphs.
- Existing fluid discovery finds the building exhaust as exactly one
  `garage.dryer-exhaust-air` circuit.
- A source-level regression test refuses the removed parallel runtime class
  and solver names.
- One existing cffi `imp` deprecation warning remains.

## Prompt History

> did you make bespoke programming instead of using the game engine

> remove all the bespoke material and make the game engine parts inside the
> relevant engines, don't layer on physics use the real object system make
> real machines, let's say for right now convenience that tools are machines

## Next Steps

Calibrate the authored vent deposit through the existing fouling/fluid system
when measured or deliberately authored obstruction data are supplied. Do not
restore a garage-local pressure law, material ledger, or interaction state to
work around that missing calibration.
