# Engine Toy starter-garage integration

> **Superseded:** the bespoke `StarterGarage`, exhaust, lint, receiver,
> textile, door-state, and snapshot implementation described below was removed
> in the corrective pass recorded by
> `1790176257_DOC_Starter_Garage_Real_Machine_Correction.md`. This report is
> retained as failure history, not current architecture.

## Result

The externally authored starter-garage catalogue was treated as an intention
bundle and integrated through existing engine-toy mechanisms. The checked-in
authoring ledger contains exactly 162 inventory-entry types and 40 major
constituent records. Starting quantities expand into 314 independent,
pickable objects; grouping labels never make sockets, bits, clamps, textiles,
or other contents inseparable.

The production scene embeds the existing generic-cabinet washer and dryer
graphs rather than implementing substitute appliances. A persistent
`StarterGarage` state selects one of the exact 47 permitted fixed-model dryer
fault combinations once per stable world seed. Snapshot/restore retains that
selection and repairs. The fixed spare carton is catalogue state independent
of the selected faults.

The mandatory obstruction is owned by the garage's `ExhaustPassage`, not by
the dryer. Its deposit has retained mass, moisture, occupied geometry,
permeability, attachment strength, and location. Every connected machine is
intersected with the same pressure-loss curve; replacing or disconnecting the
dryer leaves it untouched. Cleaning transfers finite material into the finite
vacuum drum or lint bin and lowers resistance. The air-volume model now also
tracks water-vapour mass, allowing textile water to transfer to the actual
room air while a genuine outdoor route leaves that room inventory alone.

Door interaction distinguishes an immobilized opener from its release: the
wall button cannot move the jammed drive, the red cord uncouples the trolley,
and the healthy counterbalanced door can then be lifted manually. The meter
is one misplaced physical object in its pouch behind rags; discovering it does
not modify or reveal faults.

## Files

- `engine_toy/starter_garage_catalogue.py`
- `engine_toy/starter_garage.py`
- `engine_toy/STARTER_GARAGE_README.md`
- `engine_toy/examples/export_starter_garage.py`
- `engine_toy/hardware_data/starter_garage.json`
- `engine_toy/hardware_data/starter_garage.production.json`
- `engine_toy/tests/test_starter_garage.py`
- `engine_toy/air_volumes.py`

## Verification

- Starter-garage, cabinet, cabinet-autoclave, electrical-hardware,
  electrical-distribution, machine-package, machine-options, fluid-routing,
  fluid-circuit, and thermal-recuperator suites: 80 passed.
- Generated manifest: 162 inventory entries, 40 constituent records, and 581
  production-graph nodes.
- Fault-space audit: exactly 47 unique combinations, with one fault from each
  of one to three distinct families.
- Conservation tests cover lint transfer into finite receivers and laundry
  water transfer into finite room air.
- One pre-existing cffi `imp` deprecation warning remains.

## Prompt History

> investigate, please, the git patch and document just added to engine\_toy

> oh, well, okay did we make things then? your task is to fully scrub it all
> into our game, retaining all the handy reference details and concepts, all
> the same items. this was made by an agent with no direct access so think of
> it all like an intention

> Yes—and the clog belongs to the garage’s exhaust duct, not to a “broken
> dryer” flag. Anything routed through that passage encounters the same
> obstruction: the original dryer, a salvaged blower, or the enormous duct
> fan the player builds later.

The last prompt then supplied the complete twelve-section starter-garage
catalogue, the exact dryer fault-family table and 47-combination rule, the
fixed spares, meter discovery, opener release behavior, exhaust pressure law,
material-transfer requirements, and the distinction between outdoor bypass
and discharge into finite garage air. Those detailed rules are retained in
the generated manifest's `inventory`, `constituents`, and `authoring_rules`.

## Next Steps

The scene has production geometry, persistent gameplay state, conserved lint
and water, and a deterministic export. Electrical switch/contact dynamics and
structural failure of weak vent joints remain on their existing game-system
boundaries; no parallel solver was introduced to fake those mechanisms.
