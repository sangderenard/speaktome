# Turing vehicle validator runtime and recursive compilation probe

Date: 2026-10-20

The combined pneumatic-tire and dually vehicle assembly validator is
`C:/dev/Powershell/turing/tools/run_vehicle_native_assembly.py` with the
`dually-axle` assembly profile.  A one-lane bounded fine-tire stage completed
and rendered correctly, but required roughly three minutes.  The default
eight-lane fine-tire run measured about 6.09 seconds per substep and was stopped
as a bounded diagnostic.

The strict vehicle whole-entry plan failed during topology reduction on
duplicate class identities for `STController`, `PlacementPayload`, and
`PlacementTransform`.

The existing `compile_project_bootstrap_creep` mechanism split the vehicle
source into 12 authored units.  Ten attempts completed: eight were partial and
two failed.  Those attempts preserved eight complete source-region SSA
artifacts, but installed no verified native product.  The material constructor
remained in source closure after 606 seconds and more than 1,400 discovered
work items; the root profile remained pending.  The command was deliberately
interrupted at that point.

Canonical detailed receipt:
`C:/dev/Powershell/turing/docs/compiler_convergence/24_vehicle_validator_creep_2026_10_20.md`

Durable compiler progress:
`C:/dev/Powershell/turing/build/vehicle_validator_project_creep_20261020/round_000/progress.json`

No compiler or simulator source was changed for this probe.

## Cross-trial addendum

Two isolated reruns showed that the vehicle failures are deterministic outside
project-creep scheduling.  `_outer_rate_hz` reaches a raw `call_record` rewrite
through `_BookCallList.__setitem__`; this indirect write was missing from the
bootstrap raw-writer inventory.  `visual_snapshot` maps ProcessGraph value 20
to the authored `source.items()` call and then raises `KeyError(20)` because
control lowering assumes a sequence payload column that was never published.
This is the same broad mapping/object-iteration surface as bootstrap units 081
and 115, but with a bare key error instead of a named concordance refusal.

Detailed receipt:
`C:/dev/Powershell/turing/docs/compiler_convergence/25_vehicle_bootstrap_cross_trials_2026_10_20.md`
