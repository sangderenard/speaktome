# Turing vehicle effect and iterator repairs

Canonical receipt:
`C:/dev/Powershell/turing/docs/compiler_convergence/26_vehicle_effect_and_iterator_repairs_2026_10_20.md`.

The vehicle examples exposed three general compiler defects.  Book-backed call
records were being rewritten through a list view without provenance; projected
union consumers assumed a sequence payload column their producer never
published; and product accounting treated effect-only `IndexedStore` regions
as incomplete because they had no copied output.

The fixes use the exact incumbent/decision cells, refuse missing producer
storage by name, and carry the source store-result identity through universal
address lowering onto the emitted `Store`.  Real vehicle methods now separate
effect publication from their remaining missing nested-object storage ABI.
Bootstrap unit 081 shows the same next frontier in compiler-object traversal.

Focused validation is 245 passing with six known baseline fixture failures.
The mapping concordance audit reports zero findings and zero unsourced facts.
