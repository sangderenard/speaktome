# Turing vehicle class-fragment and constructor-formal repairs

Canonical receipt:
`C:/dev/Powershell/turing/docs/compiler_convergence/27_vehicle_class_fragment_and_constructor_formals_2026_10_20.md`.

Strict whole-vehicle planning exposed duplicate class-navigation records for
three canonical classes. Recursive pursuit had retained both a
declaration-only field shell and an executable view of the same source
`ClassDef`. Declaration shells now carry an explicit fragment receipt, and
map-IR ingestion joins only exact, concordant views. It still rejects two full
definitions of one identity.

The repaired vehicle plan completes topology reduction and publishes 125
units. Its resolved graph has 30 class objects and 30 unique identities.

The first compiled unit then exposed constructor parameters used only by field
stores as anonymous formals. Receiver-slot injection now publishes the exact
version-zero name/value pairs already present in the ProcessGraph identity
table. Real `_DuallyDTState.__init__` is fully linkable with zero concordance
findings and all six authored parameters accounted.

Focused regressions pass 8/8. The five-file gate is 259 passing / 8 failing;
six are the documented fixture faults from report 26, and two additional
shell-deployment fixtures were outside that earlier gate and were untouched.
