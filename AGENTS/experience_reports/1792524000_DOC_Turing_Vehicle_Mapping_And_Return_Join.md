# Turing vehicle mapping and return-frame repairs

Canonical receipt:
`C:/dev/Powershell/turing/docs/compiler_convergence/28_vehicle_mapping_and_return_join_2026_10_20.md`.

The real `_PythonVehicleMaterial.visual_snapshot` unit is now fully linkable.
The compiler now carries exact aggregate field producers across source
partitions, treats mapping iterator views as projections of their mapping's
resident columns, and includes an open lexical tail when joining an exact
authored `None` arm to a multi-column physical return frame.

The strict receipt is 1 linkable, 0 partial, 0 failed, 0 blocked. Focused
regressions pass 4/4 and the complete precompile suite passes 132/132.

Cross-trials leave two precise frontiers. `_outer_rate_hz` is partial only at
the declared native `math.ceil` and `math.log2` boundaries. Bootstrap
`_result_layout` initially lacked the nested-row address rank on two region
formals; publishing the exact `ssa_call_rank=1` clears every source-region
shortfall. Its remaining root coordinator has six unexplained structural
formals along `Function.blocks -> BasicBlock.instrs -> Instr.args`, plus the
unsettled `next(iter(layouts))` result. Continue there, not with a full
bootstrap probe.
