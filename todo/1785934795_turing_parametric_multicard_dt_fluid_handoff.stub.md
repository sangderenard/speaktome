# Complete Turing parametric multi-card dt-system fluid compilation

Continue the work documented in
`AGENTS/experience_reports/1785934795_DOC_Turing_Parametric_Multicard_DT_Fluid_Handoff.md`.

The real 4x4 `VoxelMACFluid` + `dt_system` source now captures a 297-value
higher-order hierarchy and checkpoints successfully, but precompile SSA still
reports six loop-carried producer shortfalls: `363`, `368`, `373`, and three
instances of `6238`.

Required outcome:

1. Preserve distinct lexical initial/update SSA identities through hierarchy
   `leaves()`, `canonical_global()`, `LoopResult`, `IndexedStore`, and value
   redirects while retaining shared arena storage policy.
2. Add focused regressions for the `_cg_helmholtz_face` and
   `_cg_poisson_cc_rhs` alias patterns.
3. Produce zero precompile SSA shortfalls without disabling physics,
   validation, or solver loops.
4. Compile and compare one bounded native frame against Python.
5. Extend resident feedback to adaptive controller class state after numerical
   parity is established.

Resume with the 4x4 `--compile-only` command in the handoff and allow up to one
hour. Preserve ignored checkpoints and avoid broad staging in the shared dirty
Turing worktree.
