# Turing variadic tuple and provenance consumers

- Continue from
  `turing/docs/compiler_convergence/23_formal_consumer_provenance_continuation_2026_10_20.md`.
- Make return-slot, region-feed and display-leaf consumers read exact
  `executable_node` and producer rows before use.
- Lower declared variadic tuple displays through the resident sequence ABI;
  `(0, *x[1:])` requires clear, scalar append and slice append, not an executable
  `Starred` scalar instruction.
- Land tuple-valued mapping support only as report 22's complete general-N
  reducer/ABI/lookup/store/result-leaf feature.
- Do not run the user's full bootstrap until the focused generator regressions
  pass.
