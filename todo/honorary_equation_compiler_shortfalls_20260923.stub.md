# Honorary equation compiler shortfalls

The 2026-09-23 audit in
`AGENTS/experience_reports/1790162947_AUDIT_Honorary_Equation_AbstractTensor_LLVM_Shortfalls.md`
identified these reusable compiler frontiers:

- add reversible, concordance-visible transport from arbitrary SymPy symbol
  names to legal Python identifiers in AbstractTensor materialization;
- require/link implementations for applied functions such as `rho(t)` and
  `p_s(T_s)` before they become planned-region feeds;
- define a graph-native adjoint contract for linked calls before differentiating
  opaque functions;
- add complex-number ingestion only with an explicit numerical substrate;
- lower `cross` only after a vector-axis/shape contract is present.

Do not replace declared-domain, recurrence, or vector-shape refusals with
guesses. Re-run through SymPy -> AbstractTensor source -> batch-shaped
whole-program SSA -> LLVM.
