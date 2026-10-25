# Add canonical rational/complex/precision composite tensor types

Eager substrate landed in the working tree on 2026-09-24. Remaining work is
exact common power-of-two rebalancing, direct compiler integration, and
removing the eager whole-tensor limit-inspection cost measured by
`../turing/tools/benchmark_rational_features.py`.

Use the audit in
`AGENTS/experience_reports/1790247483_AUDIT_Turing_Precision_Complex_Composition.md`.
The proposed design is in
`../turing/docs/RATIONAL_TENSOR_COMPOSITION_DESIGN.md`.

- Pin all seven non-empty unordered feature sets and their bidirectional
  promotion matrix.
- First repair the existing `Precision <op> ComplexPrecision` asymmetry.
- Add rational numerator/denominator ownership and denominator-one swallowing
  for non-rational operands.
- Preserve repeated division structurally: reciprocal swaps components and no
  quotient is evaluated before an explicit boundary.
- Add exact identity-proven cancellation before component products, with no
  value-equality guesses.
- Add conservative `RationalLimits` facts that separate structural validity,
  operation safety, and collapse safety. Refuse before unsafe arithmetic;
  never use NaN/infinity or post-operation `isfinite` recovery as status.
- Add exact common power-of-two rebalancing only under proven exponent bounds.
- Add the rational+precision, complex+rational, and
  complex+rational+precision concrete types without duplicating `Precision`
  limb arithmetic.
- Add direct eager and compiler boundary coverage for every composite type.
- Turn `tests/test_rational_compiler.py` from its strict expected failure into
  a passing direct-wrapper lowering test: two width-two CRP inputs require 16
  scalar components and the CRP result requires 8.
- Consume the normalized `NumericFeatureDescriptor` tables now published at
  topology entry; do not reparse wrapper names in later SSA stages.
- Generic numeric annotations now bind retained eager classes and inherited
  dunder methods. Proven same-type operations now specialize into the
  authoritative `_binary_same` dependency graph before deployment, without
  changing the heterogeneous-literal tensor classifier. Next expand composite
  parameter/result ABI paths and hand their Precision leaves to the existing
  limb transaction.
- Numeric class and limb facts now transfer from exact call actuals to callee
  formals through `source_value_class_concordance`; do not add a parallel AST
  name or node-attribute-only typing channel. Component/result path facts must
  use the same receipt authority when composite ABI expansion lands.
- Authored Precision widths are no longer capped by the automatic ladder.
  Keep automatic/tuned precision opt-in: four limbs remains the default policy
  bound, while an explicitly supplied `max_limbs` may be wider.
- Resolve or baseline the current complex compiler test receipt reporting
  zero `precision_div` lowerings where two are expected.
- Make the documented Nodus unsupported-operation fallback cover native
  complex values before benchmarking the full feature lattice on Nodus.
