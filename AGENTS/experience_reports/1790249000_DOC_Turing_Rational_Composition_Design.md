# Turing rational composition design

**Date:** 2026-09-24
**Title:** Structural rational tensor family design

## Summary

Designed the proposed rational family for Turing's `AbstractTensor` numerical
substrate. The design is recorded in
`turing/docs/RATIONAL_TENSOR_COMPOSITION_DESIGN.md`.

The feasible first version preserves numerator/denominator structure and
delays division. It does not claim arbitrary-precision exact arithmetic,
because tensor elements are fixed-width and the repository has neither a
big-integer tensor substrate nor a backend-neutral tensor GCD.

The user clarified that the primary use is repeated division that fixed limb
precision cannot resolve. The design now treats rational as a lazy division
carrier: rational arithmetic never evaluates a quotient, reciprocal swaps
components, and explicit `quotient()`/`collapse()` are the only evaluation
boundaries. Precision protects numerator/denominator arithmetic while the
structural ratio preserves the unresolved division beyond the limb budget.
Exact, identity-proven factor cancellation is required before new products;
value equality is never guessed.

The user then required rational limits so exceptional IEEE values never become
the control mechanism. The design now separates structural validity,
operation safety, and collapse safety. `RationalLimits` carries conservative
outward-rounded component intervals, nonzero/finite proofs, dtype/exponent
limits, and provenance without adding another numerical tensor. Operations
prove bounds before component arithmetic, then try licensed cancellation and
exact common power-of-two rebalancing. Remaining unsafe work raises eagerly or
becomes a compiler shortfall before emission; it is never executed and
repaired after observing `NaN` or infinity.

The design defines explicit `Rational`, `RationalPrecision`,
`ComplexRational`, and `ComplexRationalPrecision` classes. Together with
native complex `AbstractTensor`, `Precision`, and `ComplexPrecision`, these
cover every non-empty unordered combination of rational, complex, and
precision features.

Binary promotion uses feature-set union and widest precision width. Rational
therefore swallows non-rational values as denominator-one values without
discarding complex or precision features. Complex rational values are pairs
of real rational coefficients, matching the existing `ComplexPrecision`
composition pattern.

The initial surface is basic arithmetic, integer powers, structural access,
and explicit collapse. Automatic GCD reduction, hashing, non-integer powers,
implicit tensor conversion, epsilon denominator repair, and backend-specific
rational machinery are excluded.

The compiler plan lowers outer complex algebra, then rational structure, then
the existing precision transaction. Each boundary must carry exact component
identity receipts; manually spelling component algebra is not accepted as
proof of direct wrapper support.

## Verification

Documentation-only change. No Turing source or test was modified or run for
this design step.

## Prompt History

> okay, now, using your deeper knowledge of industry practice, best practices, is a simple rational feasible? please start on the design deferring to what's best practice and then to what abstract tensor needs

> the intended use case is the repeated application of division inside a system that cannot be resolved by precision, if that changes anything. this is about automatic preservation of precision beyond limbs but i suppose that's all obvious

> add to the design limits, we're going to want to be able to know rational limits then we don't have to defer to nan or inf and our cancellations remain thoughtful
