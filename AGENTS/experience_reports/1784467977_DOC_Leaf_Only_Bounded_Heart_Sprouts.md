# Leaf-Only Bounded Heart Sprouts

**Date:** 1784467977
**Title:** Prevent beam width from multiplying hearts

## Overview

Fixed a tick-one heart explosion in FluxGraph. A nutrient-driven opposite-direction growth action consumed one compute-budget slot but emitted the entire model beam at level zero. With a branch factor of 64, one selected node therefore created 64 cousin seeds and 64 hearts.

Independent nutrient sprouts are now restricted to actual growth tips and commit exactly one candidate. Ordinary outward beam growth keeps its configured width.

## Steps Taken

- Traced heart creation from `_new_growth_node` through nutrient-driven direction selection.
- Confirmed that a full opposite-direction beam reaching level zero created one heart per beam alternative.
- Restricted nutrient-driven action pools to `_expandable_nodes`, the graph's live leaf/tip set.
- Added an optional branch limit to direct forward and backward expansion.
- Applied `branch_limit=1` only to nutrient-driven independent sprouts.
- Added regressions for complete tick-one heart count, single-heart commitment, internal-node exclusion, and existing cousin-heart lifecycle behavior.
- Ran Python compilation, focused regressions, and `git diff --check`.

## Observed Behaviour

- With nutrient sprouting enabled, branch factor 5, and a two-action compute budget, tick one now has exactly three hearts: the main heart and two independently selected leaf sprouts.
- Heart count is bounded by selected sprout actions rather than model beam width.
- An internal node cannot launch an independent sprout even when its nutrient stress exceeds a leaf's.
- Ordinary beam expansion remains unchanged.

## Lessons Learned

A beam is a set of hypotheses for one growth decision, not a set of independently committed organisms. Topological promotion must happen after committing a sprout candidate, especially when reaching the seed tier creates persistent system state such as a heart.

## Next Steps

None.

## Prompt History

> "at tick 1 we have 65 hearts, that's absurd and no mechanism should've allowed that except in the case of independent growth from leaves"
