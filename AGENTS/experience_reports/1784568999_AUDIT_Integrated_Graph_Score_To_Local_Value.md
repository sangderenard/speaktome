# Integrated Graph Score To Local Value Audit

**Date:** 1784568999
**Title:** Remaining score-credit loop needed for smooth local value

## Overview

Audited whether FluxGraph's integrated traversal scores currently feed back
into local value, pressure, growth, physiology, branch hardening, and
rerooting. No simulation code was changed.

## Observed Behaviour

- `FluxNode.local_value` is still exactly `exp(local_evidence)`.
  `local_evidence` is the model score captured once when the node is created.
- The graph auditor builds each traversal's `mean_score` by averaging the
  historical local evidence of its constituent grown edges.
- `audit_edge_influence()` sums `exp(mean_score)` over traversals using an
  edge, but this integrated quantity is consumed by radar telemetry rather
  than pressure or growth.
- Physiology learning softmaxes traversal mean scores and rewards opening
  shared archetype gates on high-scoring routes. This changes later
  conductance and transport indirectly, but it does not return an integrated
  graph score to node intrinsic value.
- `_digest()` backs up the single maximum descendant `path_mean`.
  `_expansion_priority()` gives a candidate a bonus from a centerward
  neighbor's backed-up maximum. This is the only direct graph-score-to-growth
  path, and it is a hard maximum rather than a smooth integrated posterior.
- Settled pressure still begins from `found_bonus + local_value`; rerooting
  chooses the highest pressure. Integrated traversal support therefore
  reaches rerooting only indirectly through learned valves and the limited
  rollup bonus.
- Rerooting changes which node is score-free as anchor and restores the old
  anchor's historical evidence, but existing traversal scores are cached and
  not rebuilt. `_recompute_depths_from()` resets only the new root's
  cumulative evidence, leaving other cached cumulative/path means anchored to
  the previous focus.
- Branch maturity currently uses absolute named-ion flow. Oscillatory or
  irrelevant ion movement can therefore harden a pipe even when it does not
  relieve scarcity or contribute to survival.
- New habitat patches have persistent phrase identity, but each begins with
  the same configured ion amounts. Phrase identity selects the store; it does
  not yet shape resource composition.

## Lessons Learned

Mutating `local_evidence` with integrated score would be the wrong completion:
it would erase the distinction between honest model evidence and downstream
graph success and would create self-reinforcing double counting.

The clean completion is to retain immutable `local_evidence` and replace the
hard-max rollup with one separate soft continuation value. Complete
seed-spanning traversals should form a temperature-controlled posterior.
Branch-local conditional normalization should back their downstream score
advantage onto each node without rewarding central edges merely because
combinatorially many overlapping subpaths cross them. A slow EMA of this
backed-up advantage can supply smoothness.

Pressure can then use an explicit effective intrinsic value in log space:
local evidence plus a configurable fraction of downstream continuation
advantage. Expansion, auxin source strength, physiology credit, and rerooting
should all consume that same value rather than maintaining several partially
overlapping interpretations of success.

Physiology and maturity also need functional credit. The same traversal
posterior should be multiplied by observed scarcity reduction or useful
deficit-directed delivery. Rewarding an opening or absolute flow by itself
does not establish that the route helped the organism.

Before introducing this field, rerooting must rebuild anchor-relative
cumulative evidence, path means, and traversal scores. Otherwise a smooth
integrator would smoothly amplify stale values.

## Recommended Completion Order

1. Rebuild all anchor-relative score bookkeeping and traversal scores on
   reroot.
2. Restrict integration to maximal seed-spanning phrase traversals instead of
   every overlapping ancestor/descendant subpath.
3. Replace hard `rollup_mean` selection with a temperature-controlled,
   branch-normalized continuation-value backup and EMA.
4. Feed one effective intrinsic value into pressure; let expansion, auxin,
   and reroot inherit it through pressure rather than adding duplicate score
   bonuses.
5. Weight physiology and branch hardening by deficit relief/useful delivery,
   not opening or absolute flow alone.
6. Optionally make a habitat's fixed total budget distribute
   deterministically from its phrase signature, giving movement different
   resource composition without allowing movement to manufacture more total
   matter.

## Next Steps

None committed pending discussion with the user.

## Prompt History

> "what remains to complete the elegance, are we fully tying the integrated scores of the graph into interpreting local value? we want this really smooth and effective"
