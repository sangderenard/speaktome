# Local Growth Commitment And Ion Exchange Audit

**Date:** 1784550818
**Title:** Local growth commitment, branch hardening, and conserved ring exchange

## Overview

Audited the current FluxGraph mechanisms that motivate or inhibit growth,
with particular attention to the exact 1:1 ring-ion exchange rule and the
possibility of a local hormone-like system that makes sustained growth at one
site progressively more likely. No simulation code was changed.

## Steps Taken

- Read the repository guidance and the recent FluxGraph experience-report
  lineage covering growth, unique ions, osmotic transport, hearts, rhizomes,
  and physiology archetypes.
- Traced tick ordering, nutrient growth interest, expansion priority, auxin
  diffusion, heart structural stress, ring ingestion, pipe osmosis, seed
  reservoirs, CSF, and soil transport in `speaktome/core/flux_graph.py`.
- Compared the implemented behavior with focused regressions in
  `tests/test_flux_graph.py`.
- Separated three concepts that are currently partially conflated:
  material scarcity, structural commitment to grow, and mature-branch
  reinforcement.

## Observed Behaviour

- Directional node scarcity already integrates over time:
  `_update_nutrient_growth_interest(accumulate=True)` adds a fractional
  opposite-ion shortfall every tick. Adequate concentration clears the
  interest, but continuing scarcity leaves it unbounded.
- That accumulated interest is only an additive action-priority term. It does
  not diffuse locally, pass to new tissue, mature an edge, or reserve growth
  capacity after prolonged deprivation. It can therefore be crowded by other
  ever-increasing interests and by the finite directional compute budget.
- Ordinary expansion motivation is pressure plus neighborhood rollup, a
  square-root wait bonus, and optional directional balance. These are global
  scheduling signals rather than a local developmental memory.
- Existing auxin is purely inhibitory: a successful tip suppresses branch
  width in competing subtrees. It explicitly does not suppress itself. This
  is a useful antagonist for a positive growth hormone but is not itself the
  requested commitment mechanism.
- Missing heart sides are the one genuinely irresistible growth case:
  structural stress preempts normal frontier competition and immediately
  regrows each absent side.
- `_ingest_from_rings()` implements exact transmutation. A node consumes an
  amount of its held opposite-side ion and creates the same amount of its own
  side's ion. No ring inventory is debited and the paid ion is not deposited
  anywhere, so named-ion identity is not conserved even though the amount is
  1:1.
- Pipe osmosis is materially cleaner: every named ion retains identity,
  diffuses independently down its concentration gradient, and can counterflow
  with another species at zero net bulk current.

## Lessons Learned

The smallest coherent hormone design is a direction-specific local
`growth_commitment`, not another token score and not a renamed ion. It should
be produced by sustained post-transport scarcity, decay slowly, diffuse only
a few causal hops, and be inherited partially by the new branch it creates.
Its scheduling advantage should rise monotonically and eventually reserve a
directional growth slot above a high threshold, while still remaining
non-absolute below that threshold. Successful supply should stop production
and let the signal decay.

Existing auxin can remain the antagonist: commitment says "grow here";
ambient auxin says "avoid lateral branching because another tip is already
winning." Keeping the positive and negative signals distinct makes the
result inspectable.

Branch hardening should be a second, slower state. Sustained survival and
useful material flow can increase edge maturity, conductance, and perhaps
survival tolerance. It must not alter `local_evidence`, because model evidence
and physiological success are different facts. This closes a natural
feedback loop: scarcity produces commitment, commitment grows a route, useful
flow hardens it, restored supply removes the scarcity source, and commitment
then decays.

The 1:1 ring rule should be replaced by conserved transfer. A backend-owned
ring/environment reservoir should supply an ion without changing its name.
If exchange is desired, the node can export a different ion into a reservoir
as a separate conserved movement; it should never destroy one species and
create another implicitly. Charge-neutral or stoichiometric coupling can be
added later as an explicit pump/factory reaction.

## Next Steps

None committed. The implementation shape remains a design discussion with the
user before any simulation behavior is changed.

## Prompt History

> "Could you look into the motivation and avoidance of growth, the 1:1 ion exchange habit, and see if we can do something like... confer advantage steadily over time to nodes until they all but can't resist growing there, locally. branch hardening, something like a hormone system. letss talk aboiut it and what we could do"
