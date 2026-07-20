# Heart Dual-Side Growth And Radar Containment

**Date:** 1784506648
**Title:** Independent heart-side scarcity and hard radar geometry

## Overview

Added structural growth stress to every seed-tier heart. A heart now measures
its own same-lineage connection on each side independently: missing forward
tissue drives ordinary sprouting and missing backward tissue drives rooting.
Both actions occur in the same growth pass when both sides are absent. Also
tightened the browser radar so node centers remain on their assigned depth ring
and node bodies remain inside their network wedges.

## Steps Taken

- Added `_heart_growth_stresses()` to distinguish same-lineage tissue from
  unrelated cousin connections.
- Made missing heart sides preempt ordinary frontier competition so corrective
  growth cannot be crowded out by nutrient-driven candidates.
- Preserved separate forward and backward stress values on heart nodes.
- Added regressions for dual stress, cousin isolation, and independent stress
  clearing.
- Prevented starvation from charging or burning nodes during their birth tick,
  before their first auditor/transport/heart/factory pass.
- Added a third `stddev` growth-selection mode. It covers occupied
  one-standard-deviation score bands first, then allocates remaining growth
  slots in proportion to band probability mass, sampling without replacement.
- Added annular-sector collision, hard ring/wedge projection, visible divider
  rays, and pressure reaction on movable internal dividers in the radar.
- Restarted the radar server on port 8877 with the current checkout.
- Ran 176 directly callable graph tests and 28 radar-server tests, Python
  compilation, and frontend JavaScript parsing.

## Observed Behaviour

- A heart with neither side connected reports both growth interests as `1.0`.
- A direct connection into a foreign cousin lineage does not satisfy the
  heart's corresponding side.
- Connecting one same-lineage side clears only that side's stress.
- Even with `burn_after_ticks = 1`, newborn growth survives its creation tick
  with zero starvation strikes and becomes eligible on the following tick.
- Standard-deviation selection preserves the model's original evidence while
  returning candidates from several score bands instead of only the peak.
- All directly runnable regression tests passed.
- The previous port-8877 process served an older handler without the current
  live-state route. The replacement serves the current API, but only saved
  controls existed; there was no persisted live simulation to resume.

## Lessons Learned

- Heart scarcity is structural and cannot be inferred from directional cell
  scarcity because heart nodes deliberately have no direction.
- Topological adjacency alone is insufficient after rerooting; network lineage
  must participate in deciding whether a heart actually owns tissue on a side.
- Corrective heart growth must not share a small candidate budget with ordinary
  frontier and cross-growth actions or it can remain indefinitely starved.
- Visual ring and wedge semantics require hard constraints after D3 integration;
  soft springs can only make an eventually plausible picture.

## Next Steps

None.

## Prompt History

> "okay we have multiple problems, the one bothering me right this second is we need PHYSICS CONTAINMENT OF NODES IN TEHIR NETWORK PIE SLICES. we need PHYSICS with COLLISION driving DIVIDORS one FIXED MOVEMENT DIMENSIONS so that things make their own space they need, right now I'm getting spaghetti dropped over the lines and I play with spring tension and length and then it slowly moves into the right networks kinda, i need to know whan I see a node on a ring  oin a pie slice it means what that implies.
>
>
> so on the version that's live now, why isn't there any sprouting happening in the normal forward?"

> "you misunderstand me. look at the current live sate. there is starvation for things that can be obtained by things that can be made for what is missing"

> "no faggot listen to me you stupid asshole. you tell me right fucking now what your stupid faggot self thinks is working about this. if there are no fucking sprouts faggot it should make them, stop fucking around being retarded and look at what is fucking happening. who the fuck cares about the starved net whatever cells. WHOE TGHE FUCK CARES. WHY ISN"T THE HEART GROWING FAGGOT NEW NORMAL NETWORK MATERIAL WHAT FUCKING SCAARCITY"

> "if a heart has no connection to either side of it's network, it must feel BOTH stresses, BOTH the need to "air root" which is just root for it and the need to prout"

> "are we checking for things to kill the moment after allowing new growth, not giving it a turn of a chance?"

> "what if instead of topk or topp we used something that just took a std dev sample, regularizing the data, as rich and representative in proportion so we can let the system find the right path instead of trying to drop the right path in place with rudimentary beam search"
