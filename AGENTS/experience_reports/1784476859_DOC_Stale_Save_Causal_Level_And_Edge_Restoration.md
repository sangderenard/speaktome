# Stale Save Causal Level And Edge Restoration

**Date:** 1784476859
**Title:** Derive placement from topology and restore physical pipes

## Overview

Investigated forward leaves appearing to remain in their old ring after an
ancestor moved backward, along with concern that backward words were ordered
incorrectly by the graph auditor. A clean current-source live graph had no
illegal signed-level edges, stale directions, or direct backward-to-forward
crossings. The observed graph had been resumed from a save made before the
current reroot changes.

The actual persistence defect was that `restore_state()` trusted cached
level/depth/direction metadata and restored nodes without recreating runtime
`Edge` objects. Legal topology could therefore display with stale placement,
while old causal connections lacked physical pipes for pressure and material
transport.

## Steps Taken

- Explicitly stopped and cleared the stale live session.
- Restarted Flux Radar from current source and started a fresh graph with the
  saved settings.
- Verified the fresh graph had zero illegal level deltas, zero stale
  directions, and zero direct backward-to-forward crossings.
- Changed restoration to recompute signed levels, depths, and directions from
  the active seed and canonical causal adjacency.
- Made contradictory causal ranks and disconnected live topology hard restore
  errors; restoration never deletes an edge to conceal damage.
- Reconstructed runtime `Edge` objects from every reciprocal saved causal
  adjacency.
- Recovered prefix/postfix formation from monotonic node allocation so
  backward auditor scoring uses the word-bearing endpoint.
- Cleared traversal runtime caches before the auditor rebuilds them from the
  restored topology.
- Added regressions for stale descendant placement, contradictory topology,
  restored physical pipes, and backward reading order.

## Observed Behaviour

- A leaf attached to an ancestor that rebases from forward to backward now
  receives its own correspondingly shifted signed level during restoration.
- Every accepted restored causal edge advances exactly one signed level.
- A damaged shortcut that gives one node two contradictory causal distances
  is rejected wholesale and remains unmodified; no pruning occurs.
- Restored backward traversals enumerate farthest word to nearest word to seed,
  matching left-to-right reading order.
- Focused reroot, auditor, backward path, fluid persistence, Python compile,
  frontend parse, and diff checks pass.

## Lessons Learned

Signed level, radial depth, and display direction are derived anchor-relative
coordinates and must not be authoritative persistence fields. Canonical causal
adjacency is authoritative. Runtime pipes can be rebuilt from it, but an
internally contradictory topology must fail visibly rather than being repaired
by deleting connections.

## Next Steps

None.

## Prompt History

> "something is fully broken. leaves connected to a node that moves to backward stay in forward where they were instead of travelling down a level like everything is supposed to, this leads to long stretched implausible and out of convention edges. additionally it appears possible that the backward words are not actually being ordered correctly by the auditor, I think"

> "could be a stale save ccausing weirdness I didn't restart live after the changes"

> "you can't just prune illegal edges you have to not make them in the first place"
