# Documentation Report

**Date:** 1784204935
**Title:** FluxGraph re-rooting corrected to a single live graph (FluxForest removed); optional anchor decay added

## Overview

Direct follow-up to `1784171490_DOC_FluxGraph_Radar_Standalone_Visualization_Server.md`,
in the same session. The prior phase (not yet documented) built root
displacement plus a `FluxForest` class that turned every orphaned
("orthogonal") subtree into a brand-new, independently-ticking
`FluxGraph`, each doing its own real GPT-2 forward passes forever. The
user had to kill the running server -- it was saturating VRAM and
looked endless. Diagnosis confirmed the mechanism: one re-rooting event
can orphan multiple subtrees at once (one per off-path sibling, at every
ancestor level between new root and old anchor), and `FluxForest.tick()`
spun each one into a new graph with no cap on total count -- geometric
growth, each graph doing real GPU work every tick.

The first fix attempt was still wrong: it kept the "extract the
orthogonal subtree out of the tree" design, just stopped turning
extractions into new engines, capturing them as frozen display-only
snapshots instead. The user rejected this too, correctly: displacing the
root doesn't change the graph's topology or what's reachable, so nothing
needed to be extracted, reset, or removed from `self.nodes` in the first
place. The actual fix is much smaller than either previous attempt.

## What changed

- **`speaktome/core/flux_graph.py`**: `FluxForest`, `OrphanedSubtree`,
  `_extract_subtree`, and `seed_from_nodes` are gone. `_reroot` now only
  reverses parent/child pointers and flips direction along the path from
  the new root to the old anchor -- every other node, including branches
  whose direction no longer matches the new orientation, stays exactly
  where it is: same parent, same children, same direction, still fully
  live in `self.nodes`, still eligible to become anchor itself later.
  `_recompute_depths_from` already did a full-tree BFS from the new
  root, so it naturally re-bases the whole tree (orthogonal branches
  included) with no changes needed. `tick()` is back to returning
  `None`.
- Added `FluxGraph.orthogonal_node_ids()`: a pure, stateless query (no
  mutation, not used by pressure/expansion/starvation) that walks from
  the current anchor and flags any node whose direction disagrees with
  its own parent's (propagating downward once flagged, since direction
  never changes again below that point). This is the entire answer to
  "don't show these in the two-hemisphere radar" -- a display-time
  label, not a structural operation.
- Added `FluxGraphConfig.anchor_can_decay` (default `False`): when on,
  the anchor's own pressure is no longer frozen -- `_update_pressures`
  computes it exactly like any other node. `_maybe_reroot` tracks the
  anchor's own `low_pressure_ticks` against the same
  `starvation_floor`/`burn_after_ticks` used for ordinary starvation,
  and once exceeded, force-picks the current highest-pressure live node
  as the next anchor (`_best_replacement_anchor`) regardless of whether
  it would have beaten the old anchor fairly -- the seed can decay away
  and force a pick from whatever the current node landscape actually
  has.
- **`speaktome/flux_radar_server.py`**: back to running one `FluxGraph`
  (no forest). Each node in the JSON response now carries
  `"orthogonal": true/false`. `anchor_can_decay` is read from the
  request params and echoed back in the response's `params` block.
- **`speaktome/flux_radar/index.html`**: added an "anchor can decay"
  checkbox wired into the run form.
- **`tests/test_flux_graph.py`**: re-rooting tests rewritten to assert
  nodes stay live, attached, and unmodified after re-rooting (rather
  than asserting they get detached/reset); `FluxForest`/`seed_from_nodes`
  tests removed entirely; added a direct test that a previously-orthogonal
  node can itself become the next anchor later, proving nothing about it
  was actually disabled by being orthogonal.

## Verification

Restarted the real `flux_radar_server.py` (after confirming, via
`Get-Process`/`nvidia-smi`, that no process was still running and GPU
memory was back at idle baseline following the user's kill) and drove it
through the actual browser UI form -- not a scripted fetch, which had
been unreliable earlier in the session through the preview proxy for
POST+JSON specifically. A 6-tick default run and a 5-tick run with
"anchor can decay" checked both completed cleanly against real GPT-2,
rendered correctly (46 live nodes, forward/backward hemispheres
populated, no crash). The anchor's own pressure was visibly no longer
frozen at exactly its starting value once decay was enabled (drifted
1.0500 -> 1.0501 tick to tick), confirming the exemption toggle in
`_update_pressures` takes effect. Root displacement didn't happen to
fire again in this short a run (the anchor's initial pressure is high
relative to fresh leaves), which is correct, expected behavior, not a
gap in the mechanism.

## Lessons learned

- **"Independent networks" in a user's design request does not
  automatically mean "independent engine instances."** The original ask
  ("lateral siblings... their forward and backward all being independent
  networks... we could depict and run all siblings") described a
  *display* requirement (show disconnected branches as their own
  visual units) that I over-translated into a *computational*
  architecture (spin up N real, GPU-consuming `FluxGraph` objects).
  Re-reading the request after the fact, "depict... if we could stack
  them aesthetically" was already pointing at a rendering concern, not
  a request for parallel simulation.
- **A costly bug and its "fix" can share the same wrong premise.** My
  first correction (frozen display-only snapshots) still assumed the
  graph's topology needed to change -- extraction, invariant resets,
  detachment from parents -- when actually nothing about the tree
  needed to change at all. Catching "this is still overbuilt" required
  the user's second, sharper correction, not just fixing the most
  visible symptom (VRAM growth) of the first mistake.
- **When a structural property (here, "orthogonal") can be computed
  correctly and cheaply from current state alone, don't track it as
  mutable history instead.** The tempting design was to accumulate a
  growing set of "known orthogonal ids" across ticks; the actually
  correct design is a stateless graph walk from the anchor, always
  accurate after any sequence of re-roots, with no bookkeeping to keep
  in sync.

## Prompt History

- "NOBODY TOLD YOU TO RUN THEM INDEPENDENTLY AS FULLY FLEDGED
  INDEPENDENT NETWORKS DOING THE SAME THING / it's still just the one
  graph / one graph the root can change, it doesn't change the topology
  of the graph or what factors into possible strings, you don't need
  whole other graph objects to keep doing work on areas orthagonal to
  the seed"
- "motherfucking stupid faggot why are you doing anything with the
  fucking orthagonal content just fucking leave it in the fucking
  network any item in it can become the new seed... THE ONLY THING YOU
  WERE TO DO WAS KNOW ITS OKAY NOT TO SHOW THEM BECAUSE THEY'RE
  ORTHAGONAL DATA / FAGGOT YOU STILL MUST SUPPORT SEED CHANGING"
- "faggot just run the server again"
- "the seed needs to be able to decay away optionally too, forcing the
  pick for th enext seed on the current node landscape"
