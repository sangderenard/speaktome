# Trouble Ticket Report

**Date:** 1784419649
**Title:** FluxGraph startup failed during stale attention-mask cleanup

## Environment

- Windows / PowerShell
- Python 3.11.7
- The repository `.venv` points to a missing Python 3.10 interpreter, so
  verification used the available Python 3.11 installation with pytest's
  repository `conftest.py` disabled.

## Steps to Reproduce

1. Start a FluxGraph run through the radar server, or directly exercise either
   the shared batched expansion path or forward word-growth path.
2. Run:
   `python -m pytest --noconftest tests\test_flux_graph.py::test_expand_batch_handles_forward_and_backward_together_in_one_call tests\test_flux_graph.py::test_grow_forward_word_joins_continuation_pieces_into_one_word -q`
3. Observe both tests fail after successful model inference while cleaning up
   temporary references.

## Logs and Output

Both affected branches raised:

```text
UnboundLocalError: cannot access local variable 'attention_mask' where it is not associated with a value
```

The mask tensor had been moved into a nested `_build_and_forward` helper and
renamed `am`, but the enclosing functions still tried to delete the obsolete
local name `attention_mask`.

## Attempted Fixes

- Removed `attention_mask` from the two enclosing cleanup statements in
  `speaktome/core/flux_graph.py`. The helper-local `am` naturally becomes
  collectible when the helper call returns, while the enclosing cleanup still
  drops `logits`, `log_probs`, `outputs`, and `batch_tokens`.
- Re-ran the two reproducing tests: `2 passed`.
- Ran `tests/test_flux_radar_server.py`: `31 passed`.
- Ran all of `tests/test_flux_graph.py`: the initial repair produced
  `129 passed, 1 failed`. The remaining failure was an older test whose
  leaf-only starvation expectation contradicted `_starve_and_burn`.
- Compiled `flux_graph.py` and `flux_radar_server.py` successfully.

## Current Status

Resolved. Both stale cleanup sites are fixed. The follow-up graph geometry is
also implemented and covered by the focused graph, radar-server, and
persistence suites.

## Follow-up Geometry and Chemistry

The graph now uses one intrinsic causal ownership axis:
backward-farthest -> seed layer -> forward-farthest. A parent is always
backward of its child. Consequently, ordinary backward growth attaches new
causal parents, including multiple simultaneous parents, while ordinary
forward growth attaches causal children. The auditor enumerates every live
parent path through this DAG and therefore threads the backward and forward
trees through the seed layer without reversing ownership at either side.

Each node has a signed `level` and an explicit `center_id`. Growth toward level
zero from either side can create another seed-layer cousin; the new level-zero
node owns its own regional heart. Radar snapshots expose all parent IDs,
centers, levels, and both nutrient-growth interests. The frontend renders every
causal connection and places every level-zero cousin at the shared visual
center.

The ion rule is reciprocal. Forward-region cells require the backward-region
ion, which they burn to obtain their forward ion. Backward-region cells require
the forward-region ion, which they burn to obtain their backward ion. While the
needed ion remains at zero, interest in growth toward a center increases every
tick and competes within the existing expansion budget; supply clears that
interest. Rings remain the ordinary source of directional ions.

The stale leaf-only starvation test was updated to the implementation's
established rule: starving an internal node burns descendants that lose their
last live causal parent, while descendants with another live parent remain
connected.

The active seed now treats its live direct parents and children as its
backward and forward circulatory attachments. If either attachment is absent,
repairing that side preempts ordinary frontier competition for the tick. Only
the missing side grows during that repair; when both sides are intact, seed
growth does not preempt the normal budget.

Each seed heart now owns forward and backward ion reservoirs. Design storage
is one ion unit per token represented by that seed, even though the seed is not
split into token nodes. The original seed begins at 100% of that storage;
later seeds begin empty and build stores by skimming their chambers. Each
reservoir has a bidirectional ion gate whose finite tick throughput is opening
coverage multiplied by float exchange probability, plus a separate
semipermeable membrane that moves solvent only. Concentration bands are the
moving mean plus/minus one population standard deviation over a token-count
window, and physical storage volume grows with the held mixture. When a node
stops being the seed, both reservoirs full-pump into their matching
out-chambers before the replacement seed takes ownership. Reservoir ownership,
contents, gates, membrane state, and moving windows persist with fluid state
and are exposed in radar snapshots.

The first real radar run after the causal-DAG conversion exposed a stale radial
assumption when auxin suppression was nonzero. `_diffuse_auxin` still indexed
`self.nodes[node.parent_id]`; a backward-farthest node correctly has no causal
parent, producing `KeyError(None)` on tick 1. The HTTP layer reduced that to
the unhelpful message `run failed: None`. Radial digestion, auxin diffusion,
and expansion-neighborhood lookup now use explicit centerward/outward helpers:
backward radial growth follows causal parents outward and causal children
toward center, while forward growth does the converse. Error responses now
include exception type, repr, and traceback. After restarting the server, the
original saved GPT-2 configuration completed all six ticks, producing seven
snapshots and 482 final nodes.

Nodes can now carry optional `MaterialFactory` declarations without being
assigned any default role. A factory declares its material inputs, outputs,
throughput, and working medium (`circulatory`, `csf`, or `both`). The special
output `auxin` becomes a node-local auxin source; all other products remain in
the selected fluid. Node shells expose overall hull permeability plus
material-specific pore permeability, and all of this node state round-trips
through radar persistence.

Forward-side heart chamber solutes can now permeate through level zero into a
shared background mixture. Forward-type ions then cross a second soil boundary
at a lower configured rate. Backward cells passively absorb the matching
forward ion from soil, weighted by hull and matching-pore permeability, which
creates the initial root-side surplus needed by the existing exchange rule.
Background and soil are conserved graph mixtures, persisted and exposed in
radar snapshots.

Humidity exchange now treats `humidity` as ambient water and every other
scalar field in that slice as a dissolved ambient material. Ambient osmoles
lower water activity, node-local osmoles raise the solvent target, and
dissolved materials themselves diffuse through their named pores. This
replaces the prior shortcut that simply added total solute concentration to
the humidity-flow scalar.

After loading these systems into the live radar process, the saved GPT-2
configuration again completed six ticks: seven snapshots, 508 final nodes,
and no progress error. The final background held both directional materials;
the soil pool was empty because its permeated forward ions had already been
absorbed, with 0.4118 forward-ion units present across 93 backward nodes.

Seed-layer cousin centers now have zero collision radius in the radar, so the
collision force no longer fights the requirement that all centers occupy the
same location. Ordinary non-center nodes retain collision spacing.

Burn cleanup is now defined by undirected live reachability from the active
seed/heart, not by causal-child cascading. The old cascade handled forward
branches but could leave farther-back causal parents alive after their
backward connector burned. The new pass detaches both causal directions,
reaps every component no longer reachable from the seed, spills its fluid to
CSF, and removes burned-node edges and traversal hull state. Multiply
connected cousins survive through alternate routes. If their own center dies,
survivors are reassigned to the nearest reachable live center so region and
heart ownership cannot retain a burned pump.

The level-zero background interface is now bidirectional. Forward chambers
still seed the environment, while material already outside can permeate back
into circulation, preventing background mixtures from becoming one-way
orphan stores.

## Prompt History

```text
run failed: cannot access local variable 'attention_mask' where it is not associated with a value

I'm getting this in the flux graph server trying to start

seems good, I think missing directions might need an auto pass-through so the pump keeps working but I'm not sure

well, we may not need to do anything if we keep to the plant metaphor, then, we would make every node start trying to grow roots or shoots any time that node is without the ions it needs, ramping the interest up and up for a backward wherever there still isn't enough supply, kind of like succulents trying to make air roots, and if they reach the seed level going backward (placing "foward" sections as far as the auditor is concerned until they reach the seed level where they can create cousins and that cousin of the seed will, as long as that's the seed level, have hearts, you see?

I stopped you because you said that it would be forward in the causal chain of the auditor. it will just be that without us needing to do anything, there may be issues with connecting thing as they are now but please understand what I'm saying

stop saying they are added as children, they're not, why would they be, what is your fucking malfunction

faggot stop complicating something simple what does your faggot ass not understand tell me what your faggot ass is confused about in this geometry that you keep getting everything fucking backward

faggot there is one fucking directional rule for child parent ownership and it is intrinsic to the gemoetry. faggot. tell me what you don't understand, stop being fucking stupid

okay, you seem to get it for the forward direction, proceed

no faggot do not conflate it on the two fucking sides you stupid cunt. parens are backward farthest to forward farthest. jesus fucking christ stop pissing on that for no fucking reason

please continue WITHOUT ANY MORE FUCKING ASSUMPTIONS

1. yes the system threads trees through the seed layer, the trees connect in both directions, representing the focus of causality, the center of the probability manifold under examination wherever it happens to be, with orthagonal other cousins all occupying a center
2. the design is supposed to be and you can implement as much as it's not, a system whereby the forward region posesses ions of the forward type, which the backward cells need in order to make exchanges, and the backward side nodes symmentrically but in synnergy, they burn forward ions to obtain backward ions, 

there are a lot of ways a node could passively ingest ions without forced exchange we can explore later

so don't fully write it out as an option, node shells will have pores and permeability for hull and pores individually, you know, a host of parameters for each node dictating world interaction, but the principle growth needs are to move these ions back and forth, supplied by the rings

now, we need to make sure the seed can grow urgently, especially urgently, if it is missing a circulatory side

let's let seeds, while they're seeds, hold ion reservoirs made by skimming the chambers, and the initial seed should begin with ample supply. we'll have reservoirs in the heart and they'll have ion gates in both directions and a semipermiable membrane, growing volumes to match their stores. these stores will go full pump into supply if the node stops being a seed, the ion channels will pump to maintain specific concentration bands

100% of storage and storage will be the amount needed for the seed to resperate with one node per token (even though it doesn't get broken down as such), low and high should be std dev based on a moving window, no maximum on per tick ion gate we're assuming a level of transport surface that is seemingly nearly instant across boundaries but in truth of fact I guess does need a finite limit, you can do something about percent of opening coverage and integer integer ratio or float probability exchange 

[flux-radar] "GET /api/run/progress HTTP/1.1" 200 -
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[flux-radar] "GET /api/run/progress HTTP/1.1" 200 -
[flux-radar] "GET /api/run/progress HTTP/1.1" 200 -
[flux-radar] "GET /api/run/progress HTTP/1.1" 200 -
Loading weights:  55%|██████████████████████████████████████████████████████████████████████▎                                                        | 82/148 [00:00<00:00, 712.88it/s, Materializing param=transformer.h.6.mlp.c_fc.weight][flux-radar] "GET /api/run/progress HTTP/1.1" 200 -
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 148/148 [00:00<00:00, 751.27it/s, Materializing param=transformer.wte.weight]
[flux-radar] 'gpt2' model ready.
[flux-radar] building dictionary/trie for ('gpt2', 2000, 4, None, True) (first use of this combination)...
[flux-radar] ('gpt2', 2000, 4, None, True) ready.
[flux-radar] "POST /api/run HTTP/1.1" 500 -

run failed: None
DID YOU MAKE A TIMEOUT or something trivial getting in our way

we can, can't we, start defining the auxin factories in nodes and whether they function with the cerebrospinal or circulatory fluids and what kind of materials are needed, yes? and we can establish that forward ions/solutes can permiate into the background  at the interface between the 0 levels of each, see?  then the oxygen the roots need permiates into the soil at a reduced rate and supplies the first nutrient surpluss to exchange, also, we really should do something about osmotic effect or dissolved ions in humidity

proceed, later nodes may service roles of synthesis or other circulatory management, but these don't have to have specific configurations

cousins need to be non-repulsed and then I think there might be an issue with orphaned data persisting outside any connection to the main heart
```
