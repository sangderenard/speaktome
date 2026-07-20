# Growth Commitment, Habitats, And Branch Hardening

**Date:** 1784552497
**Title:** Conserved habitats and an emergent developmental growth loop

## Overview

Implemented the design audited in
`1784550818_AUDIT_Local_Growth_Commitment_And_Ion_Exchange.md`. FluxGraph now
has a coupled developmental loop rather than independent growth knobs:
sustained scarcity builds local directional commitment, committed growth
inherits that signal, useful material flow hardens physical branches, and a
reroot moves the organism into a finite seed-relative habitat associated with
the phrase present when that seed location was first entered.

## Steps Taken

- Replaced ring contact's exact opposite-ion-to-own-ion transmutation with a
  conserved transfer from a finite backend-owned habitat inventory.
- Added one persistent habitat per seed node. A new seed discovers a new
  patch; revisiting an old seed revisits its depleted inventory.
- Captured the best complete token sequence at first habitat entry as the
  phrase-space habitat signature.
- Reinterpreted the existing directional growth-interest fields as bounded
  commitment state with retention, scarcity production, local same-lineage
  diffusion, inheritance, and post-growth expenditure.
- Made above-threshold commitment outrank ordinary within-direction actions
  without bypassing explicit compute-budget constraints.
- Added slow physical-edge maturity driven only by useful named-material
  flow. Maturity raises bidirectional conductance but remains ordinary scalar
  graph state, not a per-edge learned parameter.
- Persisted habitats, phrase signatures, movement count, and edge maturity.
- Added radar configuration, node commitment telemetry, current habitat and
  movement telemetry, and visible edge hardening.
- Restarted the live radar server on port 8877 through its persistence-aware
  restart helper and verified `/api/engines`.

## Observed Behaviour

- Ring uptake debits and credits the same named ion; the node's opposite ion
  is unchanged and the node-plus-habitat amount is conserved.
- A new seed receives a new finite habitat. Returning to a previous seed does
  not refill the old patch.
- Scarcity raises commitment over time, nearby same-lineage tissue receives a
  smaller local signal, and newly grown tissue inherits a divided share.
- Crossing the configured threshold lets commitment beat an arbitrarily
  higher ordinary pressure action within the same directional budget.
- Useful ion flow raises edge maturity and mature edges conduct more strongly;
  water-only flow does not count as the resource-use hardening signal.
- Habitats, movement count, and branch maturity survive persistence and edge
  reconstruction.

## Lessons Learned

The organism does not need a separate movement action. Scarcity already causes
growth; growth creates new pressure candidates; rerooting promotes one of
those candidates to seed; and seed identity selects a new persistent resource
patch. This lets movement emerge from the existing language/pressure system
while making the whole evaluated phrase the recorded location signature.

Developmental memory also does not require a second storage subsystem. The
former nutrient-interest fields already had the correct directional locality;
adding bounded retention, diffusion, inheritance, and expenditure turned them
into a hormone-like commitment field without duplicating state.

Branch maturity belongs in physical state rather than optimizer state. The
fixed-size learned archetypes remain fixed-size as the graph expands.

## Verification

- Python compilation passed for the graph, server, and focused tests.
- Frontend JavaScript parsed successfully.
- `git diff --check` passed.
- 272 directly callable regressions passed across twelve relevant test
  modules; 25 fixture-dependent tests could not run through the repository's
  uninitialized pytest gate.
- Normal pytest remains blocked by the pre-existing environment bootstrap:
  `setup_env.ps1` contains POSIX heredoc syntax and `tests/conftest.py` skips
  the suite when setup fails.
- The restarted server answered `/api/engines` on port 8877.

## Next Steps

None.

## Prompt History

> "Could you look into the motivation and avoidance of growth, the 1:1 ion exchange habit, and see if we can do something like... confer advantage steadily over time to nodes until they all but can't resist growing there, locally. branch hardening, something like a hormone system. letss talk aboiut it and what we could do"

> "Go ahead and implement this, keep an eye out not to make pointless mechanisms but elegant emergent ones. the "organism" can achieve "movement" and thus new resources as it "moves" by the seed changing, the phrase changing that the whole network evaluates to"
