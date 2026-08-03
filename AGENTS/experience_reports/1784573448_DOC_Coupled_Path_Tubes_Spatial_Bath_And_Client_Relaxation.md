# Coupled Path Tubes, Spatial Bath, And Client Relaxation

**Date:** 1784573448
**Title:** Replacing endpoint transport and score pressure with one physical fluid system

## Overview

Corrected the CRT fluid ontology so every audited traversal owns two real,
continuous, persistent path tubes. Replaced tick-time score-driven pressure
with a vectorized PyTorch compartment relaxation over node casings, ordered
tube lumen segments, larger per-edge hulls, and graph-shaped passive
CSF/lymph. Live radar sessions now delegate that relaxation to the browser
and wait for a conservation-checked result before biological growth proceeds.

## Implementation

- Each `SubEdge` now stores one lumen compartment per crossed physical edge,
  ordered in its own direction. Forward and reverse tubes have independent
  solvent, named-ion, and pressure state.
- Each physical `Edge` now stores a separate outer-hull mixture and pressure.
  Hull/node valves do not alter the inner traversal tubes.
- `bath_by_node` makes CSF/lymph spatial. Bath sections connect passively
  along graph edges and participate in the same solver. Burns, heart spills,
  heart CSF exchange, lymph return, factories, and the rhizome now use local
  bath compartments.
- Hearts draw from and discharge into the lumen segment adjacent to their
  own traversal terminal. They no longer drain or fill a remote endpoint.
- The vectorized Torch solver marshals all compartments and sparse arcs,
  computes hydration/osmotic pressure, limits simultaneous source demand
  with scatter reductions, advects mixture, and diffuses each named ion.
- Bulk-current and ion-diffusion rates are independent. The default ion
  dispersion rate is faster than bulk current, and either may be zero while
  the other remains active.
- Score no longer enters `_update_pressures`, `_settle_circuit`, or valve
  state features. It remains a soft traversal reward and reproduction/growth
  signal. Physiology reward is multiplied by observed useful delivery.
- Branch maturity now responds to terminal scarcity relief rather than
  absolute ion oscillation.
- Rerooting no longer zeroes a promoted seed node's local or cumulative
  evidence.
- Fluid, hull, spatial-bath, and lumen state are persisted and exposed in
  radar snapshots.
- Live sessions publish an immutable fluid work packet, block the tick, and
  accept only the matching client result. The server validates tensor shape,
  finiteness, non-negativity, and per-component conservation before advancing.
  The browser animates the actual relaxation substeps, including changing
  node pressure/volume and edge exchange, and reports the conservation proof.
- Removed obsolete score-pressure controls from the radar UI and added
  explicit controls for visible substeps, bulk current, ion dispersion, and
  osmotic pressure.

## Verification

- Python compilation passed for `flux_graph.py` and `flux_radar_server.py`.
- Both inline radar scripts parsed successfully with Node.
- Directly callable tests passed after updating obsolete score-pressure and
  global-bath expectations and adding tube, independent diffusion, client
  delegation, and handshake coverage.
- Ordinary pytest remains blocked by the repository's pre-existing
  `setup_env.ps1` POSIX-heredoc parse error and environment guard.

## Lessons Learned

The prior data model was closer to the intended anatomy than the transport
algorithm: a traversal already had unique directional `SubEdge` identities,
but transport treated them as direct endpoint connections. Giving those same
objects ordered lumen state preserved the auditor's semantics while fixing
the physics.

Score is most coherent as reproductive credit. Allowing it into pressure
creates matter/force from evaluation and collapses the distinction between
language reward and survival. Physical usefulness can gate how reward teaches
physiology without turning reward into pressure.

Client-owned relaxation is a natural synchronization boundary. A frozen work
packet lets the browser's visible animation be the computation itself, while
server-side conservation validation prevents a malformed result from
advancing the organism.

## Next Steps

No required implementation step remains for this pass. A future visual pass
could render every individual traversal lumen as a separately inspectable
strand inside its edge hull; current live animation aggregates their real
substep flows onto the containing edge while snapshots retain every strand's
full state.

## Prompt History

> "maybe but stop, you're reducing. we have the CSF/lymph which is the graph, you see, but separate from the circulation it's a bath. The nodesSHOULD be moving ions and water by pressure to neighbors, it's supposed to be tied to a pressure solution to the full set of lymph, circulation, node localities, and node connectors passive circulation
>
> Think, veins and arteries link regions they don't make end to end trips, they have a connectivity inherently measured by each individual small path. that whole system is inside a system of larger tubes that contain them, and can valve their connections to nodes without impacting the smaller pipes
>
> this is how it must be so if it's not this that's a problem
>
> then outside ALL that, like think of it as interstitial space, there is the ambient bath, csf/lymph and it is as connected as the graph, but fully passive and unvalved, but still it's part of the pressure solution, the lymph sections, all smaller pipes, the larger pipes, the nodes, all participate in one ion and fluid sim with various exchangers in particular places
>
> do you see?"

> "you do not seem to understand there are individual tubes that run start to finish for every single audited path?"

> "score can be pure reward and even reproduction signal, lets move forward correcting the system and solver, remember to work in vectorized torch"

> "did you keep independent ion exchange speed? is it not true that ions disperse faster than current?"

> "also big ask but, I would really like if we could force the client side to crunch the relaxation, then signal back it's ready for the next step, putting users in charge of the effort and having a kind of proof of work in the end, and the user could see all exchange playing out naturally"
