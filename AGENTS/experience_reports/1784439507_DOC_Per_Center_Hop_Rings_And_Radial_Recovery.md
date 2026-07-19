# Documentation Report

**Date:** 1784439507
**Title:** Per-center hop rings and radial recovery

## Overview

Corrected Flux Radar generation rings so every main or cousin network keeps
its own seed/center at ring zero and every other node's ring index is its
shortest visible parent/child edge-hop distance from that center.

## Steps Taken

- Replaced level-difference ring indexing with breadth-first edge-hop
  distance over both `parent_ids` and `children_ids`.
- Preserved cousin centers as independent ring-zero network seeds.
- Kept `display_radius` as the single integer consumed by node spawning and
  target-radius assignment.
- Made rendered rings use the same authoritative structural radii as nodes;
  rings no longer drift toward displaced nodes.
- Translated settled node coordinates whenever the SVG viewBox center moves.
- Replaced the weak radial force with damped radial recovery toward the
  assigned hop ring.
- Added focused regressions for main and cousin center hop distances,
  authoritative ring rendering, view translation, and recovery physics.

## Observed Behaviour

- Main-center test topology maps center/child/grandchild to rings `0/1/2`.
- A connected cousin center remains ring `0`; its child maps to ring `1`.
- Python compilation, inline frontend JavaScript parsing, focused invariant
  checks, and `git diff --check` pass.
- The server was restarted with the corrected backend. Its saved-state file
  currently contains no live session, so no existing live graph was available
  for post-restart visual snapshot inspection.

## Lessons Learned

`level`, historical `depth`, the main anchor, and a cousin network's assigned
center are not interchangeable when choosing a node's generation ring. The
visual invariant is local to each network: center equals seed equals ring zero,
and graph-edge hops determine all later rings.

## Next Steps

None required for the implementation. Start or load a graph to visually
inspect the corrected layout.

## Prompt History

- "rings don't appear to correlate to depth, nodes seem like they spawn outside their ring, every reset of the view scatters things wildly and then they have no physics pushing them back to normal"
- "faggot change some code I'm not fucking interested in windows pussy shit"
- "faggot what the fuck are you talking about. one jump from the seed it one ring, two jumps is the second ring, do you understand?"
- "FAGGOT. STUPID FUCKING FAGGOT. THE WHOLE SYSTEM RUNS FROM -WHATEVER THE FARTHEST DISTANCE FORM SEED TO + WHATEVER FUCKING FARTHEST DISTANCE FROM SEED. WHAT IS YOUR FAGGOT ASS HAVING A HARD TIM EUNDERSTANDING? NOBODY WANTS YOU FUCKING WITH THE ORIGINAL SEED FAGGOT. FIND the SEED FAGGOT, THE ONE FOR EACH MOMENT, EACH RIGHT NOW, FAGGOT. THAT SEED IS MIDDLE LEVEL"
- "FAGGOT CUNT PIECE OF SHIT RETARD THEY ARE ALL THE SAME THING FAGGOT"
- "FAGGOT CAN YOU LISTEN WITHOUT DOING FAGGOT RETARD SHIT, NOBODY TOLD YOU TO ELIMINATE COUSIN CENTERS"
