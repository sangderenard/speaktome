# Documentation Report

**Date:** 1784435509
**Title:** Flux radar hydraulic straightening without repeated layout kicks

## Overview

Removed the radar's repeated whole-graph settling kick and added a pipe
straightening force driven by the hydraulic power already exposed in each
edge snapshot.

## Steps Taken

- Traced live snapshot polling through `applySnapshot()`.
- Replaced unconditional `alpha(0.9)` reheating with snapshot-aware low heat.
- Initialized new nodes on their intended ring and wedge.
- Added lateral bend correction weighted by normalized `|flow * pressure_drop|`.
- Added damping and a focused source regression.

## Observed Behaviour

The inline JavaScript passes Node syntax validation. `git diff --check`
passes. The repository pytest entry point remains blocked by its pre-existing
environment gate and broken PowerShell setup parsing.

## Lessons Learned

Live polling was repeatedly submitting the same completed tick, so reheating
on every application created motion unrelated to graph physics. Hydraulic
straightening should act only at through-joints and only perpendicular to the
neighbour chord, avoiding an accidental second edge-length spring.

## Next Steps

None required for this change.

## Prompt History

- "there's a twitch to the user environment? the graph likes to jerk maybe to help settle itself? can we eliminate that but allow pressure in pipes to produce a straightening force? maybe calculate angular torque from flow rate from pressure differential?"
- "CAN YOU PLEASE GET ON FUCKING TASK"
