# Documentation Report

**Date:** 1784436913
**Title:** Coupled region wedge binding and ray force exchange

## Overview

Replaced the radar's weak, one-way post-contact wedge containment with a
continuous regional angular binding force and equal-and-opposite node-to-ray
wall loading.

## Steps Taken

- Added stable ID-based angular slots inside each node's assigned wedge.
- Predicted wall contact from velocity and the node's visible angular radius.
- Applied inward tangential wall impulse before a node center crosses.
- Accumulated the opposite impulse onto shared internal rays.
- Capped ray deflection relative to neighboring wedge widths.
- Kept fixed forward/backward hemisphere dividers immovable.
- Prevented wall load from leaking across snapshots.

## Observed Behaviour

Inline JavaScript syntax, focused regional-binding regression, and
`git diff --check` pass.

## Lessons Learned

The previous comments described force exchange, but the implementation only
pushed nodes after they approached a boundary. Proper regional binding needs
a persistent interior tether, geometry-aware predictive contact, and a real
reaction path into the elastic boundary.

## Next Steps

None required for this change.

## Prompt History

- "regions don't properly bind their nodes with force exchange to stay inside their wedges"
