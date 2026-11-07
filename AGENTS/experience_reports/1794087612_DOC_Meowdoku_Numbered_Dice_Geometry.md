# Meowdoku numbered dice geometry handoff

**Date:** 2026-11-07

## Scope

Isolated Turing worktree `C:\dev\Powershell\turing-dice-polyhedra-20261107`, branch `codex/meowdoku-dice-polyhedra-20261107`, commits `5da3b1cf`, `99d57569`, `b0daa7cf`, and `111007d9`. The live site and canonical Turing main were not edited.

## Findings and work

- `diceSolid` constructs closed, outward-wound, triangulated d4/d6/d8/d10/d12/d20 shells in the existing `marble_mesh.js` face tuple format. The d10 is the polar dual of a pentagonal antiprism; the d12 is the polar dual of an icosahedron.
- A separate Pluck material ID selects the numeric marking's optical finish. Flat marks use surface material assignment; raised and recessed marks displace the closed shell vertices. The resulting actual triangulated volume changes the `marbleMass` mass, center of mass, and full inertia tensor. The relief consists of seven-segment glyphs with a grid-resolved edge; it is not a font outline.
- The d4 uses top-read vertex values repeated on all three incident faces, with each numeral oriented toward its own apex. The d10 uses 0–9, upper ring 0,8,2,6,4, with opposite faces summing to nine. Other even-faced dice use opposed faces summing to side count plus one. The 6 and 9 are underlined. [Chessex's catalog](https://dicecollector.com/docs/diceinfo_chessex_2015.pdf) documents top-read d4s, 0–9 d10s and underlined 6/9. [Torben Mogensen's dice paper](https://s3.amazonaws.com/arena-attachments/1815990/d73b087f51579750940b5e032152939e.pdf?1519703478=) describes the d4 threefold corner convention and common d10 pole order; it also notes that there is no universal arrangement for other polyhedral labels beyond conventions such as constant-sum opposites.

## Verification and limits

`node tests/meowdoku_dice_mesh_check.mjs` passes all six shapes. It checks directed edge closure/winding, face labels and opposite pairs, separate mark material IDs, positive mass/inertia, and recessed < flat < raised integrated mass.

The marking material is an optical surface finish. Raised/recessed volume is body substrate with body density. A separate-density solid inlay or finite paint film is not modeled. The current marble roll contact law is spherical and must remain disabled for these polyhedra until polyhedral contact physics exists.

## Next Steps

Cherry-pick all four commits into the marble studio branch; add shape and marking controls, export `dice_mesh.js`, visually inspect marking legibility at each shape, and keep dice roll disabled while preserving real static mechanical properties. If the designer later supports a different marking bulk density, construct a closed marking material domain and integrate it separately.

## Prompt History

> "on the marble, make the shape selectable would you, and we'll give it all the classic dice polyhedron too and the knowledge of how to mark them, give the markings properties of relief or raised, what material properties the lettering has or if it's flat dye, that would make our little marble maker a pretty amazing kinda tool"

> "Work on new user request in parallel with marble_game agent, but do not edit marble_game agent's working files. Inspect repo guidance and existing marble designer mesh/physics architecture. Implement in your own isolated Turing worktree/branch a reusable polyhedron topology and face numbering/marking geometry module for classic d4/d6/d8/d10/d12/d20, with physically meaningful flat dye/recessed/raised marking mesh integration if feasible. Coordinate directly with marble_game agent about interface and cherry-pick. Do focused tests, no Wasm compile. Preserve user WIP, do not touch live main. Report commits and limitations."

> "Marble agent has current coherent studio through 2584c84a and is handling exporter. Coordinate proposed interface with them. Important: current roll integrator assumes spherical contact; dice roll must be gated until genuine polyhedron support/contact physics. Relief/recess marking should affect mesh and mass, not just shader if called mechanical. Focus on isolated topology/marking module and tests."

> "Please include marking material ID per marking so Studio can use Pluck Phong material table (including color/specular/roughness/etc) separately from body. Marked face conventions should be documented and testable, including d10 numbering and d4 convention. Coordinate actual module API with marble agent."
