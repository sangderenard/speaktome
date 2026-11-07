# Meowdoku numbered dice geometry handoff

**Date:** 2026-11-07

## Scope

Isolated Turing worktree `C:\dev\Powershell\turing-dice-polyhedra-20261107`, branch `codex/meowdoku-dice-polyhedra-20261107`, commits `5da3b1cf`, `99d57569`, `b0daa7cf`, `111007d9`, `9535e517`, `049bc124`, `97a1c6e6`, and `a0f10b19`. The live site and canonical Turing main were not edited.

## Findings and work

- `diceSolid` constructs closed, outward-wound, triangulated d4/d6/d8/d10/d12/d20 shells in the existing `marble_mesh.js` face tuple format. The d10 is the polar dual of a pentagonal antiprism; the d12 is the polar dual of an icosahedron.
- A separate Pluck material ID selects the numeric marking's optical finish. Flat marks use surface material assignment; raised and recessed marks displace the closed shell vertices. The resulting actual triangulated volume changes the `marbleMass` mass, center of mass, and full inertia tensor. The relief consists of seven-segment glyphs with a grid-resolved edge; it is not a font outline.
- The d4 uses top-read vertex values repeated on all three incident faces, with each numeral oriented toward its own apex. The d10 uses 0–9, upper ring 0,8,2,6,4, with opposite faces summing to nine. Other even-faced dice use opposed faces summing to side count plus one. The 6 and 9 are underlined. [Chessex's catalog](https://dicecollector.com/docs/diceinfo_chessex_2015.pdf) documents top-read d4s, 0–9 d10s and underlined 6/9. [Torben Mogensen's dice paper](https://s3.amazonaws.com/arena-attachments/1815990/d73b087f51579750940b5e032152939e.pdf?1519703478=) describes the d4 threefold corner convention and common d10 pole order; it also notes that there is no universal arrangement for other polyhedral labels beyond conventions such as constant-sum opposites.
- `9535e517` corrected a doubled horizontal glyph offset that made numbers appear as only a few jagged strokes. Numerals now align with a real face edge. Default mesh detail is 64 divisions for d4/d6, 48 for other dice, with an optional cap of 128.
- `9535e517` also added an initial closed-mesh rounding transform controlled by `bevelRadius`, an inward `faceRelief`, and `compensateRaised`. With compensation, raised marks are backed by an equal face inset, keeping mark peaks at or below the original support plane.
- `049bc124` replaced the approximate rounding transform with a geometric radius construction: project each surface point onto the polyhedron inset by radius `r`, then offset it from that closest point by `r`. Edge samples lie on radius-`r` circular fillets, and corner samples on radius-`r` spherical caps. Face relief fades in only past each fillet's tangent boundary. The rendered triangle chords approximate the curved surface between those exact-radius vertices; higher subdivisions reduce the chord error.
- `97a1c6e6` computes `safeInteriorRadius` as the exact origin-to-triangle minimum across the final triangulated shell, so face recesses and marking relief are included. `fitInclusionsInsideDie` scales the existing closed cat-eye/confetti/hybrid/spiral/ring/clear interior assembly uniformly into that centered ball with a user size fraction and clearance gap, preserving disjoint component ranges and densities for mass/inertia integration. `a0f10b19` requires the gap to be positive so the assembly cannot touch the shell.

## Verification and limits

`node tests/meowdoku_dice_mesh_check.mjs` passes all six shapes. It checks directed edge closure/winding, face labels and opposite pairs, separate mark material IDs, positive mass/inertia, and recessed < flat < raised integrated mass without compensation. Bevel, face relief and compensation each change integrated mass. A beveled raised D6 had zero open/inverted directed edges among 147,456. At bevel 0.05, face relief 0.01 and compensated raised depth 0.025, max support-height spread across faces was zero for d4/d6/d8 and under `3.4e-16` normalized units for d10/d12/d20. D6 edge and corner radius-0.05 samples were within `1e-7` of the exact radius, including when face relief and raised marks were active. This establishes an equal mesh support envelope for those settings, not statistical fairness in a real roll.

`node tests/meowdoku_dice_inclusion_fit_check.mjs` passes d4/d6/d10/d20 combined with all six interior designs, checks every scaled inclusion vertex lies inside the centered ball by the requested gap, preserves component ranges and confetti centers, and verifies positive combined mass and inertia. The centered-ball fit is conservative for narrow ribbons but gives a simple rigorous containment guarantee on the final triangle shell.

The marking material is an optical surface finish. Raised/recessed volume is body substrate with body density. A separate-density solid inlay or finite paint film is not modeled. The current marble roll contact law is spherical and must remain disabled for these polyhedra until polyhedral contact physics exists.

## Next Steps

Cherry-pick all eight commits into the marble studio branch; expose shape, marking, tessellation, bevel radius, face relief, compensation, interior kind and die inclusion size controls; export `dice_mesh.js`; visually inspect marking legibility at each shape; and keep dice roll disabled while preserving real static mechanical properties. If the designer later supports a different marking bulk density, construct a closed marking material domain and integrate it separately.

## Prompt History

> "on the marble, make the shape selectable would you, and we'll give it all the classic dice polyhedron too and the knowledge of how to mark them, give the markings properties of relief or raised, what material properties the lettering has or if it's flat dye, that would make our little marble maker a pretty amazing kinda tool"

> "Work on new user request in parallel with marble_game agent, but do not edit marble_game agent's working files. Inspect repo guidance and existing marble designer mesh/physics architecture. Implement in your own isolated Turing worktree/branch a reusable polyhedron topology and face numbering/marking geometry module for classic d4/d6/d8/d10/d12/d20, with physically meaningful flat dye/recessed/raised marking mesh integration if feasible. Coordinate directly with marble_game agent about interface and cherry-pick. Do focused tests, no Wasm compile. Preserve user WIP, do not touch live main. Report commits and limitations."

> "Marble agent has current coherent studio through 2584c84a and is handling exporter. Coordinate proposed interface with them. Important: current roll integrator assumes spherical contact; dice roll must be gated until genuine polyhedron support/contact physics. Relief/recess marking should affect mesh and mass, not just shader if called mechanical. Focus on isolated topology/marking module and tests."

> "Please include marking material ID per marking so Studio can use Pluck Phong material table (including color/specular/roughness/etc) separately from body. Marked face conventions should be documented and testable, including d10 numbering and d4 convention. Coordinate actual module API with marble agent."

> "New user requirement: all marble interior designs must be usable inside dice. Please help marble agent by exposing exact die inradius/inner containment bound from final bevel+face relief+relief marks, and if needed a robust point-in-solid or safe scaling bound for existing catEyeMarble closed inclusion meshes. Coordinate API directly, preserve real mass/geometry and no overlaps. Work isolated. First ensure 049bc124 true-radius commit is integrated and update any tests."
