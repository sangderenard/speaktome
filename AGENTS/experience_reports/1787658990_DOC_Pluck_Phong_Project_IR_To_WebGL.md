# Pluck Phong Project IR To WebGL

**Date:** 2026-08-25
**Title:** Full Pluck base-material graphics chain through project shader IR

## Overview

Added a whole-shader route in the adjacent `turing` repository that ingests
raw GLSL line by line into a typed, target-neutral project shader IR before
deployment planning or target emission. Used it from `spectral-analyzer` to
compile the existing desktop OpenGL base-material vertex/Phong fragment chain
to WebGL 2 GLSL ES.

## Steps Taken

- Added project-IR records for every authored line, storage resources,
  interfaces, functions, calls, controls, identifiers, and source spans.
- Added a compiler-owned graphics/compute section planner. The live Phong
  helpers were all proven transitively fragment-bound; no compute pass was
  fabricated.
- Added WebGL lowering from project IR. Four read-only std430 scalar SSBOs are
  specialized to R32 texture feeds with a complete binding/packing manifest.
- Added a generic Turing CLI and a Pluck repository bridge that emits both
  graphics stages and their manifests.
- Added focused compiler and real-shader regressions plus a small WebGL link
  harness.

## Observed Behaviour

- Turing focused shader tests: 14 passed.
- Pluck real-chain tests: 2 passed.
- Both generated stages pass `glslangValidator` as GLSL ES.
- Chrome WebGL2 compiled and linked the paired stages successfully. The tested
  device reported 16 fragment texture units and 1024 fragment uniform vectors.

## Lessons Learned

WebGPU compute availability does not make a fragment-dependent game-lighting
chain a useful compute workload. This shader reads interpolants,
`gl_FrontFacing`, texture coordinates, and writes/discards fragments, so the
correct deployment is still a graphics pipeline. WebGL lacks SSBOs, but the
four read-only scalar blocks map cleanly to texture-backed indexed reads while
remaining inside the minimum 16-unit fragment texture budget.

## Next Steps

None required for this translation tranche. A future WebGPU graphics backend
can consume the same project IR and retain read-only storage buffers directly.

## Living data-map integration continuation

The existing living-map vertex ABI is position/normal/color plus a custom
camera projection, so the Pluck chain could not be substituted as an identical
vertex+fragment pair. Added a compiler-visible `ShaderProgramChoice` contract
and retained the existing living-map program as the default selection. The
exact compiler-emitted Pluck fragment now uses a small living-map vertex
adapter, palette-derived material records, four R32F storage textures, and
identity texture feeds for the optional Pluck texture stack.

The generated AbstractUI object map now exposes `Living map` and `Pluck Phong`
in a viewport selector. Chrome loaded the default, switched to Pluck, and
switched back successfully; each selection reported its active WebGL2 backend.

In the next continuation, Pluck Phong became the preferred default (with the
Living map retained as a selectable and missing-Pluck fallback). Added a cold
JSON bridge over Pluck's `MaterialDatabase`: all 47 YAML-authored materials are
registered and exported using the exact `pbr`, `phong_compat`, `enamel`, and
`texture_stack` packed tensor rows consumed by the shader. AbstractUI carries
that neutral catalog in its model and selects the nearest authored material
for every living-map palette color before texture upload. Chrome confirmed
`WebGL2 · Pluck Phong` at startup, the 47-record catalog in the page model, and
successful switching back to Living map. The focused AbstractUI suite passed
38 tests and the combined project-IR suite passed 44; the Pluck bridge and
shader suites passed 3 tests.

## World-player auto-location continuation

Added a compiler-published navigation registry with mutable per-entity kernel
assignments and a complete freestanding WebAssembly A* kernel. The kernel owns
the 8-way/octile route search behind a caller-owned grid/path pointer ABI. The
browser host rasterizes authored walls and openings over the local document
geometry domain, requests a route, line-of-sight simplifies it, accepts
Catmull–Rom samples only when they remain obstacle-clear, and advances pose
orientation with shortest-arc quaternion slerp. Manual player movement cancels
the player's active route without changing its kernel assignment.

The generated page now spawns one `player.local` world entity instead of the
mouse-bound pointer plus four page followers. Mouse state is input/targeting
state only. The player's game-world pose drives both the first-person camera
and a sprite projected into the top-down div map. The live in-app browser test
routed a semantic room click through the real assembly kernel (44 grid cells,
3 obstacle-aware waypoints), observed quaternion and progress telemetry, and
confirmed the top-down player sprite and world camera moved together. Focused
navigation/div-map/entity/viewport tests passed 57 tests.

## Topological traversal and projectile-entity continuation

Separated path cost from the nonlinear hierarchy presentation. The navigation
host now constructs an invertible piecewise-linear chart from world bounds and
opening landmarks, compresses only long hierarchy gaps, searches and propels
the actor at 5.2 units/second in that chart, and projects every sample back to
world space. A continuous swept-clearance audit covers curve samples and the
segments between them. Unsafe A* cell transitions are fed back into the
occupancy grid and searched again; unsafe curves fall back to a certified
polyline, while genuinely unreachable targets are refused.

Physics balls are now full `physics-ball-entity` mezzanine records linked to
the compiled projectile-physics controller. Their published data-world poses
drive smaller top-down DOM markers. Expiry removes active render geometry but
retains the entity, last pose, card, and spent marker until explicit history
cleanup. Live browser verification produced an 80-cell/5-waypoint route with
`collision-certified=true`, observed a ball marker moving with its compiled
physics pose, and the focused navigation/projectile/div-map/entity/viewport
suite passed 62 tests.

## Visible routes and exact document/world frames

Added an SVG route layer inside the top-down map. Each active entity route owns
a base spline and a progress stroke; the latter advances by path distance and
changes hue continuously. Manual movement, route replacement, kernel
reassignment, invalid state, and arrival remove the owned overlay.

Replaced viewport-wide marker projection with identity-paired structural
frames. Rendered region/building/room border boxes are normalized into the map
root's local coordinate system and paired with their data-world box corners.
The smallest containing frame supplies forward and inverse piecewise-affine
mapping for entity markers, route samples, and background clicks. Viewport
translation cancels out, so scrolling cannot alter the mapping; resize and
structural world revisions trigger deterministic resynchronization. Live
browser verification observed an obstacle-certified route progress from
0.0593 to 0.0964 while its stroke color changed, kept the synchronization
revision stable, and confirmed the entity layer is locally owned by the map.

## Run and jump controls

Extended the neutral viewport-control policy with `run` and `jump`. Left and
right Shift select a 2.0 manual movement multiplier. Space edge-triggers a
grounded-only 3.6 m/s vertical launch, cancels auto-location, and forwards the
velocity to the dedicated compiled-physics worker through an explicit impulse
message (with the in-thread WASM fallback retaining the same behavior). Space
is consumed only while game focus owns the keyboard. The live device monitor
confirmed both Shift sources as `run` and Space as `jump`; the focused control,
dynamics, worker, and div-map suite passed 57 tests.

Added a visible `Return to defaults` button to the placement panel. It deletes
only the current world's hashed persistence cookie and corresponding
local-storage fallback, clears the dirty identity set, and reloads so compiled
geometry, appearance, stock, and physics defaults are reconstructed. The live
page exposed exactly one accessible reset control; it was not activated during
verification, preserving the user's current saved edits.

## Navigation/graphics thread separation

Diagnosed a false asynchronous boundary: `locateEntity` returned a Promise but
still performed grid rasterization, repeated unsafe-edge A* searches, spline
construction, and continuous clearance sampling synchronously on the graphics
thread. Added a dedicated navigation worker which instantiates the same WASM
assembly kernels and owns that entire planning pipeline. The main thread now
receives only certified route samples through structured clone and performs
installation/interpolation. Removed unnecessary scene-mesh rebuilds from route
start, cancellation, invalidation, and arrival. Live verification observed the
`planning` state while the page remained responsive, followed by an 80-cell,
5-waypoint certified route carrying `planner-thread=dedicated-worker`.

Corrected only the webpage projection of route samples: context frames now
share parent/child corner landmarks, and the SVG recursively subdivides each
transformed segment to capture nonlinear bends. The navigation worker and its
route result were unchanged. Regression verification returned the same known
80-cell, 5-waypoint, collision-certified route while producing 177 adaptive
display points.

Reduced the living map's source-DOM burden by replacing the thousands of
self-script line cards with semantic location-scoped closure openings. The
source district mounts the current region/building/room opening plus openings
for closures the player has not entered; entered non-current closures are
unmounted, and the set changes only at containment transitions. The registered
action table is now a closed disclosure. Browser verification began with 18
mounted openings (one current), then auto-located through the unchanged known
80-cell/5-waypoint route to `packets`, where the source set became 16 openings
with region, building, and room recorded as entered. The focused suite passed
45 tests.

Added per-entity click waypoint queues. Each arrival now enters a minimum
0.85-second presence pause, emits `abstract-ui:navigation-presence`, and waits
for promises returned by `abstractUINavigation.onPresence` hooks before
planning the next queued segment. Manual input/cancellation clears the route,
pause, and queue. Live testing also exposed the older apparent "unclickable
object" bug: semantic clicks always discarded their exact coordinate and used
the geometry center, while some centers were unreachable under coarse doorway
alignment. Endpoint planning now tries the exact click, center, and deterministic
inside/outside opening stand-offs until it finds a collision-certified route.
The formerly failing `identity` room produced a 93-cell, 7-waypoint certified
route. Two rapid distinct room clicks reached a first presence pause with queue
depth 2/one pending, then completed the second leg with queue depth zero. The
focused suite passed 46 tests.

## Exact rollback of DOM-owned spatial layout

Reversed the recorded DOM/world-layout patches beginning at 21:00:56 while
retaining the placement tool, mesh transforms, half-dome skybox, and celestial
Phong lighting. Restored the original CSS grid, authored `gridStyle` placement,
and explicit region/building/room grid templates. Removed all spatial-item
classes, percentage injection, context compaction, spatial min-size/aspect
mutation, and calls that rewrote the div layout from world geometry.

Three forced fresh loads produced identical room and player coordinates with
zero spatially injected nodes. The sky and placement UI remained live. A real
click to `identity` produced the established 93-cell/7-waypoint certified route
and arrived with the player marker inside the destination div. Focused tests:
49 passed.

## Prompt History

> "I want you to try to cook my phong shader chain from spectral analyzer - the one that's like... typical game shit - I want you to try to use the compiler directly on it or the text that becomes it, with webgl as the output, unless webgpu is better fragment shaders as well as compute shaders, in whcih case we need to break the precedent that we use gl for shaders that aren't compute"

> "work on the translator until it can ingest the glsl line by line through the source interpreter and output webgl"

> "the compiler might be best left to determine what sections deserve compute shaders and what has to be a texture shader"

> "after raw input of the shader program into project ir"

> "the ui is broken can you just give me a link, also, please check if the shader we just used can step right into the living data map as it's shader - if possible - can we keep the one it has and make the shaders selectable"

> "pluck phong works brilliantly make it the default, start working on getting abstractui access to the material database that goes with this shader if you could"

> "we need a pathing algorithm for the top down map to use for auto locate. when people look at the div structure that is the top down map, if they click somewhere, I want a path to be calculated and the player actor to move along that quaternion spline avoiding obstacles to get to where they're trying to go, and I want the pathing algorithm a hot swappable assembly kernel so any entity can be given any pathing algorithm at any time"

> "this also will involve removing those page entities that follow, as well as the user entity will not longer be under the mouse, but keep locked with the game world, present in the top down div map"

> "I would prefer that in spline movement the player was propelled at greater speed, so like, the div map ends up being linear traversal geometry projected into the nonlinear stretch, then pathfinding is working on a simplified grid that stretches topologically. that's alittle complex so feel free to refuse. I'd like the physics balls to become entities as well, with their own smaller marker on the dom element map, so we can persistently see where they are. also I'm seeing the auto pathing seem to walk through walls, that's a problem"

> "let's have it show the spline it makes while the traversal is in progress until it is interrupted and changing color as it goes. also the world state of the objects in the game has to correlate exactly with the grid map, meaning, if the document renders those divs some way, that's where those corners are all meant to be relative to each other. that also means, when the game changes the location or appearance of something, it should not use positioning that gets grabbed by scrolling, it should use a deterministic resynching of the two coordinate spaces, with the only edit being the nonlinearity of context containers scale"

> "also I would like it if shift was run and space was jump"

> "also I need a return to defaults button somewhere that clears the cookie"

> "something has been compiled/rendered to block graphics instead of keeping it in it's own smooth thread"

> "path lines projected back onto the webpage are failing to use the nonlinear transform between the game map and the website map"

> "I want to be clear, the pathing was working perfectly, it was only the map display that was wrong, so I hope that nothing has been broken"

> "lets cut down on page burden and have the list of action registered things on the page collapsed, and hide every source line except the opening of the present closure containing the player and the openings of the closures the player has not entered yet, so the source displayed is the source precisely in scope of the user's location"

> "lets allow clicks for navigation to stack as waypoints, at each location pause a moment for game hooks for presence, then continue"

> "there is an issue, a bug, confounding our attempts to use the cache, where some items, no matter how you click, do not register it seems as navigable endpoints or are not submitted as candidates"
