# Turing HTML interface containment graph

**Date:** 1787602339
**Title:** HTML as a first-class interface containment language

## Overview

Added the first bounded HTML-to-`ProcessGraph` frontend. HTML is treated as
an authored compiler language rather than a web-publishing wrapper. A page is
normalized into three language-neutral node names: `InterfaceRoot`,
`InterfaceContainer`, and `InterfaceContent`.

Containment follows the repository's dependency orientation: content points
to its containing element and outer elements point to the synthetic system
root. Thus the document root is the final graph consumer, analogous to the
root operation of a SymPy expression.

## Steps Taken

- Surveyed the SymPy/JavaScript `ProcessGraph` schema and role-vocabulary seam.
- Added `turing/src/compiler/html_process_graph.py`.
- Defined a small tag vocabulary for document, `div`, form, control, and
  resource elements.
- Preserved authored order through structural position tuples and explicit
  containment-edge ordinals.
- Added normalized graph-to-HTML emission.
- Added strict structural validation and explicit unknown-tag shortfalls.
- Added `turing/tests/test_html_process_graph.py` and architecture notes in
  `turing/docs/HTML_INTERFACE_GRAPH.md`.

## Observed Behaviour

- `python -m pytest tests/test_html_process_graph.py -q`: 6 passed.
- `python -m pytest tests/test_oop_language_translations.py tests/test_html_process_graph.py -q`:
  10 passed.
- `python -m py_compile` and `git diff --check` passed for the new files.
- The repository `.venv` could not start because its executable points to a
  removed Python 3.10 installation. The available system Python 3.11 already
  had the repository's required test dependencies, so it was used without
  installing or modifying packages.

## Archetype-library continuation

The interface work now also has a construction layer in
`turing/src/compiler/abstract_ui_archetypes.py`. This deliberately separates
an interpretive metaphor from an archetype: a metaphor projects an existing
program object into a representation space, while an archetype is a reusable
recipe whose instantiation edits the living program.

The initial `class-panel` recipe creates a box, nested labelled buttons, and
displays; places them in authored order; binds displays to `class.members` and
buttons to `class.methods`; publishes the resulting panel in a hierarchical
namespace; and infers a structural `IntelliType` from its capabilities, slots,
and bindings. Instantiation returns a revisioned edit receipt carrying actor,
location, added graph facts, published symbols, and recipe statements.

Context nodes are shared anchors rather than archetype-owned copies. Two
explicitly named panels can consequently bind the same class/member/method
nodes, while reuse of an occupied namespace path is rejected before graph
construction. The implementation accepts both Python classes and the
transportable `ClassSchema` representation.

- `python -m pytest tests/test_abstract_ui_archetypes.py tests/test_abstract_ui_introspection.py tests/test_abstract_ui_vocabulary.py -q`:
  18 passed.
- `python -m py_compile src/compiler/abstract_ui_archetypes.py tests/test_abstract_ui_archetypes.py`:
  passed.

## AbstractUI packets and self-inspecting div map

Added `turing/src/compiler/abstract_ui.py` as the deliberately small common
object convention. An `AbstractUI` value carries its neutral model plus
ordered, identified language packets. The first conventional packet set is
HTML structure, CSS presentation, JSON graph data, and JavaScript behavior;
packets carry dependency identities and can be extended with later backend
languages.

`turing/src/compiler/abstract_ui_div_map.py` projects the deterministic
introspection grid as nested world/region/building/room divs. The executing
JavaScript captures its own exact script text and creates identified source
line objects in the same onscreen inspector. A generated example maps the
`AbstractUI` class itself at
`turing/docs/generated/abstract_ui_object_map.html`.

The abstract event contract was kept to two facts: interaction type and
destination node identity. Individual nodes carry no callbacks or DOM event
names. One delegated, omnipotent browser event host translates click and
keyboard activation into that pair, resolves the destination through the
identity index, and dispatches it. Browser verification selected both the
`AbstractUI.interaction` room and executing JavaScript line 74 through that
host; both reached their correct inspector record and the console had no
errors.

- `python -m pytest tests/test_abstract_ui_div_map.py tests/test_abstract_ui_archetypes.py tests/test_abstract_ui_introspection.py tests/test_abstract_ui_vocabulary.py -q`:
  27 passed.
- Python compilation, `node --check`, and `git diff --check` passed.

## Entity-mezzanine continuation

Added `turing/src/compiler/abstract_ui_entities.py`. Entities now live in an
identified mezzanine beneath the AbstractUI system root and may be corralled
into named organizations governed by an isolated four-phase entity cycle:
control input, integration, conceptual interaction, and presentation.

The initial `pointer-being` archetype owns point geometry, an orb texture, and
shared capabilities but no controller assumption. `pointer.primary` and the
NPC `pointer.echo` are spawned through the same operation from that exact
archetype and enter the same `pointer-beings` organization. The first binds to
the neutral `mouse.primary` control-input ABI; the second binds to a damped
second-order controller which reads the first entity's completed control-phase
pose. `inline` and `worker` policies preserve identical reference semantics so
the complete cycle can later move across a thread boundary.

The div-map backend projects both entity records and identical sprite bodies.
Live browser verification moved the native actor to `(620, 260)`; the follower
advanced from approximately `(156, 66)` to `(609, 256)` over the observation
window, and its entity/controller/geometry/texture record remained inspectable.
No browser errors were reported.

- The combined focused AbstractUI suite now reports 35 passed.
- Python compilation and JavaScript syntax validation passed.

## System timer and action-edge table

Added `turing/src/compiler/abstract_ui_actions.py` and made the standard
system-root timer prelude mandatory for assembled AbstractUI JavaScript
documents. The neutral model connects the timer to the mezzanine action-edge
table with the explicit operation `update(actions)`.

Every projected interactive node registers a row containing source identity,
interaction type, destination identity, count, and recency state. The event
host issues an action record but never changes table presentation. On each
frame the root timer drains its action queue into `actionEdges.update(actions)`;
the table increments matching counts, lights recently issued edges, and
extinguishes them after the configured 800 ms window.

Live browser verification issued `inspect` on the
`AbstractUI.interaction` room. Its row incremented to one and acquired the
`recent` class, then lost that class after the timer window while retaining its
count. The emitted page registered 332 inspectable action rows and reported no
browser errors.

- The combined focused AbstractUI suite now reports 41 passed.
- Python compilation and emitted JavaScript syntax validation passed.

## Factory-archetype continuation

Added `turing/src/compiler/abstract_ui_factory.py`. A factory archetype now
accepts either a Python class or `ClassSchema`-shaped contract, captures ordered
fields and public methods without constructing the host class, merges explicit
factory defaults over class defaults, and leases a bounded heap when a live
factory is instantiated.

Each `FactoryRequest` overrides defaults for one dispense operation. Missing
and unknown fields are rejected. All live instances remain reachable through
the factory with `dispensed-instance` relations. Destruction releases a heap
slot and increments its generation, preventing stale instance identities from
aliasing later allocations.

Public class methods become an explicit factory broadcast surface. A broadcast
produces identified per-instance call records between the class method and
each currently reachable instance; the neutral factory does not execute host
methods or retain callbacks. This leaves JavaScript, C++, Java, and SSA
backends free to implement their own object and memory ABIs from the same
dispatch records.

- The combined focused AbstractUI suite now reports 47 passed.
- Python compilation and `git diff --check` passed.

## Primitive-archetype continuation

Added `turing/src/compiler/abstract_ui_primitives.py` with the first canonical
constructible set: `div`, `input`, `palette`, `bbox`, and `graph`. `div`,
`input`, and `graph` are immutable structural objects; palette and bbox are
facets carried by those objects. The canonical `input` spelling and a
source-safe `input_` alias are both available.

Palettes now type and layer all requested facets: foreground, background,
margins, radii, font, decoration, visibility, and locking. Unspecified values
inherit through overlays. Locking is enforced by structure/style/placement
methods rather than being left as a renderer hint. Bounding boxes retain an
explicit coordinate-space identity, and geometry operations refuse accidental
cross-space comparisons. Graph construction validates global object identity
and rejects dangling edge endpoints.

- The combined focused AbstractUI suite now reports 53 passed.
- Primitive Python compilation and `git diff --check` passed.

## Nonlinear hierarchy space and physics-program boundary (2026-08-25)

The living map now realizes containment with nonlinear conceptual distance.
The global representation envelope is a broad sky floor; class/namespace
regions are defensive courtyards; buildings and rooms remain locally legible
inside them. The courtyard wall is taller and thicker than contained building
walls, and leaving the global envelope issues the neutral
`switch-map-representation` action rather than extending one coordinate system
indefinitely.

The player embodiment was reduced to one quarter scale. Embodiment scale, eye
height, collision radius, and movement speed are published model facts rather
than duplicated JavaScript constants.

Added a reversible compiler specialization from authored object/surface
identities to deterministic, dense, 1-based `u32` IDs. Zero is reserved for a
missing identity. Mesh spans carry compact runtime IDs while string identities
remain authoritative for editing, persistence, introspection, and reverse
lookup. The generic JavaScript world utility accepts the explicit table and
derives an appearance-order table for older minimal world records.

The reserved world-physics lane now publishes selected-but-unbound stages for
identity specialization, static welding, broad phase, contacts, player
resolution, and pose publication. Its executable contract is an authored
SymPy equation set lowered through the canonical ProcessGraph and compiler SSA
to WebAssembly; it does not introduce a hand-written JavaScript solver.

Validation evidence:

- focused viewport/dynamics/entity/world/div-map suite: 54 passed after test
  expectation corrections;
- expanded compiler/AbstractUI gate: 131 passed and exposed one legacy-world
  utility compatibility failure;
- focused compatibility rerun after correction: 70 passed;
- regenerated `docs/generated/abstract_ui_object_map.html` JavaScript parsed
  successfully in Node (320093 bytes).

### Prompt History addition

> "I need the player to bea quarter their current height/size, and I need the
> scale of empty space to nonlinearly warp with heirarchy structure, so the
> outer envelope is quite large and forms the floor of a sky box."

> "to give you a look into the future, the physics block is going to be an
> equation set from sypmy reduced into a program by our compiler in web
> assembly"

## Lessons Learned

- `ProcessGraph` already has the correct orientation for an interface graph
  if a container is understood to depend upon its ordered contents.
- Exact HTML tags do not need to become canonical graph node types. They can
  remain vocabulary tokens with capabilities such as `structure`, `form`,
  `value`, and `action` on one neutral `InterfaceContainer` concept.
- Authored position, source identity, and edge ordinal are separate facts and
  should remain explicit.
- CSS, program-value bindings, and event routing must become additional typed
  dependency relations. Folding them into containment would destroy the
  authority/scope meaning of the containment tree.

## Next Steps

- Establish `AbstractUI` and its backend registry following `AbstractTensor`'s
  semantic-operation/backend implementation pattern.
- Make the prosaic annotation compiler and interactive world-map projection
  mandatory reference backends, preserving the same state/action identities
  and an accessible non-spatial route.
- Reduce the new 905-word executable `AbstractUI` intention boneyard into
  canonical existence/navigation/action subsets as real backend coverage is
  added.
- Extend the passing class-map prototype from Python classes and individual
  SSA `ClassEmission` records to correlated multi-class plans and real
  projection backends.
- Promote `LivingDocumentEdit` from a serializable sidecar to a typed DualIR
  member correlated with class navigation and reference tables.
- Add archetype action execution: resolve method bindings through compiler
  identities, validate arguments, then record the resulting program edit.
- Add library composition, recipe parameters, selector cardinality, and
  explicit replacement/migration operations without weakening namespace
  collision rules.
- Promote runtime-discovered JavaScript source lines into compiler-emitted
  AST/SSA nodes with real operand, control, member, and event edges.
- Define the event-host operation table independently of DOM event mechanics;
  backends should translate their native inputs into only interaction type and
  destination identity.
- Correlate entity-mezzanine revisions with living-document edit receipts and
  the DualIR namespace rather than leaving the projection model as their only
  common container.
- Add organization-local collision/broad-phase interaction without allowing
  graphics backends to mutate controller state.
- Replace the current DOM-sized complete action table with a virtualized view
  while retaining the full neutral edge table and stable row identities.
- Feed timer/action batches into living-document edit receipts and deterministic
  replay rather than treating recency as browser-only observation.
- Publish factories, heaps, dispensed instances, and destroy edits into the
  shared living-document namespace and correlate allocations to target ABIs.
- Lower factory broadcasts through the class emission plan so instance calls
  retain receiver and method identities in JavaScript, C++, and Java.
- Rebuild the current self-map's room, inspector, entity, source, and action
  districts as library archetypes composed solely from the canonical primitive
  set, then make the browser backend render primitives generically.
- Add palette authority/cascade identities so unlocking and scoped overrides
  become living-document edits rather than local object replacement.
- Design the CSS selector/rule/property vocabulary as graph relations over
  interface containers.
- Correlate form value/action containers to program class fields and methods.
- Attach the resulting interface graph to the typed dual-IR shell once its
  relationship with existing hierarchy/reference tables is fixed.

These are recorded in
`todo/1787602339_turing_css_and_interface_binding_graph.stub.md`.

## Prompt History

> "you're making this about web publishing, I'm saying we need it as the primary language in which we develop UI by making it a language the translator understands and can translate around with it as a graph with nodes from a basic set, a minimal architecture using form elements, divs, and that's about it"

> "okay, so, let's work on how do you rip a page to a dependency graph where \"dependency\" becomes the document root container or system root in whatever ui enriched programming language - they're all still containers in containers, we want to make something like the sympy node schema and vocab, we need to make the html one, and design our neutral conceptual node names"

> "might I suggest we call it AbstractUI and we make the individual backends like abstract tensor used"

> "now, it's important, vitally so, that we put two things in the ui tables. one is a backend that is prosaic compilation of annotations, the other is a backend that expresses the ui in the form of interactive map elements in a game, the root is a world, documents are regions, things in them that are objects with interiors are buildings, etc.. both of these will be vital to selling the uniform universal experience of consequential flow and accessibility. they will also form the contract by which we will laboriously adorn our OOP with so that we provide a richness that is more okay in an OOP environment where you're expected to invest time in heirarchical concept archival"

> "can you make a huge aspirational skeleton - bone yard really - of all this vocabulary, seemingly defined like it were intuitive, like it were made by someone writing perl, you just have human language intentions and basic syntax and assume it will work"

> "Okay we need a few tests coming at this from all angles or a panagram kind of test, we want to take a class that's from the repo somewhere, then turn it into an introspective map, random collections of types of rooms can be used for types of programmatic metaphor, or room for such fluid interpretation should exist, for taking pure programming and auto annotating it slightly whimsically just because it gives the code a spatial story that feeds the user, then we want something that makes our free form expressive language construction and wire it into a pythonic or ssa object definition or both, we want to break the ice and see that we can come up with a sensible auto grid to get a nice planar map of a class with methods members etc, absorbing the world that way from either side as many recursive levels as selected going up and down. And can we establish what that means to abstract ui and what objects it instantiates, what code does it imply"

> "Okay now we have some good parts but we're going to need an archetype library, which is mildly different than the things we've gone over, it's the user's ability to instantiate something into the program, while we do this keep in mind were also setting up our intelitype, because doing things like building in the world is going to introduce something to the namespace of the world, just like every line connects into the context of everything around it. The user being somewhere, doing something, is an edit in the living document in whatever form it's in among the countless representation spaces (dual ir probably). Imagine:
>
> class panel=box.with(class, inside).with(buttons.with(label, front),front).with(displays, top-front).connect(displays, class. Members)
> Class panel.connect(buttons, class.methods)"

> "now, let's design a top down map made from our grid concept of the map of data, using divs, so we can see that a page can hold javascript as inspectable onscreen elements - that very javascript - designed by our developing abstract ui system"

> "we may as well make this a abstractui object/set of methods, you know? the bare-bones map"

> "then abstract ui objects we'll make it a convention, they can carry html, script, and css packets"

> "I mean we'll be playing with this clay for a minute"

> "lets make the abstract code assume event listener is omnipotent and we only need to identify on the node interacted with the type of interaction and the node destination"

> "proceed, also, allow for the creation of npc entities, like, we'll aim to give the mouse an actor and then spawn a mouse like entity - identical entirely, spawned the same way, but instead of hooking to a mouse it hooks to a 2nd order integrator that reads the mouse location, pushing for the need for organization structures that corral our entities, allowing an entity cycle to administer to them alone, later, but just becoming a part of, I suppose they'd be... in system root, we give entities a mezzanine between fundamental timekeeping and grpahics and when-we-get-around-to-it actions where we can use a thread or not, keep things organized, let them interact with each other as conceptual objects"

> "under system root mezzanine we'll pop in action edges which will just be a table wired to a system timer we'll put in every javascript emission's root, we'll have a row, timer connects to action edges with update(actions) which will light up any row that has recently been issued an event on that edge"

> "let's make a factory archtype, takes a class, obtains a heap, dispensese and destroys out of it, holds defaults but can be invoked with a different request, all instances from a factory reachable through the factory for broadcast methods that sit between class and instance"

> "archtypes we're going to need (some of which may already be basic ui objects or constructable from them):
>
> div
> input
> pallet:
> fg, bg, margins, radii, font, decoration, visible, locked
> bbox
> graph"

## 2026-08-24 — Ordered follower family and entity tools

The pointer demonstration now instantiates one native pointer plus first-,
second-, third-, and fourth-order followers from the same `pointer-being`
archetype. Their colors are per-instance neutral traits and are emitted in both
entity descriptions and sprite presentation. The follower cycle is a general
derivative-chain integrator based on `(D + omega)^n`; order is controller data,
not four separately named implementations. Existing stiffness/damping
second-order records retain their earlier behavior.

`abstract_ui_tools.py` introduces backend-neutral `color-selector`,
`EntityInventory`, `InventoryItem`, and `ActiveTool` records. Inventories refer
to entity identities and enforce one active item with the `tool` role, without
asserting ownership or requiring archetype provenance. This preserves the
architectural distinction: archetypes are themselves objects and organize
recipes, but they are not a mandatory gate through which all objects must pass.

The generated self-map was rebuilt and its embedded JavaScript passed Node
syntax checking. The focused AbstractUI suite passes 57 tests. Interactive
localhost inspection could not be repeated because the in-app browser's URL
policy rejected the local page; no workaround was attempted.

> "we don't want to force people to make things in any particular language. it would be best if we were all the time completing and refining how we use our vocabulary, there's going to be a thing where like, archtypes are objects, for sure, but not every object is going to have to come from an archtype, archtype just places them in a universe of recipes"

> "just for me while we work on this can you put a first, third, and fourth order follower in the entity table just to be silly, and color them all different and put color onto the entity descriptions"

> "which make me realize you probably want a colorselector and entity inventories and active tools"

## 2026-08-24 — Port misread corrected to viewport

The initial reading of “port” as a compiler connection boundary was wrong. The
temporary `Port -> AbstractUIPort / ShaderPort` hierarchy was removed in full,
including its tests and documentation. The intended object was spatial: a
viewport and a shader-filled viewport inheriting it. Existing precedent was
found in `InterfaceRoot`, the web shell's shader surface, `LiveSSAViewport`,
and spectral-analyzer's `GLViewportWidget`.

> "lets make the abstractui's port and shaderport inheriting port"

> "did any precedent of that exist? I did not mean, uh, network ports, sorry, I meant a viewport and then one that is filled with a shader"

## 2026-08-24 — First-person shader view of the living data map

Added neutral `Viewport`, inherited `ShaderViewport`, `ViewerCamera`, and
ordered `FragmentOperation` records. The self-map now places a shader viewport
inside system root above the document regions. Its neutral document geometry
extrudes courtyard, building, and room boxes from deterministic map grid
positions. A WebGL2 fragment renderer ray-intersects those boxes from a
first-person camera.

`UIPalette` now supports role-named colors with layered merge semantics. One
resolved appearance record generates both CSS variables and shader material
uniforms. Entity poses now carry facing: native direction follows consecutive
pointer positions and follower direction follows velocity. Entering the
document grid maps the pointer actor into data-world coordinates and attaches
the viewer camera to that pose and direction; no artificial sixth-order
orientation follower was added.

The focused AbstractUI plus shader-component suite passes 72 tests. Node accepts
the emitted JavaScript, both embedded GLSL ES stages pass `glslangValidator`,
and the regenerated page is served successfully from localhost with the viewer
model and fragment source present. Browser-policy restrictions prevented an
automated GPU-frame inspection.

> "above the \"document reagion\" of the \"living data map\" still inside \"system root\" we could put, viewer camera, a shader port that we use with a fragment chain that takes our top down \"map\" of the data world and gives things height, courtyard containing buildings containing rooms containing objects"

> "when the mouse is inside the map, we're going to show in the shader port the output of putting the map as a document geometry into a first person camera of the map in 3d"

### Shader visibility repair

The initial viewport changed brightness on map entry but showed no surfaces.
Two renderer defects were repaired: WebGL uniform arrays now use the required
`uBoxes[0]` and `uMaterials[0]` base locations, and box intersection returns an
exit hit when the camera begins inside an extrusion. A real Canvas2D
first-person backend now projects the same neutral geometry when WebGL2 is
unavailable; the earlier implementation only reported that situation while
leaving the viewport dark.

A second visibility audit proved the neutral self-map contained 18 extrusions
and that its default center ray intersected a room. The remaining collapse was
photometric: the front wall shaded to nearly the same 3–11% intensity range as
the sky. Separate palette-owned courtyard/building/room face colors, a visible
sun and sky gradient, and stronger key/fill lighting now make geometry
unambiguous. The viewport readout publishes backend, extrusion count, camera,
and facing. The focused suite passes 73 tests after regeneration.

> "your browser doesn't support webgl2, I don't know if that's the only one we support for this, in chrome it works, when the mouse is in the map area, the shader is brighter, however there is no light, no surfaces, just a brighter dark"

The contrast change still did not make Chrome's procedural fragment-ray scene
visible. The WebGL2 realization was therefore simplified structurally: each
neutral extrusion now expands into ordinary position/normal/palette-color
triangles, rendered with a depth buffer and conventional perspective vertex
shader. The fragment shader can no longer suppress geometry through ray-box
logic; it only applies palette-owned warm light, key/fill illumination, and
fog. The 18 current boxes produce 648 inspectable vertices. Canvas2D remains an
independent fallback over the same neutral geometry.

> "the canvas 2d projection has some quirks but it shows something, the shader version is still not showing anything but the focus illumination"

## 2026-08-24 — Python-authored Canvas mesh projection through WebAssembly

The Canvas fallback now consumes the exact extruded triangle buffer uploaded
by WebGL rather than rebuilding each neutral box as one screen rectangle. Its
perspective transform is a retained Python function compiled through the
captured numerical region and common fused IR into a 491-byte WebAssembly
kernel. The emitted AbstractUI model carries source language, exact Python
source, lowering stages, ordered ABI parameters, binary payload, byte count,
and operation count. JavaScript owns only camera-basis preparation, WASM
linear-memory arrays, painter ordering, and palette-equivalent lighting.

This exposed and repaired a general operator-vocabulary defect: Python AST
captured division as `Div`, while the WASM numerical backend accepted only the
canonical `truediv`. Elementwise spellings are now case-folded through aliases
and normalized once at the backend boundary, including reflected spellings,
instead of teaching this one projection a private translation table.

The focused projection tests execute the emitted module in Node through its
published byte-offset ABI and verify perspective coordinates. The generated
self-map script passes Node syntax checking. In the in-app browser, WebGL2 was
unavailable as expected; Canvas initialized as `Canvas2D + Python→WASM` and
after the pointer entered the document region the readout switched to live
camera/facing coordinates and reported all 216 mesh triangles at positive
depth. Visual surfaces were still not distinguishable in that active state, so
near-plane clipping/projected bounds and palette parity remain an explicit
compatibility follow-up rather than a claimed completion. Work stopped at that
honest boundary at the user's request so architectural progress would not be
held hostage by one browser.

> "shader is now perfect, the canvas projection is a little rough, can we get something closer to a mesh with perspective like the shader, we can program it in python and then compile it to web assembly"

> "I left the machine i can't check it for you right now and my phone works with shaders so i dunno what to tell you, but we can make more important headway moving on and ensuring that compatibility later"

## 2026-08-25 — Viewport-owned actor control context

The first-person viewport is now an input context independent of its shader or
document renderer. A neutral `ViewportControlPolicy` names the controlled actor,
highlight activation, escape/focus release, captured device classes, movement
and look rates, first-connected-gamepad selection, and ordered mappings from
keyboard/pointer/gamepad inputs to semantic actor actions. The policy belongs
to base `Viewport`, not `ShaderViewport`, so an SDL, pygame, Java, native, or
headless realization can consume the same contract.

The browser backend focuses and visibly highlights the viewport on click,
requests relative pointer lock when permitted, moves and looks with WASD/mouse,
and polls the first available gamepad each entity cycle. Standard left/right
sticks supply movement/look. Mouse button zero and gamepad button zero issue
the same `primary-action` action edge to the tracked actor. Escape, focus loss,
or selection of another AbstractUI object relinquishes the context. The
focused viewport/div-map suite passes 25 tests.

> "the fps to me signals a moment of unification, because we can provide that view anywhere even without a doc renderer. One of the things the viewport is going to need to be able to do is steal controls when highlighted, causing wasd and mouse to become the actor controls, and hopefully if possible, picking up any gamepad to use as input"

### Canvas control-capture regression

The first control-host pass made Canvas active but black by replacing the last
working actor-derived camera with a generic center-line pose; active state was
also latched after release. Control takeover now inherits the most recent
inhabited camera. If none exists it deterministically enters opposite the first
room and faces it. A geometry-side projection audit finds 190 of the current
216 triangles intersect the Canvas from that entrance. Canvas additionally
falls back to its rectangle projection only when the compiled mesh reports
zero on-screen triangles. The WebGL path was not changed by this repair.

> "something broke the shader, it doesn't work anymore I'm just getting active black again"

> "sorry just the canvas one is dead"

## 2026-08-25 — Live device pickup and explicit dynamics lanes

The viewport now carries two more Python-authored AbstractUI objects beneath
its presentation surface. `DeviceMonitor` is mechanically derived from ordered
control bindings and groups pointer, keyboard, and gamepad signals without
duplicating their semantic action table. The browser renders a narrow live
strip: relative mouse motion, M1/M2, W/A/S/D, four left-stick directions, right
stick, and gamepad buttons zero/one illuminate while sampled and remain dark
while inactive. Gamepad presence is polled independently of viewport capture so the
strip can show device pickup before control is taken.

`DynamicsSpace` reserves separate `user-dynamics` and `world-physics` lanes.
User intent, position, velocity, and facing are bound to the actor and displayed
live. World geometry is bound; contacts, collision, and gravity are explicitly
unbound and visibly say so. These are stable compilation destinations, not
claims of implemented physics. Both records belong to base viewport data and
can be realized without a document renderer.

> "can you give me a narrow section under the shader viewport that shows the mouse, mapped keyboard keys, and mapped gamepad buttons, that light up when on and are dark when inactive, so you see the live pickup of devices. further we need to carve out a space just for user dynamics and world physics"

## 2026-08-25 — Identity-preserving scene mesh and Form round trip

Added `turing/src/compiler/abstract_ui_scene_mesh.py`. Its parametric box
vertex constructor is authored as retained Python, lowered through the common
captured numerical/fused-program path, and emitted as a 240-byte WebAssembly
array kernel. Box corner and triangle topology are published in the AbstractUI
model rather than hidden in WebGL code. Every 36-vertex instance receives a
parallel span containing its courtyard/building/room identity, geometry index,
and revision.

The browser backend now uses that compiled constructor for the shared WebGL
and Canvas triangle mesh. A camera ray through the viewport crosshair resolves
back through the span table to a page object identity. Right mouse or mapped
gamepad B1 opens that object's context menu. The model-authored `Form` submenu
offers height, width, depth, and reset instructions. Applying one mutates the
shared geometry parameters, reruns WebAssembly, reuploads the GPU buffer,
increments the mesh revision, emits an `apply-form` action edge, and publishes
the same identity/revision/parameters to its DOM node. Region and building
containers now expose `data-node-id` as rooms already did, completing all
current geometry-to-document matches.

This is a live in-memory edit, not yet a durable source mutation. The contract
keeps Form instructions and identity spans independent of the browser so the
next step can validate them as archetype parameters and record them as typed
LivingDocument/DualIR edits for Python, SSA, C++, Java, or other source
authorities.

- The focused scene-mesh/div-map/software-mesh/viewport/dynamics suite passes
  36 tests.
- The WebAssembly kernel executes through its published byte-offset ABI in
  Node, and the complete emitted JavaScript packet passes `node --check`.
- `turing/docs/generated/abstract_ui_object_map.html` was regenerated.

> "let's work up some web assembly to drop the scene to mesh with page object identity, and able to push that mesh+identity state to the document to render it back into DOM, and in that, we're going to need the player in the shader to be able to open a context menu about whatever is in the crosshairs, and that context menu should have the option \"form\" which opens into a submenu of parametric mesh instructions used to instantiate the appearance of the object in both the document and the game map"

### Scene visibility initialization repair

The first scene-mesh integration accidentally made initial visibility depend
on `structuredClone` and successful WebAssembly setup. A compatibility failure
before buffer construction consequently left the independent HTML crosshair
and target label visible over a black canvas. Initialization now installs the
known-good portable triangle mesh and identity spans first. The compiled mesh
may replace it only after exact-length, finite-value, and non-collapsed-bounds
validation. Form edits use either realization, and geometry cloning is an
explicit portable operation. The regenerated script passes Node syntax
checking and the focused suite passes 37 tests.

> "we've lost visibility of the shader environment, I'm only seeing the tooltip and crosshair"

The first repair had zero visual effect and its diagnosis was rejected. The
next pass addressed composition directly: the canvas now owns an explicit
base stacking layer, while crosshair and target-label layers are isolated,
transparent, and narrowly bounded. More importantly for the in-app Canvas
backend, coarse neutral geometry now remains as a visible floor beneath the
compiled triangle refinement. A numerically “onscreen” compiled projection can
therefore no longer erase the only visible environment. The page was
regenerated and the focused suite remains at 37 passing tests.

> "no. whatever you did it had ZERO effect. perhaps there's no alpha on your overlays"

The overlay experiment was then removed rather than refined further. The
viewport again appends its canvas directly beneath the header with no stage
wrapper, crosshair element, or target-label element above it. The live renderer
also stays on the last-known-visible host extrusion buffer; the compiled scene
constructor remains in the neutral model but cannot replace presentation.
Identity picking and Form actions remain data operations without an onscreen
crosshair until a renderer-native presentation is reintroduced behind a visual
gate. The regenerated script passes syntax checking and 37 focused tests.

> "did you literally cover the shader with dom elements"

The crosshair was subsequently restored as renderer output. WebGL now performs
a dedicated fullscreen GLSL alpha-blended reticle pass after drawing scene
geometry; Canvas performs an equivalent final compositor draw. Reticle color
comes from the AbstractUI palette and changes when identity picking resolves a
target. No persistent DOM crosshair, tooltip, or stage wrapper exists. The
context menu is instantiated only on secondary action because its textual Form
controls require an accessible interaction surface. Both new GLSL stages pass
`glslangValidator`; emitted JavaScript syntax and 37 focused tests pass.

> "put the features back ... but do them right ... and put them in the shader"

### Bounded focus tooltip and border-wall contract

A single bounded focus tooltip is now appended as a sibling after the direct
canvas, never as a canvas wrapper or viewport-sized overlay. It is absolutely
positioned from the measured canvas center, clamps to the port width, has a
translucent background, ignores pointer input, and is removed from layout when
no identity is picked. It reports object kind, name, identity, shared wall
height, and mesh revision. The first right-click now opens the identity-aware
Form menu without requiring a preceding viewport-capture click; gamepad B1
retains the same secondary action.

Document geometry and the scene-mesh model now publish `dom-border` as the
wall boundary source and box `height` as wall height. Runtime publication keeps
mesh height, DOM `data-wall-height`, CSS `--wall-height`, and tooltip height in
sync. The neutral contract reserves document-ordered door/window/portal
openings and names the future operation `boundary-union-minus-openings`, while
honestly leaving composite-solid boolean geometry unimplemented. The generated
script passes syntax checking and the focused suite passes 38 tests.

> "I miss that little tooltip. also we still need the context menu. let's very very carefully try to use dom elements to make the tool tip float as a dom element nearby the crosshair showing info on whatever is focused on. ... let the borders be the walls and then consequently give them a height parameter."

## 2026-08-25 — Inventory hotbar and captured-input focus routing

Extended `abstract_ui_tools.py` with one-based inventory slots and a ten-slot
`Hotbar` which is explicitly a view of inventory positions 1–10. Numeric keys
map in authored order from `Digit1` through `Digit9` and `Digit0`. The pointer
actor's initial inventory now contains one equipped `Form tool` in slot 1;
remaining hotbar positions are explicit empties. Keyboard or delegated hotbar
selection updates the shared inventory active-tool identity and issues a
`select-hotbar-slot` action edge.

Added `abstract_ui_control_focus.py`. Its neutral policy separates physical
device capture from routing authority across `game`, `projected-pointer`, and
`dialogue` contexts. Secondary action switches game/projected modes. Projected
pointer motion is clamped and published in document coordinates while logical
viewport ownership remains active. Dialogue focus preempts both, requires a
response, and resumes its prior mode. Numeric hotbar bindings are ignored while
a dialogue owns focus, preventing tool selection from consuming typed answers.

The browser renders the hotbar below viewport telemetry and exposes current
focus mode. Right mouse and gamepad B1 use the same switch action; entering
projected-pointer mode opens the focused object's context menu, while switching
back closes it and restores game pointer capture. The regenerated JavaScript
passes syntax checking and the focused suite passes 46 tests.

> "we need to bind numbers to a hotbar and add the first tool, 1 slot, also represented in inventory in the first 10 slots the hotbar is represented and edited. we're going to have to handle focus switches between live dialogues that need a response and the control capturing game service"

### AbstractUI tool hooks and aesthetic editor

Promoted tools from inventory flags to `AbstractUITool` objects. A tool now
owns semantic primary/secondary `ToolHook` records and may own a model-authored
`ToolDialogue`. Inventory and hotbar records reference the tool identity rather
than duplicating its behavior. Browser pointer and gamepad actions route
through the active tool's hooks and issue operation-level action edges.

The initial Form tool maps primary action to `open-dialogue` and secondary to
`toggle-focus-context`. Its aesthetic dialogue claims high-priority response
focus for the crosshair object and offers live face color, wall color, wall
height, wall thickness, and radius controls plus Verdant, Warm, and Stone
presets. Edits mutate the shared geometry/appearance record, rebuild the scene
buffer, and republish coordinated DOM wall/color/radius values. Done or Escape
releases response focus and resumes the previous game/projected mode, including
pointer capture when appropriate. The full focused suite passes 49 tests; the
final tool/focus/div-map gate passes 37 tests and emitted JavaScript syntax.

> "primary button and secondary button are going to be routed to hooks the tool posesses. we need to define tool as an abstractui object. it's going to pop up a dialogue that lets you edit the aeshthetic properties as well as pick from presets"

### Hollow layout solids, subtractive openings, and edit autosave

Replaced the live solid-box layout realization with composite geometry under
one variable-length identity span. Every courtyard/building/room now emits a
mandatory floor slab, four narrow boundary-wall prisms, and a hollow interior.
Ordered openings subtract intervals from their named wall and retain a lintel
when shorter than wall height. A ceiling is emitted only at the declared 4.0
absolute maximum. Default courtyards use low fencing and a full-height gate;
buildings have an enclosing entry; rooms have their own door-bearing walls.

Floor and wall surfaces now consume separate palette roles, making the Form
tool's face and wall color controls independently visible. Wall thickness
changes actual wall-prism thickness and DOM border thickness. Radius remains
coordinated in DOM/model data and is explicitly marked as an unimplemented
mesh bevel rather than being falsely represented. Identity spans now record
the actual composed primitive count instead of assuming 36 vertices per object.

User changes autosave by edited identity. The browser writes a versioned,
one-year cookie and verifies its exact encoded value, while mirroring to local
storage because `file://` cookie persistence is unreliable. Height, appearance,
and ordered openings restore before initial mesh construction. The focused
suite passes 51 tests and emitted JavaScript passes syntax checking.

> "the interior region of a div is a floor - an empty space ... the borders should honestly be walls, extrusions that climb up and leave a hollow interior. we can cap them with ceilings when wall height hits the absolute max. ... configure the outer spaces to be like courtyards with fencing and gates ... autosave all user edits to cookies for now"

## 2026-08-25 — Pluck-compatible world registry and WASM plugins

Added `turing/src/compiler/abstract_ui_world.py`. The living map now separates
conceptual `WorldObject` authority from mesh realization and rendered bake
products, paralleling Pluck's `PlacedObject`/`RoomWorkspace`, procedural mesh
builders, and render-asset catalog. A lossless `pluck_placed_object()` adapter
promotes recognized fields while retaining the complete source dictionary in
a namespaced extension, including future game fields unknown to AbstractUI.

Every document courtyard, building, and room now carries parent containment,
position/yaw, form recipe, material-role bindings, capabilities, static
collision intent, persistence authority, and semantic parts. The live mesh
publishes variable-length object spans plus individual floor, wall, opening
lintel, and ceiling spans. Form edits update the corresponding world-object
recipe, mesh span, DOM facts, and shared revision; GPU buffers remain derived.

The world registry embeds three Python-authored WebAssembly plugins through one
published ABI: parametric box vertices, Pluck-style position/yaw transforms,
and perspective projection. Direct Node execution verified the new transform
plugin. The generated `abstract_ui_object_map.html` contains 18 world objects
and all three plugins.

- Focused AbstractUI suite: 99 passed.
- Cold adaptation of Pluck's real `configs/room_station/scene.yaml`: 13 objects
  across camera, duty-station, enclosure, light, and portal kinds; every
  serialized source record survived in its namespaced extension.
- Direct generated-page JavaScript compilation and embedded-model parsing:
  passed.
- `git diff --check` for touched source, tests, and documentation: passed.

> "I'm giving you carte blanche to inherit the pluck systems for the webgl system in ways that seem appropriate ... when all that game metadata is present and riding along and those helpers have been compiled into web assembly plugins for the living data map, we should be extremely well positioned to start dealing with physics and design permanence and emergence"

## 2026-08-25 — Emitter-owned runtime utilities and loose performance objects

Added `javascript_runtime_utilities.py` and an optional `runtime_utilities`
dependency request on the repository-SSA JavaScript emitter. Utilities have
semantic identity, SHA-256 content identity, deterministic dependency closure,
capability, callable exports, and performance metadata. The initial inventory
provides a content-addressed lazy WASM instance cache, an identity/containment/
semantic-part world registry, and a monotonic revision channel.

The living map now obtains these services from the emitter utility inventory.
WASM binaries appear once in a content-addressed module table; three plugin and
presentation descriptors reference them. Scene/software consumers instantiate
through the shared promise cache. Form and aesthetic edits publish monotonic
revision events in addition to updating their coordinated world/mesh/DOM data.

Every emitted SSA function now carries an advisory performance record with
inline intent (`prefer`, `neutral`, `avoid`, or `forbid`), authored-vs-estimated
basis, hot-path/frequency facts, instruction/block/call/branch counts, async
boundary, and allocation risk. AbstractUI realizes method labels as small
derived `performance-observation` objects loosely contained inside the method
room. They are inspectable but do not become source members or structural
rooms. The generated AbstractUI map currently contains six such observations.

- Expanded JavaScript-emitter plus AbstractUI gate: 116 passed.
- Generated page JavaScript compilation and embedded model parse: passed.
- Page model: three content-addressed WASM modules, three plugin references,
  zero duplicate binaries on plugin records.

> "i was thinking of using labeling of inline functions or methods, to promote performance awareness ... in the visualization world, these will be stray objects inside the domains"

## 2026-08-25 — Executable symbolic world physics

Added `abstract_ui_physics.py`, whose numerical authority is eight simultaneous
SymPy equations. The transition covers semi-implicit gravity/force integration,
implicit linear drag, compliant unilateral AABB contact, and source-relative
yaw/translation transposition between portal or representation boundaries. It
also publishes contact-penetration and specific-kinetic-energy metrics.

The equations lower through the existing canonical SymPy ProcessGraph and
repository SSA compiler into a 1,759-byte WebAssembly module. The common world
plugin ABI now describes direct scalar-arena input/output offsets. A Node
execution test compares falling and floor-contact behavior, and the SymPy
oracle separately verifies a 90-degree portal traversal.

The generated page binds the compiled artifact to the player entity cycle.
Twenty-five parameter records cover gravity, force, inverse mass, drag, body
radius, contact softness, bounds, and portal anchors/yaw. Identified number
inputs edit the WASM feed values without recompilation and persist alongside
the living representation edits. Portal activation remains an event-supplied
input so a saved value cannot repeatedly teleport the actor.

- Focused physics/div-map/dynamics/world/emitter gate: 63 passed.
- Expanded AbstractUI/compiler gate completed with 138 passing tests.
- Generated artifact: 387,165 bytes; JavaScript parses in Node; four
  content-addressed WASM modules, with the physics module using
  `ssa-scalar-arena-v0`.

> "can you express in sympy in a tight set of equations, gravity, boundary traversal transposition? how much can you give us of basic world physics as functions of available programmatic parameters, compiled parametrically so we can live edit those parameters on the site, using more tight adequate equations than strict, you know, soft body physics or anything, unless you want to try giving everything some soft body XD XD"

### Interior wall rejection correction

The initial active physics binding constrained only the outer representation
envelope; internal wall physics intent was not connected to the executable.
`buildExtrudedBoxMesh()` now emits planar collider records from the exact same
segmented wall/lintel prisms used for rendering. Open door/gate intervals have
no collider. Vertical overlap prevents high lintels from blocking the player.

The host selects the contacted face and supplies obstacle activation, planar
normal, and plane coordinate to four new non-editable WASM parameters. The
SymPy program applies one additional unilateral plane projection and publishes
the semantic wall identity plus dense runtime part ID in live contact state.
The generated page is 469,897 bytes and embeds the resulting 2,153-byte physics
module with 36 scalar inputs.

- SymPy oracle plus direct WASM physics suite: 6 passed.
- Focused page collider/source tests: 2 passed.
- Generated JavaScript parsed successfully in Node.
- Automated interactive inspection was unavailable because the in-app
  browser policy rejected the local `file://` page; no alternate route was
  attempted.

> "I'm not getting rejected by walls"

## 2026-08-25 — Human artifacts and logical filesystem graph

Added `abstract_ui_filesystem.py` to keep three commonly conflated relations
orthogonal: semantic ownership, logical filesystem containment, and current
world placement. The graph seeds source, test, README, annotation, and scratch
artifacts. Each has one identity across the path graph, DOM inspector, world
registry, and shader geometry.

Loose artifacts realize as small colored dynamic solid boxes. Their attachment
contract advances `loose -> settling -> welded` only after sustained proximity
below a relative-speed threshold; disturbance resets settling. Welding changes
the physics realization to an owner compound child without renaming the path,
moving the representation, or transferring semantic ownership. The page runs
the same deterministic transition during the entity cycle and displays each
attachment state.

The filesystem graph defines internal, native-C, web, and WASM realization
contracts. Native structure binding uses translation-unit/symbol edges; web
uses module URLs and bundle manifests; WASM uses host imports and reversible
dense path IDs. All preserve authored ownership edges. The accompanying design
document records graph invariants and explicitly rejects ambient host-path
authority.

- Filesystem/world/focused div-map gate: 12 passed.
- Vocabulary gate after adding filesystem/artifact/weld words: 5 passed.
- Remaining div-map tail gate: 7 passed; the preceding 30 tests emitted passing
  progress before the desktop command-output deadline.
- Generated page: 522,738 bytes, 12 filesystem nodes, 5 artifacts, 30 world
  objects; embedded JavaScript parsed successfully in Node.

> "Annotations, readme, scratch files, source files, tests, they can be anywhere
> on the map but they end up owned by something"

## 2026-08-25 — Placement custody, subtractive stock, and parent-map skybox

Added `abstract_ui_placement.py` with an explicit representation-custody
lifecycle (`inventory -> preview -> placed`). Payloads retain semantic owner,
source container, filesystem relations, authored identity, and their complete
representation packet. Taking and placing therefore does not masquerade as an
ownership transfer.

The browser projection now includes a placement tool, X/Y/Z/yaw gimbal,
free/grid/object-center/object-face/opening-track snap vocabulary, non-colliding
preview meshes, placement revision publication, and persisted pose/custody.
Inventory entries expose quantity, maximum stack, and stack key. Unique authored
objects remain one-item stacks; recipe stock starts with 8 doors, 12 windows,
4 gates, and 2 portals.

Door/window/gate/portal recipes commit as identified subtractive objects owned
by their boundary host. They enter the host opening and semantic-part tables,
decrement stock, rebuild both visible wall segments and colliders, and persist.
Window openings now include a sill prism instead of incorrectly cutting from
the floor.

The global representation envelope is now a persistent, non-colliding
12-unit skybox wall with no ceiling. Its horizon is explicitly the parent world
map boundary and crossing retains the existing `switch-map-representation`
contract.

- Placement/tools/viewports/vocabulary and focused page gates: 27 passed.
- First 32 div-map tests emitted 30 passing results before output handoff; the
  remaining two were rerun directly and passed.
- Final 7 div-map tests: passed.
- Generated page: 550,002 bytes; embedded JavaScript and model parse passed in
  Node.

> "we're going to need a placement tool ... snap to other objects ... doors,
> windows ... subtractive elements ... inventory counts ... a skybox on the
> outer layer"

## 2026-08-25 — Player physics-ball gun

Added `abstract_ui_projectiles.py` and equipped the player with a physics-ball
gun in slot 7 plus 64 rounds in slot 8. Primary action mints a stable projectile
identity, adds it to a projectile-only entity organization, emits a spherical
mesh and collider, and decrements ammunition. Active population is bounded at
24 and lifetime is 12 seconds; expiry removes presentation/active membership
while retaining a spent identity record until explicitly cleared.

Projectile motion invokes the same compiled SymPy→SSA→WASM transition as the
player, with per-ball position, velocity, radius, drag, floor bound, wall
contact, and world bounds. JavaScript administers rows and republishes poses;
it does not contain an alternate gravity integrator. The portable mesh baker
now emits identified latitude/longitude sphere triangles and mesh-part spans.

A concurrent A* navigation integration increased its static WebAssembly memory
requirement beyond its declared 512 KiB while this gate was running. Its initial
memory declaration was narrowly raised to 2 MiB, preserving the integration and
restoring clean page generation.

- Projectile/vocabulary/focused page gates: 13 passed.
- Fresh combined projectile/physics gate after navigation correction: 6 passed.
- Generated combined page: 570,957 bytes; JavaScript/model parse passed in Node;
  navigation kernel and projectile gun are both present.

> "can you make the player a gun that shoots physics balls"
