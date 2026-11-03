# Documentation Report

**Date:** 1784612211
**Title:** SDF/metaball hull plan for flux_radar -- designed, not yet built

## Overview
The user asked for the flux_radar WebGL hull (the rendered "body" of the
FluxGraph creature) to stop looking like discrete geometric quads and
instead have a viscous, gloppy, soft-body appearance -- specifically a
metaball/SDF hull that lets high-growth-interest nodes stretch pseudopodia-
like reaches outward that relax back when growth interest fades. The user
was explicit that this must **not** be generic procedural noise: every
visual behavior must trace back to a real simulated state variable
(pressure, volume, solvent/solubles, hull_permeability, growth interest,
etc.), the same way the existing edge shader's flow-driven pulse already
does. This work was planned in detail but never implemented -- the thread
got redirected to a real bug in the water/ion-transport system (see the
water-gating work landed in `core/flux_graph.py` in this same session;
check git log around this report's date) before any shader code was
written. This report is the handoff so the next agent doesn't have to
re-derive the plan.

## Current State (as of this report)
`speaktome/flux_radar/src/webgl_renderer.ts` (`FluxWebGLRenderer`) renders
three things every `update(frame)` call:
- `rebuildHulls()` -- one flat quad per edge, using a custom shader
  (`VERTEX_SHADER`/`FRAGMENT_SHADER` near the top of the file) that already
  has one real-data-driven animation: a per-fragment pulse keyed off that
  edge's real `flow` value (`fract(vAlong*7 - time*vFlow)`). This is *not*
  decorative -- keep this shader technique as the template for "animation
  that tells the truth about a state variable."
- `rebuildLumens()` -- inner per-segment tube quads, colored by
  `dominantFluid()` (single dominant soluble only), width from
  pressure+solvent+solubles.
- `rebuildNodes()` -- instanced flat circles (`nodeMesh`) plus an inner
  "fluid fill" circle (`fluidMesh`) scaled by `volume`.

Decision already made in planning discussion with the user: **keep
`rebuildLumens` (the edge/tube rendering) as-is.** Only `rebuildHulls` /
`rebuildNodes` (the node "body") should be replaced by the metaball
approach described below.

## The Plan (not implemented)
Replace the discrete per-node circle + per-edge quad hull with a single
fullscreen-quad + fragment-shader raymarch pass. Each node is one metaball.
Every shader input must be a real field already present in the snapshot
(`snapshot_graph()` in `flux_radar_server.py` already emits all of these --
no server-side changes were believed necessary for this specific piece):

- **position, radius** <- node `volume` / `pressure` (bigger/more
  pressurized node = bigger blob). `volume` is `solvent + sum(solubles)`,
  so a freshly-spawned or still-dehydrating node legitimately starts near
  radius 0 and grows as it hydrates -- see the "Water-gating interaction"
  note below, this is now *more* true than when the plan was drafted.
- **blend softness** (the `smin` k parameter used when merging
  neighboring blobs) <- `hull_permeability`. High permeability = blobs
  merge smoothly (reads as one connected fluid body); low permeability =
  blobs stay visually separate (reads as walled-off compartments).
- **color** <- a full weighted blend across *every* entry in `solubles`
  plus the `solvent` fraction, not just the single dominant soluble the
  way today's `dominantFluid()` helper does. The color should genuinely
  represent the real dissolved mixture.
- **pseudopodia reach** <- `forward_growth_interest` /
  `backward_growth_interest`. Extend a capsule-shaped SDF from the node
  toward the position of whichever child it is actively trying to grow
  (or a generic outward direction if there's no live target yet), length
  scaled by the growth-interest magnitude, springing back toward a plain
  circle as growth interest decays tick to tick. This makes the reach a
  direct visualization of the growth_commitment/growth_interest mechanic
  that already exists in `core/flux_graph.py`, not invented motion.

Math sketch: for screen point `p`, per-node distance
`d_i = length(p - center_i) - radius_i`, optionally unioned with a capsule
SDF toward the reach target scaled by reach amount. Combine every node's
distance into one field via a polynomial smooth-min (`smin`), with each
node's own `k` blended from its `hull_permeability`. Final alpha via
`smoothstep` on the combined distance for antialiasing. Color via a
normalized weighted sum, e.g. weight `exp(-max(d_i, 0) * falloff)` per
node so nearer blobs dominate a pixel's color smoothly rather than a hard
cutover.

### Open / unresolved design questions
- **Max blob count.** Uniform arrays need a fixed cap (proposed ~128)
  since node count varies frame to frame and a long-running live session
  can accumulate hundreds of nodes even with `burn_after_ticks` pruning.
  No prioritization rule was decided for which nodes get a full blob vs.
  get dropped if the cap is exceeded (candidate: sort by `volume` or
  `pressure`, keep the top N). This was flagged, not resolved.
- **Performance.** Never benchmarked. A per-pixel loop over MAX_BLOBS
  distance evaluations for a ~900x560 canvas is probably fine on any real
  GPU but this was never actually measured.

## The Real Blocker: Build Pipeline, Not Yet Resolved
`index.html` imports the renderer from the **compiled bundle**
`/static/webgl/webgl_renderer.js` (served from
`speaktome/flux_radar/webgl/webgl_renderer.js`), **not** directly from
`src/webgl_renderer.ts`. Per `package.json`, the TS source is built via
`npm run build` (vite, `vite.config.ts`).

This was the actual point where this task stalled: I was mid-way through
checking whether a working `npm`/`node`/`vite` toolchain is even available
in this sandbox (`ls node_modules`, `which npm node`) when the session got
redirected to the water bug and never returned. **This is the first thing
the next agent must check before writing any shader code:**
1. Does `speaktome/flux_radar/node_modules` exist / does `npm run build`
   (or `npx vite build --config vite.config.ts`) actually work here?
2. If yes: edit `src/webgl_renderer.ts`, run the build, and confirm the
   output actually lands at `webgl/webgl_renderer.js` (the vite config's
   output path was never verified against that exact path).
3. If no working npm/vite toolchain is available: the fallback is
   hand-editing the compiled `webgl/webgl_renderer.js` directly (loses
   TS type-checking, uglier, but works).
4. There is *also* a separate `speaktome/flux_radar/dist/webgl_renderer.js`
   in the same directory. Its relationship to `webgl/webgl_renderer.js`
   was never established -- stale alternate build output? built by a
   different command/config? This needs to be sorted out first so edits
   land in the file the server actually serves (`webgl/`, confirmed by
   grepping `index.html`'s `<script type="module">` import path), not the
   one nobody reads.

## Water-Gating Interaction (context for whoever picks this up)
In this same session, `core/flux_graph.py`'s water/solute system was
audited and fixed (solutes/ions could previously move -- and, depending on
how far the growth-gating work in this same session went, nodes could
grow/metabolize -- with zero solvent present anywhere in the transaction;
see git log around this report's date/commit for the exact diff). Practical
effect for this graphics plan: expect early ticks of any run to show more
static, mostly-still, near-zero-radius nodes for longer than before, while
they wait to hydrate from ambient humidity -- this is intentional, correct
behavior now, not a bug, and the SDF hull's "small dull blob slowly
plumping up as it hydrates, then reaching pseudopodia toward what it's
trying to grow" is a good visual match for it, not something needing a
special case.

## Next Steps
- Resolve the build-pipeline question above before writing shader code.
- Implement the SDF/metaball hull per the plan above, replacing
  `rebuildHulls`/the node half of `rebuildNodes` in
  `speaktome/flux_radar/src/webgl_renderer.ts` (or the compiled JS
  directly, per whichever build path turns out to work).
- Decide and implement the max-blob-count policy.
- Record this file under `todo/` as a `.stub.md` (see
  `1784612211_SDF_Metaball_Hull.stub.md`).

## Prompt History
> "could you change the site controlling the server so it didn't kick off - or the server didn't kick off the non-live demo? I want to pick what starts, live or fixed"

> "are there any really bad efficiency problems? is the web interface using the gpu the server needs while the server needs it? is there any way we can make the site animation meshes of the pipes have a kind of procedurally fluid gloppy appearance, or we could put such a fluidic seeming hull around the creature that lets rays hold out regions that try to relax back into the volume to give pseudopodia appearance"

> "let's proceed, and maybe if you could clean up my default settings and put something on the controls that lets you know if your configuration is bonkers"

> "1. I don't care about the autosave thing ticks are taking like hours right now or aren't advancing in live or I had some kind of failure occur 2. I don't want procedural animation I want the physics to be represented by animations unique to each feature and state variable, something rich and meaningful that loops while waiting for another tick but tells the actual story of every micro-edge transport or managed fluid 3. I wanted a SDF volume smoother thing and if you can't do that in a quick browser shader I guess that's fine"

> "write notes on what you were going to do with the graphics that you never got around to for the nexdt agent, then take another look at water, turn off grows without water, no metabolism happens anywhere without water, even if the first many ticks are just sitting around waiting to prove the dry nodes are actually drawing in humidity the way they are supposed to supply things with water"
