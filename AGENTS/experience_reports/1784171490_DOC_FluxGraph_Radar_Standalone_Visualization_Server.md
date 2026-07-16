# Documentation Report

**Date:** 1784171490
**Title:** FluxGraph radar: standalone local web server + live visualization (concentric rings, hemisphere-split direction, spring physics)

## Overview

Follow-up to the same session as `1784162289_...` and `1784167169_...`
(GPU efficiency pass, then trie-gated word growth). The user asked for
visual feedback on the graph's physics -- concentric rings by generation,
pressure/score encoding, spring clustering -- and iterated through three
rounds:

1. First pass: a static claude.ai Artifact with one baked-in dataset.
   Forward/backward were distinguished only by a soft angular bias
   (up/down), no hard boundary.
2. User feedback: "there doesn't seem to be anything orienting forward
   and backward... two hemi-radars fitting in their arc segments but
   springed around" + "give the ability to run new data at different
   parameters."
3. Final: abandoned the static Artifact in favor of a real, standalone
   tool living in the repo -- a stdlib-only local HTTP server
   (`speaktome/flux_radar_server.py`) that runs actual FluxGraph sessions
   on demand from browser-submitted parameters, plus a frontend
   (`speaktome/flux_radar/index.html`) with a hard left/right hemisphere
   split and a "run new scenario" form.

## What exists now

- `speaktome/flux_radar_server.py`: `python -m speaktome.flux_radar_server
  [--port 8877] [--preload]`. Loads GPT-2 + the curated dictionary +
  both WordTrie instances once (`ModelBundle`, lazy on first request
  unless `--preload`), then serves `GET /` (the frontend) and `POST
  /api/run` (JSON params in: seed/ticks/budget/branch/alpha/beta; JSON
  tick-by-tick history out -- same per-node fields as FluxNode:
  pressure, local_evidence, path_mean, direction, depth/height, burned,
  created_tick, decoded text). Deliberately stdlib `http.server`, not
  Flask/FastAPI -- no new dependency (see `AGENTS_DO_NOT_PIP_MANUALLY.md`).
  One GPU run at a time via a lock; static-file serving path-traversal-
  checked (`_serve_file` resolves and verifies the path stays under
  `STATIC_DIR`).
- `speaktome/flux_radar/index.html`: a real D3-force-driven radar view.
  Concentric rings = generation depth (`FluxNode.depth`). Forward is a
  hard-constrained right hemisphere, backward a hard-constrained left
  hemisphere -- not just a soft angular bias: a custom `hemisphereForce`
  nudges velocity every simulation tick, *and* `render()` hard-clamps
  `d.x`/`d.vx` directly (not just the drawn attribute) so there is zero
  visible crossing even on the very first frame after new data lands,
  before the soft force has had time to converge. Node radius = pressure,
  brightness = `exp(local_evidence)` (the same quantity
  `FluxNode.local_value` computes), labels = decoded token text, hover
  tooltip = full per-node stats. Tick scrubber + play, live spring-
  stiffness/resting-length sliders (direct visual tuning input for
  `graph_layout.py`'s `spring_strength`/`spring_length`), a run-history
  selector (multiple `/api/run` calls populate a dropdown, switch
  between them client-side with no new server call), and a parameter
  form that triggers a fresh real FluxGraph run.
- `.claude/launch.json`: `flux-radar` config so the Browser pane's
  `preview_start` can launch/reuse the server directly.
- `tests/test_flux_radar_server.py`: routing/safety tests only (root
  page serves, unknown path 404s, static path traversal rejected 403,
  `/api/run` error path returns JSON 500 not a crash, success path
  returns the bundle's JSON) -- deliberately does not load the real
  model (monkeypatches `get_bundle`), since a full GPU-dependent test
  is too heavy for this test file's purpose.

## Steps taken

- Exported real tick-by-tick FluxGraph snapshots via a throwaway script
  before building any visualization, specifically to avoid designing
  against mock data. Discovered mid-way that the demo seed's backward
  side reliably starves out by tick 3 regardless of `compute_budget_per_tick`
  (tried 2, 3, 4) -- real FluxGraph dynamics, not a bug; used a shorter
  run for the first prototype to keep both hemispheres populated long
  enough to be useful as a demo.
- Verified the first Artifact prototype by manually flushing d3's
  internal timer (`d3.timerFlush()`) rather than trusting a screenshot,
  because the automated preview tab is backgrounded/hidden
  (`document.visibilityState === "hidden"`), which browsers throttle
  `requestAnimationFrame` for -- d3's force simulation never ticked on
  its own in that environment. This is a real, repeatable testing-
  environment quirk worth remembering for any future d3/rAF-driven
  artifact: `screenshot` timing out and zero rendered elements beyond
  static/synchronous ones is consistent with this, not necessarily a
  code bug.
- Once the server replaced the Artifact, verified end to end through
  the actual Browser pane preview (`preview_start` with the `flux-radar`
  launch config): first run completes and populates the graph, hemisphere
  boundary holds with zero crossings pre- and post-convergence, a second
  `/api/run` with different parameters correctly appends to and can be
  switched to via the run selector, error path and path-traversal
  protection covered by the new test file.
- The `ThreadingHTTPServer` process appears to get restarted by
  `preview_start`/`navigate(force=true)` in this environment (observed:
  the cached `ModelBundle` was gone and GPT-2 reloaded from scratch
  after a `force` navigate) -- worth remembering that a forced browser
  reload isn't guaranteed to hit the same long-lived server process
  when iterating on frontend-only changes; the backend doesn't need to
  restart for a static HTML edit, so this cost real time in this session
  but is a preview-tooling quirk, not a server bug.

## Lessons learned

- **Ask "what am I actually testing" before trusting an automated
  screenshot tool.** A hidden/backgrounded tab silently disables
  rAF-driven code paths; a synchronous/static-only render still
  succeeds, producing a misleading "nothing happens" signal that looks
  like a code bug but isn't. Manually driving the relevant timer/event
  loop (`d3.timerFlush()`) isolated code correctness from environment
  quirk immediately.
- **A soft physics bias is not the same guarantee as a hard constraint,
  and the user's own framing ("hemi-radars fitting in their arc
  segments") was pointing at exactly that gap.** The fix needed two
  layers: a force nudging the simulation toward the right shape over
  time, *and* a hard clamp guaranteeing the constraint on every single
  rendered frame regardless of convergence state -- the first alone
  left a real, visible, intermittent violation (one node 2px past the
  boundary before full convergence).
- **When a user says "let me run new data" during a demo built from a
  static export, the honest answer is usually "turn it into a real
  server," not "add more baked-in datasets."** A dropdown of pre-baked
  runs would have technically satisfied the literal request while
  missing the actual want (a tool they can keep using, not a one-off
  visualization).

## Next Steps

- Spring stiffness/resting-length values found by tuning the live
  sliders haven't been ported back into `graph_layout.py`'s
  `ForceLayout` defaults (`spring_strength=0.02`, `spring_length=60.0`)
  -- that requires the user actually using the tool and reporting back
  numbers that look right, per the plan discussed with them.
  `graph_layout.py`'s own layout is a single up/down-pinned axis with
  no hemisphere split at all; if the hemisphere-split design proves
  useful in the radar tool, the *real* pygame `FluxGraphVisualizer`
  still uses the old design and was not touched this session.
  `graph_layout.py` also can't be verified visually the way this
  server-based approach was, per the existing experience report
  documenting that limitation.
- Backward-direction starvation (real nodes dying out by tick ~3 in
  multiple tested configurations) might be worth a dedicated look if
  the user wants longer, more balanced bidirectional demos out of the
  radar tool -- not investigated further here since it's a pressure-
  dynamics question, not a visualization one.

## Prompt History

- "can I get some visual feedback maybe? we need to see concentric
  designs for parallelism having more rings for more elements, pressure,
  score, I need to see how the physics is working and how the work is
  working... just need to see it all as it's forming, and maybe labels
  and can we make sure the springs are stiff enough and of uniform
  resting length to cluster around their parents"
- "there doesn't seem to be anything orienting forward and backward,
  the backward should really go one way while the forward goes another
  way, two hemi-radars fitting in their arc segments but springed
  around like t his, and give the ability to run new data at different
  parameters to populate the system"
- "I like the interface, I'd like it if we made a program that ran to
  generate more data and offered this up and was the web server, so I
  could start using the system through this web interface stand alone
  as it is"
