# Review: uncommitted Marble Studio / car-game changes in the live Turing tree

**Date:** 2026-11-10, read-only, by the Apache/service/console lane (Claude).
**Subject:** the other agent's uncommitted edits since turing `b2242dae`
(web/marble_*, web/mechanical_creature_studio_*, javascript_runtime_utilities,
tools/stage|publish|add_studio_*, tests/*.mjs, workspace_accounting.py).

## Parse status

All 37 modified files and every untracked .js/.mjs/.py pass
`node --experimental-default-type=module --check` or `py_compile`. No edit to
server.py, accounts.py, deploy/ or state_paths.py: the lane split holds for
files.

## Findings

(a) Correctness
- `workspace_accounting.py:350-351`: `begin_release` refuses whenever any
  review ticket exists for the lease; `decide` never clears snapshots, so a
  reserved lease that was ever submitted can never be released and its charge
  leaks permanently.
- `workspace_accounting.py:186-189`: `reserve` requires a positive integer;
  an unknown build peak cannot be recorded as unknown, which the operator's
  thread asked for ("an unmeasured build peak must remain unknown, not zero").

(b) Shared-schema rule
- `workspace_accounting.py:93-117, 148` creates `workspace_*` tables in
  accounts.db through its own `sqlite3.connect(...?mode=rw)`, not a shared
  connect(), and records no `schema_version` (docstring at :413 says so).
  `Store.__init__` runs the DDL immediately, so `for_state(<live root>)`
  would create tables in the live accounts.db. Nothing imports it today.

(c) Live-service hazards
- Only `tools/publish_studio_game_layer.py` touches the live tree
  (`build/meowdoku/web`) and posts to https://127.0.0.1:8766 (L28, L302);
  it runs only when invoked. Receipts record `server_restarted: False`.
- `javascript_runtime_utilities.py` changes a generated template
  (`TuringWorldRegistry.appendObjects`); the next product rebuild picks it up.

(d) Tests that need frozen artifacts
- `tests/test_studio_game_entry.py:17` and `tools/stage_studio_game_layer.py:19-33`
  read `build/dot-newer-tire-dynamics-20261110/index.html` and
  `build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json`;
  they fail on a clean checkout.

## Verdict

Safe to leave uncommitted beside the live service. JS, compiler, tools and
tests can be committed; the frozen-artifact tests will not pass elsewhere.
Hold `workspace_accounting.py` until it uses the shared connect() with
`schema_version` (this lane is adding that gate now in
`examples/meowdoku/schema_version.py`) and until the release-after-ticket
leak is resolved. Re-verify the live layer before any publish.

## Prompt History

User: "review changes by the other agent, they are working on marble studio and the kinda unnamed car game"
