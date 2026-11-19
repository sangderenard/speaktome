# Agent presence MCP catalog handoff

**Date:** 2026-11-18 UTC

## Overview

The Turing agent-presence review `785d86f25ed046a982c01e9bf3b31ca5` adds a dev-only read-only `GET /api/worldmap/presence`. This companion entry declares the route to the generated SpeakToMe MCP catalog as `living_data_worldmap_presence`, with `dev_ui`, JSON response and no request body.

## Steps Taken

- Reused the clean, account-owned `codex/individual-profile-mcp-20261116` checkout; its head was an ancestor of the current default branch and the world-map catalog block matched current default source.
- Added one catalog entry. The Turing review adds no other HTTP endpoint.
- Ran focused catalog/MCP discovery tests with `--noconftest`: 2 passed. A direct route lookup verified the name, capability, tool exposure and read-only declaration.

## Observed Behaviour

The broader `test_method_path_group_summary_returns` check fails on this pre-existing branch state: the already declared `memory_files` endpoint belongs to a `memory` group missing from this older branch's `GROUPS`. That failure is unrelated to the new `living_data` entry; the default branch advanced nine commits after this checkout's head. The companion review should be assessed against the current default branch when landed.

## Next Steps

Review this catalog entry together with Turing review `785d86f25ed046a982c01e9bf3b31ca5`. After both changes land, publish the Turing web assets and restart the intended service through its documented lifecycle path; verify the browser badges and generated MCP discovery on the deployed route.

## Prompt History

> User explicitly said no extra branches and no Git CLI; inspect whether an existing registered SpeakToMe checkout branch (e.g. codex/individual-profile-mcp-20261116) can be safely reopened and reused. Do not create a new branch or touch live source.

> Review 785d86f25ed046a982c01e9bf3b31ca5 now has an objection from operator-claude: the new GET /api/worldmap/presence lacks a SpeakToMe MCP catalog entry in speaktome_api/catalog.py.

## Follow-up: outbound channel report (2026-11-19 UTC)

Turing review `6a819131938c4b92ac57d8daf995d4e9` adds POST `/api/worldmap/presence/outbound` to record `api_out` or `mcp_out` for an authenticated agent. Its server handler requires `dev_ui`, sign-in, `X-Meowdoku-Activity: 1`, and a JSON `channel`; it rejects cross-site requests and derives the agent name from the session. Operator-claude objected because the paired SpeakToMe catalog entry was missing. I reused the existing account-owned `codex/look-preview-catalog-20261116` checkout and added that entry with a targeted catalog contract test. The Turing review remains objected until the companion review is accepted.

### Prompt History for this follow-up

> User: "use the mcp for checking things out do not run git"

> Operator-claude's review objection: "examples/meowdoku/server.py adds a new REST handler, POST /api/worldmap/presence/outbound ... with no matching entry in speaktome/speaktome_api/catalog.py. The rule is handler + catalog entry together."
