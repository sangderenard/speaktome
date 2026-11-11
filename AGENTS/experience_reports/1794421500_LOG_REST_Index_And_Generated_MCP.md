# REST Index And Generated MCP

**Date/Version:** 2026-11-11 (UTC), speaktome `9e359cf`, turing `fcd0606b`

## Overview

The REST interface is now declared once as data (`speaktome_api/catalog.py`), served marked for the caller at
`GET /api/index`, and turned into tools by `python -m speaktome_mcp.api`. The same turn pushed every repository,
checked the Codex lane's uncommitted work before committing it, and added reviewed changes and branch cleaning in
turing (`branch_review.py`, `branch_gate.py`).

## Prompt History

> The other agent is having trouble with being in charge ofa user account with access, if you could coordinate the
> whole repo into a pushed state, do it under the nose of the other agent and check their work, then i want a big
> mcp dynamically built from our rest api and i want branches auto cleaned after a grace period and if at all
> possible we want to somehow lock github down behind the as dmin approval ticket sql putting changes up for peer
> review, waiting accepted after a time out, branch deleted or not ata parameter we need to get this environment
> intuitive and navigable

> oh can we use uh the script github runs after changes, and have it reject if there isn't an admin approval

## What Was Done

- Catalog: 140 endpoints in 13 groups, declared by hand from the handlers (a Sonnet worker read the handlers; the
  entries it could not verify are listed in the session, not hidden). Keys: capability (any-of or all-of), account,
  query, body fields, fixed headers, json or binary request, returns, and `tool` (never offered to an agent:
  sign-in, role grants, loopback operator calls, the browser game table).
- `routes.index_route`: `/api/index[?group=]`, each endpoint marked `allowed`. A description, never a grant.
- `speaktome_mcp/api.py`: the verbs, `api_index`, and one generated tool per endpoint that is offered, takes JSON
  and is allowed. Path variables are percent-encoded; undeclared arguments are refused unless the body is open.
- `index(5)` in the manual.

## Measured

- `tests/test_speaktome_catalog.py`, `tests/test_speaktome_mcp_api.py`: 20 tests pass.
- Live, as the `claude` account against the running service (which does not serve `/api/index` yet, so through the
  packaged catalog with whoami's capabilities): 117 tools, 64 read-only; `messages_conversations`,
  `nodus_instances` and `control_instances` answered.

## Not Done

- `/api/index` and `/api/reviews*` answer 404 until the service is restarted.
- The catalog's fields are declared, not proven: the test checks shape and that each path occurs in the handlers.
  Required-ness of several living-data and issue fields was not verified.
- Binary uploads (`photo_submit`, the divider upload) are catalogued but are not tools.

## Next Steps

- When a handler changes a path, field or capability, change its catalog entry in the same commit.
- A check that walks the handlers' `_require` calls against the catalog would turn "declared" into "checked".
