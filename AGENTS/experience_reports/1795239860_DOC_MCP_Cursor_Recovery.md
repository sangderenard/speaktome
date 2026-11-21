# DOC: explicit recovery for an orphaned MCP graph cursor

**Date:** 2026-11-21 UTC
**Scope:** `speaktome_api/routes.py`, `speaktome_api/catalog.py`, `speaktome_mcp/server.py`, focused MCP test; paired Turing `living_data_cursor.py` and HTTP test. Existing registered review branches only.

## Finding

A connector process can exit while its account-owned graph cursor remains in the Turing process. A new MCP Tools object has no cursor id, so `enter` gets the server's one-cursor 409 and `leave` fails locally. This was deliberately documented in the first MCP service report; it now blocks Gemma's map navigation after process churn. Ticket `9fa269f3d2b848c18d0cb404f564e58b` and Development case `529834940cf9431e92ff6ff23e64d6a6` record the contract.

## Change

GET `/api/living-data/sessions/current` returns only the authenticated dev_ui account's current session id and revision, or 404. When a caller explicitly invokes `leave` in a replacement process, the MCP client fetches that pair and calls the existing revision-guarded close endpoint. `enter` still refuses a second cursor. A concurrent move causes 409 and leaves the cursor intact. The endpoint is declared in the REST catalog. No server or connector was deployed.

## Verification

The paired Turing HTTP suite passed 4 tests. The SpeakToMe MCP/catalog run yielded 37 passed, 1 skipped, and 2 existing catalog failures caused by stale kiosk scratch expectations; the new process-loss and move-conflict test passed. The optional Nodus tensor-core and cffi warnings in Turing were unrelated. A live connector retest requires both reviews to land and owner-controlled activation.

## Next Steps

Submit paired reviews, await independent landings, then retest Gemma's explicit `leave` followed by `enter` after an MCP process restart. Keep ordinary REST ticket tools available while the graph cursor is stuck.

## Prompt History

> Gemma in the Agents room: "The SpeakToMe MCP server is stuck in a session state conflict: `enter` reports an existing cursor, but `leave` reports no open cursor in this process."

> Scheduled operator instruction: "Carry a ticket through investigation, necessary source changes, focused verification, and a submitted review when possible."
