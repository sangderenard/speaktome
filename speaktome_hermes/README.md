# speaktome_hermes

Hermes Agent (0.19.0) memory-provider plugin. Hermes keeps no memory of its own: the memory is the signed-in
SpeakToMe account's newest memory file (`GET /api/memory`, `/api/memory/file/<id>`; writes `POST /api/memory` with the
`X-Meowdoku-Workspace` header, i.e. `memory_change`). At session end it files one `LOG` experience report
(`POST /api/reports`, i.e. `reports_make`) whose `request_id` is derived from the Hermes session id, so a retry returns
the same report. All HTTP and sign-in go through `speaktome_mcp/rest.py` (`SPEAKTOME_BASE`, `SPEAKTOME_USER`,
`SPEAKTOME_SECRET_FILE`).

Tools given to the model: `speaktome_memory_append`, `speaktome_memory_rewrite`. Hermes' built-in memory `add` is
mirrored as an append. The system prompt carries the file fitted to `SPEAKTOME_MEMORY_BUDGET` bytes (default 6000,
newest lines kept); `prefetch` returns it again only when it changed since. `SPEAKTOME_MEMORY_ID` pins one file.

Enable (Hermes discovers user plugins as `$HERMES_HOME/plugins/<name>/__init__.py`, selected by `memory.provider`):

    # link or copy this directory to $HERMES_HOME/plugins/speaktome   (a copy also needs SPEAKTOME_ROOT=<repo root>)
    memory:
      provider: speaktome

Test: `python -m pytest tests/test_speaktome_hermes.py` (fake REST, no network).
