# LOG: a local MCP service that offers the system's verbs as tools

**Date:** 1794378300 (2026-11-11)
**Title:** `speaktome_mcp`: the basic vocabulary as tools, one account per process

## Scope
`speaktome_mcp/` (server.py, rest.py, README.md) and `tests/test_speaktome_mcp.py`. Standard library only. No REST
handler was added or changed; the service calls the existing API.

## Environment
Windows, system Python 3.11. Tests: `python -m pytest --noconftest tests/test_speaktome_mcp.py -q` (the repository's
conftest requires its `.venv`; these tests need nothing from it).

## Commands Run
- The test file above: 21 passed in 10.3 s.
- A real stdio session against the live service as the operator account `claude`: initialize (2025-06-18), tools/list
  (16 tools), then whoami, system_digest, enter, where, go exit 1, go exit 3, look, leave, contacts, pending.

## Log Summary
Every call answered. The walk went from the live canvas to its `market` module and along the `feeds` edge to `worker`,
then closed the cursor. Nothing was written to stderr; stdout carried only JSON lines; the process exited 0.

Tools: whoami, look, enter, where, go, leave, say, pending, acknowledge, contacts, system_digest, system_graph,
system_tools, system_activity, system_instances, diagnostics. One process is one account. The credential is given only
as a file path, read at sign-in, never stored, logged or returned. Text that came from another account is returned
under `untrusted` with its sender, never merged into the tool's own sentence.

Deliberately absent: state-changing verbs (take, use, make do not exist as endpoints yet), a remote transport, more
than one cursor per account. A cursor opened by another process cannot be continued or closed from here.

## Prompt History
Owner: "would it be convenient for you in any way if we now provided the mcp service"

Owner: "yes, let's get it working and continue your experiment."
