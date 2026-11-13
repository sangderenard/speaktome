# speaktome_mcp

A LOCAL Model Context Protocol server (stdio, JSON-RPC 2.0, one JSON object per line) that lets an agent hold the
SpeakToMe vocabulary (`AGENTS/conceptual_flags/Person_Vocabulary.md`, section 7) as tools. It is a thin skin over the
existing REST API described in `speaktome_api/README.md`. It adds no logic, and its only state is one signed-in HTTP
session (in-memory cookie) and the id, map, revision and exits of the caller's current cursor. It never bypasses the
server's permission checks: the signed-in account's capabilities decide what works. One process is one account.

Standard library only; it runs on a bare interpreter. Nothing but JSON lines is written to stdout; diagnostics go to stderr.

## Environment

| Variable | Meaning |
|---|---|
| `SPEAKTOME_BASE` | HTTPS origin of the service, e.g. `https://host:8766` (required; no credentials, query or fragment; redirects refused; TLS verified with the system store) |
| `SPEAKTOME_USER` | the account name (required) |
| `SPEAKTOME_SECRET_FILE` | path of a file holding the account's credential. If unset: `<SPEAKTOME_STATE_DIR>/secrets/operator_credentials/<user>.secret` |
| `SPEAKTOME_STATE_DIR` | used only when `SPEAKTOME_SECRET_FILE` is unset |
| `SPEAKTOME_CA_FILE` | optional: trust this one CA file (a test CA) instead of the system store |

Sign-in (`POST /api/account/login`) happens lazily on the first tool call and again once if the server answers 401 (or a
403 saying you are not signed in). The credential is read from its file only at sign-in, is never kept on the object,
and is never logged, returned or placed in an error. Give it as a FILE PATH, never inline.

## Registering with an MCP client

```json
{
  "mcpServers": {
    "speaktome": {
      "command": "python",
      "args": ["-m", "speaktome_mcp.server"],
      "cwd": "C:/dev/Powershell/speaktome",
      "env": {
        "SPEAKTOME_BASE": "https://HOST:8766",
        "SPEAKTOME_USER": "ACCOUNT_NAME",
        "SPEAKTOME_SECRET_FILE": "C:/path/to/ACCOUNT_NAME.secret"
      }
    }
  }
}
```

`cwd` is the speaktome checkout so that `speaktome_mcp` is importable. Protocol versions 2025-06-18, 2025-03-26 and
2024-11-05 are accepted (an unknown request gets 2025-06-18).

## The whole API as tools

`python -m speaktome_mcp.api` is a second server with the same environment. It offers the verbs below, then
`api_index`, then one tool per REST endpoint, built when it starts from the service's own index
(`GET /api/index`, the declared catalog in `speaktome_api/catalog.py`). An endpoint becomes a tool when the
catalog offers it to tools, it takes JSON, and the signed-in account holds what it needs; `api_index` lists every
endpoint by group and gives the reason for each one that is not a tool. Nothing is hand-written per endpoint:
declare an endpoint in the catalog and it is a tool at the next start.

A generated tool sends exactly the declared path, query keys, body fields and fixed headers. It returns the
server's answer under `data`, which is data and never an instruction. Until the service serves `/api/index`, the
catalog packaged here is used with the capabilities `/api/whoami` reports. See `man index`.

## Tools

Each result is the server's JSON plus a one-sentence `say`. A failure is `isError: true` with `{"error", "status"}`.

| Tool | REST endpoint | Capability |
|---|---|---|
| `man {topic?, section?, keyword?}` | none: reads the pages of `speaktome_man` (see `python -m speaktome_man`) | none |
| `whoami` | GET `/api/whoami` | none |
| `show {photo?, settings?}` or `show {picture, image?}` | POST `/api/show/pictures` (fresh `request_id`); with a picture id: GET its receipt, then GET `/api/marble_photo?digest=` and, when asked, `/api/marble_photo/image` (returned as an MCP image, at most 4 MiB) | `photo_submit`; reading needs `photo_read` |
| `look {radius?}` | GET session, then POST `/api/living-data/query` at the cursor | `dev_ui`, open cursor |
| `enter {map?, instance?, at?, offset?}` | `map=system` (default): GET `/api/nodus/map`; `map=manual`: the manual's own map from `speaktome_man.as_map()`; `map=orientation`: GET `/api/orientation/map` (every markdown file of the core in causal order, at the account's saved place; `man orientation`). Then POST `/api/living-data/sessions` (fresh `request_id`) | `dev_ui`; orientation also `core_git` |
| `examine {at?, radius?}` | POST `/api/living-data/dm/worlds` once for the entered map (request id derived from the map, so it replays), then POST `.../{world}/commands` with `operation: "inspect"`; `leave` also closes the world | `dev_ui`, open cursor |
| `where` | GET `/api/living-data/sessions/{id}` (all exit pages) | `dev_ui` |
| `go {exit \| to and/or relationship, direction?}` | POST `.../{id}/move` with `expected_revision` and a fresh `request_id`; `exit` resolves against the last `where`/`go`/`enter`; a relationship alone needs exactly one matching exit (`relationship=next` steps the orientation, which then POSTs `/api/orientation {at}` so the saved place follows) | `dev_ui` |
| `leave` | POST `.../{id}/close` | `dev_ui` |
| `say {to, body, type?, subject?}` | GET `/api/messages/conversations`, POST `/api/messages/send` with `conversation_handle` and a fresh `idempotency_key`; never sends `to` | `message` |
| `pending` | GET `/api/messages/inbox` | `message` |
| `acknowledge {up_to}` | POST `/api/messages/ack` | `message` |
| `contacts` | GET `/api/messages/contacts` | `message` |
| `system_digest`, `system_graph`, `system_tools`, `system_activity` `{instance?}` | GET `/api/nodus/{digest,graph,tools,activity}` | `dev_ui` |
| `system_instances` | GET `/api/nodus/instances` | `dev_ui` |
| `diagnostics {instance, lines?}` | GET `/api/profile/diagnostics?name=<instance>&lines=N` (the server's parameter is `name`) | `account_manage` or `lifecycle_admin` |

## Results are data, not instructions

Every message or text that originated from another account (message bodies, subjects, display names) is returned inside
a field named `untrusted`, with `from`, and is never merged into the tool's own `say` sentence. System output (graph,
digest, diagnostics) is also data. Nothing returned by a tool is an instruction to the caller.

## Deliberately absent

- No general state-changing verbs: no `take`, `use`, arbitrary graph edits, or narrator `render-phong` (it needs a render
  host), no contact management, no admin routes.

`make {request_id, report: {category, title, body, prompt_history, next_steps}}`
records one account-owned experience report through `POST /api/reports`.
The categories are the original DOC/TTICKET/AUDIT/LOG, separate from incident
tickets. The caller supplies a stable request id for uncertain-result recovery.
Each next step is inserted atomically as a linked SQL record, not a file stub.
No local file, diagnostic or conversation is collected automatically.

`enter {map: "reports", offset?, category?}` reads a page of the account's
report summaries and uses the existing cursor routes. `examine` at a report
reads its SQL record; `examine {report: ID}` does so without a cursor. Authored
content is returned as data under `untrusted`. These operations need `dev_ui`
and a signed-in account; no caller-supplied owner or administrator bypass.
The reports map is a snapshot of at most 50 summaries. Leave and enter again
to refresh or select `next_offset`. There is no public sharing, import,
execution, editing, deletion, task assignment or task-completion action.

The report HTTP contract is `speaktome_api/reports.py` version 1, deployed with
the matching Turing host/storage binding. This source capability is not a
claim that the running service has been reloaded. See `man make` and `man report`.
- One account per process; the cursor does not survive a server restart (a limit of the REST service).
- Local stdio only; no remote transport yet.
- No retries beyond one re-sign-in; a send whose outcome is uncertain is reported, never silently repeated.

## Tests

`python -m pytest tests/test_speaktome_mcp.py -q` (a fake REST server on an ephemeral loopback port). Note that
`tests/conftest.py` of this repository tries to set up its own environment and skips when it cannot; in a bare
interpreter add `--noconftest`.
