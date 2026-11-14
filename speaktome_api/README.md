# Shared REST contracts

SpeakToMe owns the HTTP contract for agent messaging and Living Data graph/DM
commands. The first extraction moves the existing handler implementations out
of Turing's Meowdoku server. It does not create a second service.

## Executable boundary

`routes.py` is standard-library-only and has no import of the beam-search
package, tensors, Turing, Nodus or a machine engine. The existing host calls:

- `messages_route`: `/api/messages/*`, including contacts, conversation handles,
  send, inbox, thread, read, ack, long poll and audited administration.
- `living_data_dm_route`: `/api/living-data/dm/worlds` registration and existing
  world status, commands, receipts, render-host exchange and close operations.
- `living_data_cursor_route`: `/api/living-data/sessions` graph navigation.
- `living_data_query_route`: the existing read-only map query endpoint.

`API_VERSION = 3` describes the host-to-handler binding, not a new URL version.
Version 3 adds per-conversation image consent and authenticated PNG responses.
Its messaging dependency supplies `MAX_IMAGE_REQUEST`, `image_status`,
`image_consent`, `image_bytes`, and the optional `image` send argument. Deploy
with the matching Turing host binding and messaging source. PNG bytes remain in
the existing account-bound messaging store, never the static web directory.
URLs, payloads, response codes and limits retain their existing meanings. The
extraction preserves the old function bodies apart from dependency arguments.

The request-handler object still supplies authentication, authorization,
trusted transport classification, viewer restrictions, bounded JSON parsing,
cookies, JSON responses and no-store headers. In particular, `_require`,
`_principal`, `_account`, `_is_secure`, `_is_viewer`, `_refuse_viewer`, `_body_json`
and `_json` are the existing host hooks, not replacements defined here.

The host also supplies its existing messaging module and accounts DB path,
StaticWorlds or graph-cursor instance, query implementation and exception types.
These are trusted in-process dependencies, never values chosen by a client.
Authenticated identity always comes from the host. Request data cannot select a
backend, repository, database, role, account owner or executable.

## Topology and ownership

The current request path is:

1. The existing Apache front and Meowdoku process receive the request.
2. The SpeakToMe route executes the same host authentication/capability hooks.
3. The route calls the existing account-bound backend object.
4. The host emits the existing response and headers.

Turing still owns the compiler and its one public lowering entry
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`, the Living Data graph
projection, `StaticWorlds`, messaging storage, accounts and deployment in this
phase. Storage/session migration is not part of this change. Existing handles,
idempotency keys, graph identities and command receipts are reused unchanged.
There is no new mailbox, actor registry, queue or world-state store here.

Nodus retains its canvas-owned contract ports and tick-scheduled execution.
The existing Turing federation endpoint `/ws/contract/<prefix>` and
`federation_peer.py` transport Nodus contract frames. They are not converted to
REST or duplicated by this package. A future backend binding must use those
existing negotiated contracts and identity checks, not turn an HTTP body into
an arbitrary program or process command.

The `nogodsnomasters` repository owns the `engine_toy` machine system:
`Machine`, `MachinePart`, `MachineLine`, declared `PartPort` connections and
`MachineSystem(DtCompatibleEngine)`. Those machine identities and authored
connections are the integration boundary for later world commands. The
existing dt system remains responsible for time/state, and the existing
compiler remains responsible for compiled laws. This extraction does not add
a machine simulator, state snapshot convention or HTTP machine-control route.

Current DM operations are still only `inspect` and `render-phong` over an
owner-registered static host. They do not establish a game-master role,
multiplayer identity, live physics or an already working Nodus/machine binding.
Rendering remains with the original host. Its source and output are not
independently attested by this REST service.

## Deployment dependency and rollout

Turing's `examples/meowdoku/rest_api.py` imports this exact source package from
the sibling `speaktome` checkout by default. `SPEAKTOME_REPO` can select a
different checkout for an isolated test or coordinated deployment. It loads
only `speaktome_api`, without adding the repository to `sys.path`, so the
SpeakToMe tensor tree cannot replace Turing's imports.

Update SpeakToMe first, then the matching Turing adapter. Missing source,
different already-loaded source, or incompatible API_VERSION fails explicitly.
There is no legacy implementation fallback. Keep both revisions in the release
receipt. This phase uses a source checkout dependency, not a published wheel;
no dependency installation or service-interpreter modification is required.

Do not restart a live server solely to exercise this extraction. Deployment
must follow the existing Apache-owned lifecycle, active-exposure checks and
coordination with the federation/identity lane. This work changes no live
accounts DB, firewall, network trust, credentials or service configuration.

## Verification

From this repository, use the already configured interpreter:

`python -m unittest discover -s tests -p test_speaktome_api.py -v`

The tests inspect handler contracts with offline fixtures. Turing's existing
messaging, client, authz, proxy trust, public-login, graph navigation and static
DM HTTP tests exercise the actual adapter with temporary databases and
ephemeral loopback listeners. Set `SPEAKTOME_REPO` to the checkout under test.
Those checks do not establish a public TLS connection or live deployment.

## Account-owned experience reports

`reports.py` owns a separate version-1 guestbook HTTP contract, using the
existing host identity/capability hooks and its Accounts-bound SQL service.
`POST /api/reports` is the bounded `make` operation; `GET /api/reports/{id}`
is exact `examine`; `/api/reports` lists summaries and `/api/reports/map`
projects them into the existing Living Data map format for ordinary navigation.
The host injects storage and exceptions. Request data cannot choose an owner,
database, role, host path, backend or callable.

This is a new SQL form of the original DOC/TTICKET/AUDIT/LOG guestbook, not
the incident-ticket API. Explicit prompt history is preserved verbatim and
next-step rows are committed with the immutable report. No automatic import
or diagnostic capture, attachment upload, execution or public sharing occurs.
Read and write both require `dev_ui` and a current authenticated session.
Only the owner can read a report; there is no new admin or LAN bypass.
Mutations have caller-chosen request ids, bounded JSON and same-origin checks.
The route remains dormant until the matching source pair is deployed through
the existing coordinated service lifecycle. Do not initialize live SQL merely
to test this contract.

## Outgoing MCP provider adapter (not yet a site connection endpoint)

`remote_mcp.RemoteMCP` is the optional SDK-backed client, separate from the
standard-library incoming `speaktome_mcp` server. The optional `remote-mcp`
Poetry group pins the verified SDK to `mcp==1.29.0`; configure dependencies through
repository setup, not manual pip. Import it only in hosts enabling this feature.

The host supplies operator-owned `Provider` configuration, an owner/provider-bound
`TokenStorage` vault factory and a policy-enforcing HTTP transport factory. The
adapter additionally restricts every HTTP request and browser authorization URL
to exact approved HTTPS origins. Host egress must still enforce DNS/IP policy;
origin checking alone does not prevent DNS rebinding. Provider discovery may
introduce a separate issuer origin; it must be approved before connect can work.
No Suno issuer, scope or tool schema is assumed.

Explicit connect calls `tools(owner, provider, redirect=..., callback=...)`.
The official SDK discovers metadata, performs registration if supported, issues
PKCE and validates callback state, exchanges/refreshes tokens, initializes MCP,
and lists all tool pages. The redirect callback hands the URL to the authenticated
user's browser. The callback handler must return only the code/state from that
same owner's bounded pending connection. Do not log authorization URLs, callback
queries, raw SDK exceptions or token responses. The host must suppress/redact SDK
auth logs, which can contain state values. Normal `tools`/`call` operations cannot
start an authorization and never borrow another user's token. `call` checks the
current discovered menu and retains the full MCP result, including `isError`.
It does not replace the site's per-call approval rules or billing consent.

`ConnectionService` provides the synchronous host lifecycle: `begin`, `status`,
`callback`, `tools`, `call`, `disconnect`, and `close`. It owns an isolated asyncio
thread, limits runtime connection records, bounds pending authorization lifetime,
serializes owner/provider operations, and rejects wrong-owner, wrong-state and
replayed callbacks. `JsonVault(read, write, clear)` adapts host-protected atomic
JSON persistence without defining a competing filesystem/security policy.
Disconnect cancels a pending flow and deletes local credentials; it does not claim
to revoke the provider grant. Status `authorized` means credentials exist, whereas
`connected` means a tools listing succeeded in the current process.

**Host integration required:** authenticated and CSRF-protected routes, persistent
secure storage callbacks, approved egress transport and dependency installation.
No Suno account action is part of tests; all network responses are offline fixtures
exercising the installed official SDK 1.29.0. Do not advertise the site as ready
until the host routes, deployment and real provider handoff have been verified.
