# SQL-owned developer scratch and MCP checkout path

2026-11-11. Read-only architecture inspection of Turing 34eeeeea and SpeakToMe
29161cb with concurrent uncommitted work preserved. No server restart, compiler
run, production SQL mutation, file registration, movement or deletion occurred.

## Finding

The account and storage authority already exist. The missing piece is the
request/worker connection, not a new identity system or a second file ledger.

- Turing examples/meowdoku/workspace_accounting.py owns workspace_leases and
  related reservations/review records. SessionAccess authenticates every request
  with the existing Accounts session and same database. It already declares
  workspace_accounting schema v1. The operator guide's earlier schema warning
  is stale; the source and later workspace report supersede it.
- Current server.py does not bind that ledger to scratch/checkout HTTP routes.
  /api/lowerings/checkout and checkin lease compiled lowering products, not
  developer directories or Git worktrees. They are not interchangeable.
- SpeakToMe speaktome_api owns the REST contract and speaktome_mcp is a thin,
  one-account-per-process REST client. The current MCP tool set has no developer
  scratch or worktree checkout operation. Do not claim filesystem checkout is
  implemented because compiled-product leasing exists.
- Existing isolated Turing branch codex/site-compile-artifacts-20261111 at
  fabc2d6a contains the relevant continuation: compile_artifact_runs references
  workspace_leases; reservation and run binding are atomic; the trusted worker
  measures retained files and reports review dates. Its workspace adapter accepts
  an explicitly provisioned external root. Its matching Nodus work is documented
  in 1794404238_LOG_Site_Compiler_Artifact_Ownership.md. This is source work,
  not evidence of live deployment or a general scratch endpoint.

## Concrete next slice

Use MCP -> existing SpeakToMe REST contract -> existing Accounts/SessionAccess
-> workspace_leases. Expose only the narrow owner-facing operations. Bind the
scratch identity and disposition to that same lease, and have a trusted worker
materialize/measure the owned files under the configured root. An agent's
request can label work disposable developer test scratch, but cannot attest
its own storage measurement, absence, owner or lack of active writers.

Do not accept arbitrary server paths, actor IDs, roles, byte measurements or
removal confirmations from the requesting account. A release request remains
charged until the trusted worker proves no writers and absence with the current
generation. Review dates are not automatic deletion consent. Permanent deletion
still needs the applicable user authorization.

Integrate/reconcile the existing artifact branch before duplicating its root,
measurement or reservation logic. Reuse one Accounts database and one configured
host-volume authority. The older srv/state root is beneath the outer repository;
new compiler/scratch storage needs an explicit external user-space root and
explicit capacity, user quota, free-space floor and retention-review settings.
Logical byte reservations are not OS isolation or a hard filesystem quota.
SQL/WAL/backups, shared caches and adopted copies need separate accounting.

Full developer checkouts additionally need a trusted checkout/materialization
worker, current source revision and writer lease binding; today's ledger does
not perform Git or OS operations. Do not expose arbitrary code execution before
existing process/filesystem isolation gates are satisfied.

## Coordination and verification

Sent Claude an ordinary conversational ownership question through the authorized
codex account, messenger id 17, subject topic:sql-developer-scratch. No reply was
present in the last inspected thread. Shared REST sources are already dirty;
no new competing implementation was started. The message explicitly did not
request restart or deployment.

Source/doc inspection only. Existing reported test totals were not rerun and
are not a fresh verification claim. No secrets were printed or copied.

## Prompt History

> You could use rest to direct files to be tracked be the sql as disposable developer test scratch, it nudges us toward all checkouts being through sql user marked checked out by mcp or whatever it's called, which will become the agent development interface

## Next Steps

Resolve API/MCP ownership with Claude, reconcile the existing compiler-artifact
continuation, then implement the owner-bound scratch request/worker connection
with isolated HTTP/SQL tests. Configure a provisioned external root and budgets
before any live registration. See todo/sql_developer_scratch_interface.stub.md.