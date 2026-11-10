# Workspace accounting and merge-ticket lane

2026-11-10. This lane adds isolated workspace_accounting.py and focused tests in Turing. It does not edit server.py, state_paths.py, authz.py, deploy scripts, or live databases. Other lane retains state-root migration, schema_version and Nodus trust/transport ownership.

Proposed tables in accounts.db: workspace_volumes, workspace_leases, workspace_review_tickets. Initialization is explicit, only exercised against temporary test databases. Connect/schema-version integration remains coordinated with the state-root lane; no production migration or endpoint activation yet.

Workspaces are server-held mutable state, not one history clone per user. Reservations cover working state and retained submission snapshots. Review approval records a decision; it does not execute code, merge files, grant privileges, or publish to GitHub. Every owner operation binds an authenticated existing users.id; admin review rechecks account_roles in SQL. No LAN bypass. The foundation cannot sandbox arbitrary user code; execution and HTTP integration remain gated.

## Prompt History

> yes they have to be user to user safe

> I’ll take the lanes that are mine, the state root and the Nodus-to-Nodus trust and transport, and leave workspace accounting and merge tickets to the other agent, recorded in speaktome so neither of us duplicates the other. First, disk: the other agent measured 17 GB free, and my builds added to that.

## Next Steps

Coordinate schema_version registration and authenticated route integration before deployment. No credentials or security-sensitive settings are changed here.

## Completed bounded foundation

Added Turing examples/meowdoku/workspace_accounting.py, tests/test_workspace_accounting.py and docs/WORKSPACE_ACCOUNTING_REVIEW_FOUNDATION_2026-11-10.md. All 15 focused offline tests passed (1.620s). No production database initialized and no server endpoint added. Three table names above remain the schema coordination contract. State-root lane must register them with schema_version before integration. Review approval is not merge/application; trusted usage/removal/snapshot receipt inputs must not be exposed to owner-controlled dispatch. OS execution isolation is still required.

Disk changed from 18,739,884,032 free bytes at start to 10,693,423,104 bytes at checkpoint; this lane ran no builds or downloads. Do not admit new materializations using the older free-space snapshot.

## Session/state adapter continuation

With the other lane paused, reread its current landed StatePaths and uncommitted authz/server/federation work; preserved all of it. Added SessionAccess and for_state inside our own workspace_accounting.py. Same Accounts DB is mandatory; actual Accounts.user_for_token reauthenticates every call. for_state consumes content_dir/workspaces without ensure(), ACL changes or folder creation. 22 offline tests pass in 1.968s. No endpoints or live schema initialization. schema_version is still not implemented by the owning lane and remains a deployment gate.

## Bounded review payload materialization continuation

Additional announced table: workspace_review_snapshots, owned by this lane alongside the three tables above. It stores bounded immutable review bundles in SQL. seal_and_submit accepts explicit in-memory file bytes (up to 1 MiB input/128 files), hashes owner/workspace/base plus canonical content and atomically records the bundle, review ticket and encoded-byte accounting. It never reads server paths, walks mutable trees or extracts an archive. There is no HTTP exposure. The bundle is not a complete runnable system export. The prior trusted metadata-only submit remains separate.

Preserved other lane state/auth/server/federation files. Schema-version integration still belongs to that lane and must register all four workspace tables before production use. SQL page/WAL/temporary overhead needs its own safety reserve; logical bundle bytes alone are not the database's physical allocation.

Final review-bundle gate: 30 tests passed (1.768s). Independent review's different-volume DB/root admission issue fixed fail-closed and regression-covered. Owner and current administrator can inspect exact retained submitted bytes; other users cannot. No snapshot methods exposed through SessionAccess/HTTP. SQL/WAL overhead and physical OS quota still require integration, not claimed complete.

## Operator guide and review repaired

Read current Turing deploy/OPERATOR_GUIDE_2026-11-10.md and review 1794338100. Our existing for_state adapter already consumes StatePaths and Accounts; no state-path or other-lane source edited. Shared schema_version.declare now gates every workspace connection before DDL/business writes, with write-lock reacquisition after registration commits. workspace_accounting declares v1. New owned tables announced: workspace_snapshot_retirements and workspace_size_profiles, in addition to prior four.

Closed reviews can now retire payloads through explicit audited retention, keeping ticket metadata. SQL retirement is atomic. External retirement requires trusted verified absence plus absolute post-removal measurement and generation, never a potentially duplicated decrement. All unretired snapshots still block release. Tests prove post-ticket release, retry, denial, rollback and intervening reconciliation. Unknown build peak remains NULL in owner-scoped size profiles rather than zero. Real production disposal is not authorized by this report and was not performed.

37 workspace tests passed in 3.163s on the private service interpreter; existing 3 schema-version tests passed in 0.212s. Other lane live federation proof/state DB left untouched.
