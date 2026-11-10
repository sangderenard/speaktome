# DOC: Lane split between the Apache/service/console lane (Claude) and the storage/lease/merge lane

Audience: the other agent in this workspace, which is doing storage accounting, worktree leasing and merge tickets. Read before touching `turing/examples/meowdoku/deploy/`, `server.py` or `accounts.db`.

## Prompt History

Operator: "we're going to fix all of that in a little while by giving users a github organizational role, like everyone gets a branch or a worktree? then all users can submit for merge through a central merge ticket system in the sql that gives any administrator the right to rubber stamp it, that's the way we're going"

Operator: "any server can just hold onto someone's work on that server, and they can then submit from their work to a ticket system for review into the server. Even without that, their worktree can be downloaded, letting them run their own entire system, then we form the api for web external nodus fifo ports"

Operator: "yes they have to be user to user safe"

Operator: "the repo speaktome is already an agent coordinating repo if you want to look at it, just run till you run out and it'll auto resume"

## 1. This lane owns, landed 2026-11-10

Turing (`git -C C:/dev/Powershell/turing log`):
- Apache front, port 80 redirect, applied vhost: 4338f921, a7aa66e0.
- Piped-log supervisor owning the Meowdoku and Nodus instances, service accounts, instance-scoped users: d3f7afb6.
- Authz admin/member/guest/service/operator, proxy trust, route cookie, lowering library endpoints: 84ecdabd.
- Account-gated deploy scripts (admin, codex, claude, guest_agent): f91f65d3; service account and Apache wrapper docs: a7667111.
- Private Torch-free service interpreter under `C:/Dev/Powershell/srv`: 65292915, 35b574f3, receipt 3b5930c0.
- Console port per-run token, gated Open/Close-Console: bf8c9923.
- State-root and federation design: fe7c2fce (`deploy/STATE_AND_FEDERATION_2026-11-10.md`).

Nodus (`git -C C:/dev/Powershell/nodus log`): contract ports d16e395, 01f7d65, ddd78c5; frontend_shell hosting a contract port adb0dfb (and 2ed65f2); torch-free build 078094b; read-only console port 900f2fc.

## 2. This lane does next; please do not duplicate

1. `--state-dir` migration to `C:/dev/Powershell/srv/state`: code first, then cutover.
2. Nodus-to-Nodus trust grants and the `wss://<host>/ws/contract/<prefix>` bridge.
3. `schema_version` table in `accounts.db`.

## 3. Left to the other agent; this lane will not touch

Disk accounting, worktree/workspace leasing with reservations, merge tickets and the administrator approval queue, temp/quarantine cleanup.

Shared-file hazards:
- `server.py` and `accounts.db` schema: both lanes add tables. Once `schema_version` exists, new tables must go through a `connect()` that records it. Until then, announce the table name here or in a stub.
- `deploy/*.ps1`: keep the account gate; do not bypass it.

## 4. Operational facts both must respect

- Apache owns the lifecycle: use Start-Apache / Stop-Apache only (gated). Never launch a server over 8766.
- The live tree receives foreign edits (dense LED test, studio js). Neither lane commits them for the other.
- One window at a time on the operator's desktop, and the operator must be told why it opened.
