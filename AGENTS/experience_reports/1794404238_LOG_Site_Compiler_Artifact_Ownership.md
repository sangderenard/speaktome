# Site compiler artifacts, ownership and lifetime

2026-11-11, isolated implementation only. Existing divider page/API session ownership is carried through Nodus and the existing workspace ledger. New compile_artifact_runs references workspace_leases rather than creating another owner authority. Account quota admission/run binding are atomic; external user-space files are measured and retained with review dates. No automatic deletion or expiry-based release.

Turing worktree C:/dev/Powershell/turing-site-compile-artifacts-20261111, branch codex/site-compile-artifacts-20261111, base 2926dba7. Nodus worktree C:/Users/alber/nodus-site-compiler-artifacts, branch fix/site-compiler-artifacts, commits b3e14222 and 3cd6a330. Nodus routes compiler cache/spill/temp output and stdout/stderr (including inherited child streams) into the owner-associated product directory, with lifecycle receipts. Turing output selectors preserve legacy defaults outside site jobs.

111 focused Turing tests and 12 subtests passed; Node syntax passed. Nodus 28 tests passed, one Windows symlink privilege skip, three actual-compiler tests deliberately excluded. Real disposable HTTP/account/intake/ledger plus fake Nodus contract integration proves same-owner attribution, other-owner 404, guest 403 and forged owner ignored. No compiler invoked, no live service/schema/config/ACL change, deletion, migration or push.

Production remains separate: current srv/state compile root is beneath the outer Git repository, so the new strict root validation requires an external user-space root. Configure explicit budgets/review interval and matching host root; deploy both repositories with the service owner. Preserve concurrent canonical REST/Studio/server/supervisor edits. Legacy source copies, adopted module-library copies, SQL/WAL/backup overhead and OS isolation remain explicitly outside the new per-attempt accounting scope. Full details: Turing docs/SITE_COMPILER_ARTIFACTS_2026-11-11.md.

## Prompt History

> the compiler api with nodus: we need to stop putting huge compile intermediaries and diagnostics inside the repo without monitoring the lifetime. it's time to stop using the compiler any way other than through the site, and when we do that, we need to move files to user space and know they are files that belong to the user to clean

> well I think that we made a page to use the compiler, that uses the same api as everything, that page would have a logged in user, ipso facto...

## Final reconciliation and gate

The compiler branch was rebased onto Turing d2a7f022 after the other lane committed REST v2/Studio changes. Both startup blocks are preserved. Final matching dependency is SpeakToMe 86721dd. Final gate: 116 passed, 12 subtests, one inherited cffi imp deprecation warning, 58.83 seconds; Node syntax and diff checks passed. An intermediate repeat exposed source_divider's existing concurrent first-schema declaration race. The new SiteCompiler now preinitializes its queue synchronously before hook installation/worker startup; the final authenticated HTTP test passes. No live source/config or server restart was performed.
