# Site control banner, 2026-11-10

Local edits only: site-nav.js has a named Site controls region, wrapping safe-area layout and menu below it. Opted marble_studio.html, marble_jobs.html, studio.html and log.html into that module. Login, account, server switch and capability logic unchanged. Shared server.py/web_export.py edits preserved. No auth, SQL, live file or deployment changes.

Initial unpublished old diagnostic-report wiring was removed after the user requested replacing the old ticket system. No old ticket link is added. The replacement ticket endpoint/page contract remains missing; workspace_accounting.py is an offline SQL review foundation, not a diagnostic HTTP interface.

Node syntax check passed. Manual tests/site_control_banner_browser_check.py covers guest/operator controls, narrow viewport, menu reopen/close and disabled Development routing. Chrome and Edge failed to produce DOM with crashpad errors in this executor; browser behavior is unverified. No full suite or live publication.

## Prompt History

> and put that ticket button into something with the login thing, a site wide banner control region

## Next steps

Obtain the replacement ticket contract, finish browser QA and coordinate publication. Do not reconnect the old diagnostic flow.


## Completed local SQL incident implementation

The user approved a distinct incident table/API, separate from workspace snapshot reviews. Added issue_tickets.py with schema module issue_tickets v1, owning issue_tickets and issue_ticket_events in Accounts. Owner sessions and SQL admin role are checked transactionally. New /api/issues routes, issues.html/issues.js, game report migration to that API, and the shared banner are source-complete. Existing diagnostic file history is preserved; no live schema or deployment changed. The parallel generic-suite lane owns site-nav registration and abstract_ui_div_map integration.

61 focused issue/auth/workspace/schema tests plus 22 subtests passed. Node syntax and generic suite lifecycle tests passed. Real connected Chrome against disposable Accounts/HTTP fixture verified report, resolution history, shared Report issue navigation, signed-out denial, draft cancel/reopen, account-menu Escape and narrow 390px layout. The fixture was shut down. Three review findings were fixed and independently rechecked: explicit POST connection closing, successful report retry-state reset, bounded 20-event history pages. Full repository, real game runtime and production integration were not tested.

First browser fixture exposed the pre-existing authz initial-schema registration race when concurrent first requests initialize a fresh database; preinitializing the existing authz schema made the disposable fixture stable. New issue schema initialization uses BEGIN IMMEDIATE and is not claimed to fix that unrelated shared declaration race.

Rollout and actual incident submission remain blocked by the existing production network restriction and required coordinated backend publication. No direct live DB write or alternate request route was used. Source/contract/rollout details: Turing docs/ISSUE_TICKETS_2026-11-10.md.

### Additional prompt history

> May I add an issue-ticket table and API to the existing account database, using the same authentication and keeping the workspace-review tickets separate?

> yes, totally, and good eye they are not the same table
