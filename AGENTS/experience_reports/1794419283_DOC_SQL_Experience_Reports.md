# Account-owned SQL experience reports

The new reporting system preserves the original agent guestbook categories
DOC, TTICKET, AUDIT and LOG. It does not reuse the incident-ticket system.
It is source-complete and tested in disposable state, not deployed live.

## Function and original policy

MCP make writes one explicitly authored report through POST /api/reports.
The existing Accounts login supplies its author/owner. Caller identity fields,
file paths and extra metadata fields are rejected. The server supplies id and
creation time. The category body follows the existing guestbook template;
explicit prompt_history strings are stored unchanged. Each supplied next step
is a separate linked SQL row, replacing a file stub for that SQL report. No
prompt, diagnostic, history file or attachment is collected automatically.

A stable caller request_id returns the original receipt on exact retry and
refuses changed content. Report and next-step rows commit atomically. Reports
are immutable and owner-only in this first slice. No public sharing, moderator
bypass, import, execution, edit, deletion, follow-up assignment or completion
workflow has been added. Historical agent profiles remain history, not users.

enter map=reports projects a bounded page of the owner's summaries through the
incumbent graph format and cursor. go, where, look and leave are unchanged.
examine at a report reads the exact SQL record; examine report=ID does so
without a cursor. Authored content is data under untrusted in MCP. Pages are
snapshot summaries; next_offset and category permit bounded navigation.

## Ownership and implementation

Turing examples/meowdoku/experience_reports.py is the sole writer of
experience_reports and experience_report_followups under schema module
experience_reports v1 in the existing Accounts database. It rechecks the actual
session in each transaction. No new identity registry or separate database.

SpeakToMe speaktome_api/reports.py owns report HTTP contract v1. Turing adds
small host/loader bindings without changing dirty messaging routes. Existing
host dev_ui, viewer, cookie, origin, framing and no-store conventions apply.
MCP and manual add only the narrow make/report-map/examine capability. General
make graph editing and the take/use verbs remain unimplemented.

## Verification

Final storage/real HTTP/MCP gate: 16 tests passed in 5.172 seconds in independent
review, including atomic rollback, ownership, session expiry, schema-version
refusal, rate limiting, replay/concurrency, category/page filters, viewer/dev_ui
refusal, malformed framing/connection closure, and production UTF-8 transport.
Existing Turing issue/schema/dependency/cursor/static gate: 29 passed and 10
subtests in 9.27 seconds, with an inherited cffi imp deprecation warning.
SpeakToMe MCP/manual/report-tools/REST gate: 108 passed and 14 subtests in
12.41 seconds. No full-suite, live reporting SQL or browser deployment
acceptance is claimed.

Independent review found ASCII JSON escaping could reject a valid Unicode
report by inflating its wire size. Existing MCP transport now sends normal
UTF-8 JSON; a real Rest/HTTP roundtrip covers that case. Initial new graph-count
expectation and legacy MCP tool-schema expectations were updated to the actual
existing graph semantics and the deliberately added make/array contract.

## Coordination and live boundary

Claude received messages 19 and 20 describing scope and ownership. The Studio
publisher confirmed current live PID 23588 started at 17:37:11Z and all report files
postdate that boot. An in-memory comparison excluding only these new host hunks
exactly reproduced the publisher's frozen server/rest loader hashes; live reports
still returned 404. No reload or production report request was made by this lane.

## Prompt History

> Work on using the established rest vocabulary in order to satisfy reports, the agent environment in speak to me has elements we now want in the sql

> I mean you will be making a new reporting system just on sql based on the original policies

## Next Steps

Coordinate the matching SpeakToMe/Turing source deployment and active-work-safe
service reload before using make against the real account. Then verify one
explicit operational report through REST/MCP. Do not bulk import old guestbook
files or collect optional diagnostics. See todo/sql_experience_reports_release.stub.md.
