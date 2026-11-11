# LOG: a manual for the vocabulary, one page per verb

**Date:** 1794402124 (2026-11-11)
**Title:** `speaktome_man`: man pages for the verbs, the formats and the concepts

## Scope
`speaktome_man/` (the reader, `__main__.py`, 25 pages under `pages/`), the `man` tool in `speaktome_mcp/server.py`,
`tests/test_speaktome_man.py`. Standard library only. No REST handler was added or changed.

## Environment
Windows, system Python 3.11. Tests: `python -m pytest --noconftest tests/test_speaktome_man.py
tests/test_speaktome_mcp.py -q`.

## Commands Run
- The two test files above: 82 passed in 10.5 s.
- `python -m speaktome_man`, `python -m speaktome_man go`, `python -m speaktome_man -k cursor`.
- A real stdio session against the live service as the operator account `claude`: initialize, tools/list (17 tools),
  man (index), man speaktome, man go, man keyword=cursor, man entr (refused, nearest: enter, entity), whoami, enter,
  where, go exit 1, leave. Every call answered; stderr empty; exit 0.

## Log Summary
The manual has three sections, numbered as on Unix: 1 verbs, 5 formats, 7 concepts.

- Section 1 (18 pages, 22 names): man, whoami, enter, where, go, look, leave, say, pending, acknowledge, contacts, the
  five system readings (one page), diagnostics, examine, show, take, use, make.
- Section 5: map, message.
- Section 7: speaktome (the introduction), person, entity (also found as cursor), narrator, roles (also capabilities).

A page is a plain text file written as `man` shows it. Its NAME paragraph gives its names and one-line summary, which
is all the reader parses. A verb's page has fixed parts: SYNOPSIS (tool form and REST form), DESCRIPTION, IN A STORY
(the sentence a player would say), NEEDS (the capability), ANSWER, REFUSALS (the server's own status and message),
STATUS and SEE ALSO.

Every stated limit and refusal was read from the service source as it stood today: `speaktome_api/routes.py`, and in
turing `examples/meowdoku/living_data_cursor.py`, `living_data_query.py`, `living_data_static.py`, `messaging.py` and
`authz.py`. The pages are not generated from that source, so a change there will not change a page.

What the tests hold in place: every page is well formed, ASCII and at most 78 columns; every SEE ALSO reference
resolves; every tool has a verb page; a verb page states a tool form exactly when a tool exists; the planned verbs
(take, use, make) say Planned and are not tools.

Honest limits:
- examine and show are documented from the handler and the static world source, not from a run. No tool offers them.
- The server does not serve the manual. A page on the site, or a `GET /api/man/...` route, would be a route added in
  `speaktome_api/routes.py`; it was not asked for and was not built.
- The roles page names the capabilities the verbs need. The full table of which role and which network origin grants
  what is in turing's `authz.py` and is not repeated.

## Second pass, same day: the manual in the system's own vocabulary
The owner corrected the first pass: the man pages were meant in the vocabulary of the system. The first pass was a
Unix-style reader beside the system. The manual is now also a place in it.

- `speaktome_man.as_map()` gives the manual as an `abstract-ui-div-map-v0` map, in the same form the service gives a
  Nodus canvas (flat `regions`, relationships as `tracks`). The manual is the root; a section is a place contained by
  it; a page is a place contained by its section; each SEE ALSO reference is a `see-also` track. Each page is also an
  object of `world.objects` under the same identity, and the object's record holds the page text.
- `enter map=manual` enters it through the existing cursor endpoint. `where` and `go` walk it unchanged.
- `examine` is a new tool over the existing narrator endpoints: it registers the entered map as the account's world
  once, sends `inspect`, and `leave` closes the world. The registration's request id is derived from the map, so the
  same map replays to the same world. On the manual the record is the page; on the Nodus canvas it is the place's
  kept fields (identity, kind, name).
- Nothing was added to the server.

Run against the live service as `claude`: enter map=manual (3 exits: Verbs, Formats, Concepts), go to Verbs (19
exits), go to the page `go` (9 exits: its section and its see-also edges both ways), examine (the page text came back
from the server in `object.text`), examine a section (kept fields only), leave; then enter the Nodus canvas, go to
`market`, examine, leave. All answered; stderr empty. This is also the first run of `inspect` from this lane, which
the first pass had documented from source only. Tests: 85 passed.

Still true: `show` is documented from source only and has no tool; the `man` tool and `python -m speaktome_man`
remain as shortcuts that read the same pages without the server.

## Prompt History
Owner: "hey let's also take a crack at man pages"

Owner: "in the vocabulary of the system i mean"
