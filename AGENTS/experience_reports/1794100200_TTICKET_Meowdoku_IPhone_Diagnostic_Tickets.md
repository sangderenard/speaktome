# Meowdoku iPhone diagnostic tickets

**Date:** 2026-11-07
**Title:** Capture the unresolved oil-toy force from the affected phone

## Environment

Canonical Turing repository `C:/dev/Powershell/turing`; live HTTPS Python server
on port 8766. Existing `/api/clientlog`, persisted `/api/log` ring and browser
`local_log.js` were reused. The site's play/dev/edit access tiers remain intact.

## Steps to Reproduce

1. On an affected iPhone, open the live site and run Fountain or Fountain3D with
   the oil and glycerine toy chamber.
2. When grains rush around the perimeter or pile into corners as shown in the
   user's supplied photo, tap **Report issue**, add a short note, and send.
3. On an edit/dev-tier device, open **Tickets** and inspect the new ticket's
   compute inputs, particle positions, browser log, and server log sequence.

## Logs and Output

The preexisting server log had periodic `fountain-diag` entries but no linked
one-time client snapshot for the user's failure. The supplied photo shows
grains at the upper-left and lower-right walls and at the basin edge; it does
not establish the force's cause. Live browser ticket `1794100068-6d88ea9738`
proved capture of a running Fountain session: 4,096 ids, 670 live particles,
21 position samples, 100 local log lines, compute diagnostic values, and a
matching server game. It is a resolved smoke test, not a reproduction of the
iPhone failure. Open ticket `1794100166-7b1daa6134` records the user's report
and photo path while awaiting a device capture.

## Attempted Fixes

No toy force or geometry code changed. `tickets.py` persists bounded reports
under the live build directory. `/api/tickets` accepts play-tier submissions;
dev/edit may read; edit may update status and handling notes. Viewer links and
shader tier cannot submit or read tickets. `tickets.html` presents a summary
and collapsible full state. The game page has an in-page Report issue form.
The capture preserves nonfinite numeric values as strings and includes
runtime law inputs, diagnostics, phone sensor state, selected settings, a
position sample and the existing local log.

Python syntax, Node syntax, store roundtrip, live API submit/read/update and
live browser submit/read passed. Static export used `--skip-boards`; no Wasm
build occurred. Live 8766 was restarted with the previous arguments.

## Current Status

Ticket flow is live. The force regression is unresolved. Capture its actual
iPhone state, then compare it with the persisted log ring before changing the
toy physics. See the matching `todo/` stub.

## Prompt History

> Check logging we were trying to have client and server logging running, we should be able to use it to let me submit state logs and general logs through the server connection as actual tickets we can make a ticket handling dev page and get it from the iphones's mouth

> No, as I've repeatedly said faggot it's not gravity

> [User supplied iPhone photo of the oil toy with grains packed against multiple boundaries.]
