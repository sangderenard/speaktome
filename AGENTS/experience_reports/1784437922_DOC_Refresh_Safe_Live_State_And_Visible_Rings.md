# Documentation Report

**Date:** 1784437922
**Title:** Refresh-safe live state and visible radar rings

## Overview

Fixed Flux Radar browser refreshes that produced repeated aborted-connection
tracebacks and left the UI blank long enough to make its ring guides appear
missing. Also increased the ring contrast so the guides remain distinct from
the dark panel after the state arrives.

## Steps Taken

- Reproduced the refresh path against a mature live session.
- Measured the unfiltered live-state response at roughly 35 MB for a
  30-snapshot, 109-node history.
- Added `latest=1` and `after_tick=N` filters to the live-state endpoint.
- Changed refresh reattachment to request only the latest snapshot.
- Changed polling to request only snapshots newer than the browser's last
  received tick, then merge those snapshots into its rolling window.
- Treated browser-aborted response sockets as normal navigation rather than
  server errors.
- Raised ring stroke contrast, width, and opacity.
- Added focused regression coverage for filtering, routing, disconnects,
  frontend request modes, and ring styling.

## Observed Behaviour

- Direct endpoint and state-filter checks pass.
- Restarted the real server and resumed the saved `the ocean` live graph.
  At tick 40, refresh returned exactly one snapshot (2,021,849 bytes in
  0.283 seconds); an immediately current incremental poll returned no
  snapshots (1,904 bytes in 0.022 seconds).
- A simulated `ConnectionAbortedError` is absorbed and closes the handler
  connection without another response attempt.
- Inline frontend JavaScript parses successfully.
- Python compilation and `git diff --check` pass.
- The repository's normal pytest entry point remains gated before collection
  because the environment is not initialized; its attempted PowerShell setup
  also reports Bash heredoc syntax errors in `setup_env.ps1`.

## Lessons Learned

The rings still existed in the DOM after a delayed load, but their original
grid-colored stroke was too close to the panel background. The more serious
refresh symptom came from serializing and downloading the entire rolling
history on every 400 ms poll. Incremental state transport fixes both the
refresh latency and the expected socket cancellations.

## Next Steps

None required for this change.

## Prompt History

- "this happens when i refresh the browser, and rings aren't showing in the browserui at all anymore"
