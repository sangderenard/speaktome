# Direct FLAC upload for Nodus Radio

**Date:** 2026-11-19 UTC  
**Scope:** Turing Radio intake, browser control, SpeakToMe REST catalogue

## Findings

The jukebox already accepted requests and account-owned Suno imports, but no
direct FLAC bytes. The live `/api/radio` returned `unavailable`, and the
inspected state had no `run/nodus_radio.json`. Two FLAC masters are held by
barnacleboy4 on another volume and were not accessible from this Windows
workspace.

## Changes

The Turing checkout `codex/agent-presence-20261117` adds a bounded binary
`/api/radio/uploads` route and Radio panel file input. It retains one FLAC
original, records owner/title/bytes/SHA-256 in SQLite, and adds validated
records to the shared jukebox catalogue. The full file is decoded through
the existing Nodus audio codec row before publication. Thus upload fails
closed until an operator configures the Radio codec. It does not spend
private-file or Suno per-owner quota. It does not start playback.

This SpeakToMe checkout declares the new REST route and read endpoints.
The binary POST is declared with `request="binary"`; the MCP generator
excludes binary requests because JSON tool arguments cannot carry a 256 MiB
FLAC. The browser can send it directly.

## Verification

Focused Turing tests: 18 passed, covering admission, owner visibility,
incomplete and invalid files, duration ceiling, missing codec, and existing
Suno/jukebox behavior. `node --check` and Python compile passed. No live
FLAC was uploaded and the absent Radio configuration was not supplied.

## Next Steps

After code review, configure the existing file station and Nodus codec,
deploy, and ask barnacleboy4 to upload both masters through the Radio panel.
Compare the returned SHA-256 receipts to their ledger and verify playback.
See `todo/1795114200_radio_direct_flac_deploy.stub.md`.

## Prompt History

> "can you handle this: Checked Nodus Radio directly: the upload ability is not implemented, at least not exposed to accounts."

> "user barnacleboy4 is standing by to try uploading them when the server is ready to accept jukebox direct uploads"

> "jukebox files shouldn't depend on user space data limitatoins but should be limited only by the jukebox streaming library size plus any originals kept around, which, we don't necessarily have to keep, but since these are going to be flac, they're kind of acceptable as is"

> "use the mcp for checking things out do not run git"
