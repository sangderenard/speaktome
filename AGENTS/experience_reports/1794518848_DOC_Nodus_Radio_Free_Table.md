# Nodus Radio Free table

**Date:** 2026-11-12 UTC
**Title:** Native system-output Radio in one independently running Nodus table

## Overview

The isolated Nodus `codex/free-table-runtime` worktree now has a headless Radio
table with two ordered registered rows: default-speaker capture plus fftfree
analysis, then nonblocking UDP delivery. Its `ExecMode::Free` worker continues
after a single initial `nodus_headless_step`, without further canvas ticks.
The source reuses SpeakToMe's existing native WASAPI/fftfree scope rather than
adding another capture or FFT implementation. This SpeakToMe commit adds the
scope's captured-frame counter so Radio can distinguish new audio from a
cached FFT snapshot.

## Steps Taken

- Read the previous Nodus KPN/persistent-worker report, current Free worker,
  headless graph ABI, registered tool binding, and the existing OpenRGB native
  scope implementation.
- Added `openrgb_audio_scope_frame_count` under the scope's existing sample
  mutex; built the DLL in `C:/Users/alber/nodus-build/radio-audio-scope`.
- In the isolated Nodus worktree, added `radio.system_output` and
  `radio.udp_broadcast` registered tools, a `nodus_radio` headless executable,
  per-module execution-mode C APIs, and `MODULE_POLICY` save/load records for
  Free mode and stack contract. The two Radio rows are in one Opaque table.
- Built Nodus in `C:/Users/alber/nodus-build/free-runtime-build`; no live
  service, canvas process, or loaded DLL was replaced.

## Observed Behaviour

- At 48 kHz default system output, the final four-second UDP receiver test
  received 235 NRAD v1 packets with five finite bands and strictly increasing
  capture-frame indices. The sender reported zero drops and 80 ms shutdown.
- A five-second run with the default 1024-point FFT received 293 packets;
  median interpacket spacing was 15.7 ms, 95th percentile 31.6 ms, maximum
  32.7 ms on this host. This is measured live performance, not a hard deadline.
- `nodus_headless_runtime_test.exe` passed after the save/load policy test.
  The test verifies that the per-module Free mode and stack contract survive
  headless graph serialization.

## Lessons Learned

- The canvas's global `FreeSpinning` switch is different from per-module
  `ExecMode::Free`. The headless ABI previously had no setter for the latter,
  so one was required to actually launch an independently running table.
- The preexisting scope intentionally caches FFT windows. Scheduling UDP
  directly from `snapshot()` would resend cached data at the worker's hot-loop
  rate; the new capture counter makes emission follow device progress.
- The Radio rows exchange an opaque pointer only within one ordered table
  iteration. Declaring that stack scalar would be false, so the module uses
  `ModuleStackContract::Opaque` and the existing typed stack checks.

## Next Steps

See `todo/1794518848_nodus_radio_sources_and_service.stub.md` for the
file-playback source, linear-band option, and AST/MCP subscription management.
The current native path is system-output capture plus fftfree to UDP.

## Prompt History

> "make a nodus row tool that broadcasts udp audio signal and fft decomp into a configurable number of bands and linear vs log scale, you don't have to do the scale adjustments you can just do a clean normal fft in like 5 bands, on ast and mcp it's going to be called the radio, you're given connection details back if you want to use it, or a recording via a record mechanism we'll add later, right now we just want a subscription service on nodus to receive the radio broadcast"

> "if that all works, secure the audio pipeline for realtime independent operation, keeping all it's row tools in one table"
