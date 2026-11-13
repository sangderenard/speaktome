# Nodus Radio HLS stack and browser handoff

**Date:** 2026-11-13 UTC
**Title:** Radio PCM conversion row, persistent FFmpeg exec pipe, Meowdoku HLS listener

## Overview

The existing Nodus exec row gained a declared stream mode in PR #72, merged as
460d566. Nodus Radio then gained a PCM-to-s16le tensor conversion row in PR
#73, merged as 4c31de3. The row uses the same Free table stack: the Radio
source provides actual float PCM, the conversion row places a U8 tensor under
the spectrum pointer, the spectrum row consumes that pointer, and the
existing RTP row keeps its canonical FIFO while `exec.radio.hls` receives the
tensor. There is no second PCM FIFO or Python re-encoding path.

## Evidence

- The isolated native Nodus build succeeded. A five-second WAV through the
  real canvas produced HLS before FFmpeg stdin closed. FFprobe identified
  48 kHz stereo AAC. The same run delivered 938 RTP packets and zero native
  PCM drops with SDL dummy audio.
- Turing PR #1219 adds operator opt-in, a fixed exec declaration, a
  `view_site` capability-gated playlist/segment route through Apache's
  existing HTTPS proxy, and an explicit browser HLS control. The focused
  Radio tests passed 26 cases; a local HTTP route test passed for MIME types,
  fixed names and the access gate. Native browser playback and live Apache
  delivery have not been verified.
- No `run/nodus_radio.json` was present in the inspected stable or development
  state root. No public or server-speaker audio source was started.

## Lessons Learned

- The Radio source and spectrum row already exchange a pointer through one
  Free table. Placing the converted tensor beneath that pointer allows the
  existing spectrum/RTP path and the exec row to share one iteration.
- Meowdoku has modern and legacy state layouts. The first HLS helper assumed
  a `run/` parent; an HTTP test exposed that mistake. It now takes the
  existing `StatePaths` object for all run/content locations.
- FFmpeg's default probe waited for stdin EOF in the open-pipe experiment.
  Declared raw PCM format with `-probesize 32 -analyzeduration 0` produced
  live HLS segments.

## Next Steps

See `todo/1794604800_nodus_radio_hls_deploy.stub.md` for the operator-owned
native binary, catalogue and browser proof. A browser without native HLS still
needs a local player implementation before it can listen in the menu.

## Prompt History

> "proposal: rss or whatever is best at this day and age on apache, gets fed the local udp and serves consumers"

> "can we make an exec row that forms an open pipe for ffmpeg"

> "I don't mind vendoring ffmpeg for this btw, we could even strip it down to single purpose builds for each task"

> "carry on forward I'm excited"

> "counter offer: make that conversion a row and use the stack"
