# Marble orbit flipbook and AVI receipt

## Prompt History

> in the marble studio, the flipbook animation isn't forcing redraw on each frame and the encoding may work and be slow for 1024x1024x32 or it might silently not work, check the logs maybe?

## Findings

- The live `ca214a82...` photo job used 1024 × 1024, RGB, 32 orbit frames, and 30 seconds per frame. Spectral completed all 32 exposures in 970.875 seconds. The Turing JPEG/AVI step then wrote `orbit.avi` (1,998,460 bytes) by 21:29:47 local time, roughly five minutes later.
- `ffprobe` identifies MJPEG, 1024 × 1024, 32 decoded frames. `ffmpeg` frame hashes are distinct for all 32 frames. Encoding worked, but its old status provided no frame count while it ran.
- The browser changed `<img>.src` every 125 ms. On large PNGs, fetch/decode can miss that cadence, so frames could be skipped before paint.

## Changes

- Marble Studio now paints decoded frames on a canvas, preloads completed orbit frames, and waits for decode before advancing playback. Progressive preview images are discarded from the frame cache as newer previews arrive.
- The Spectral worker now publishes an encoding frame count after each JPEG append. The Turing photo status reports temporary AVI size and treats a worker exit before completion as an error, including exit code 0.
- Exported the updated web files to `turing/build/meowdoku/web`; the static page is served by 8766 and 8768 without a server restart. Python status changes take effect when those server processes next restart; newly spawned workers read the updated Spectral source immediately.

## Verification

- `python -m pytest tests/test_meowdoku_marble_photo.py -q`: 2 passed.
- `node --check examples/meowdoku/web/marble_photo.js` and both Python files compiled.
- HTTP reads on 8766 and 8768 confirmed the exported JavaScript and Studio canvas.
- Browser opened the updated 8768 Studio page. Playback of an already completed job was verified through its AVI and source logic, not yet through a fresh browser session, because starting a duplicate job would replace the 16-minute result.
