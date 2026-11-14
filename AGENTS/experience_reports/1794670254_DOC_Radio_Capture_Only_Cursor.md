# Radio capture-only cursor for Nodus

The existing `openrgb_audio_scope` DLL coupled WASAPI capture, a PCM ring,
and fftfree analysis. Nodus Radio used its spectral snapshot instead of
feeding the registered `nodus.fftfree` row.

This change adds `openrgb_audio_scope_create_capture` and
`openrgb_audio_scope_read_stereo`. Capture-only construction uses the same
WASAPI worker and PCM ring but loads no FFT DLL and creates no FFT plan.
The reader owns an absolute frame cursor. If the ring overwrites unread
frames, the read advances to the oldest retained frame and reports the exact
number lost. A zero-sample read does not repeat a prior window.

The focused native test feeds known stereo samples, verifies sequential
reads and no replay, then overruns the ring by 4,096 frames and verifies
the reported gap. The old external-feed/snapshot test also passes against
the same new DLL. Both use an isolated user-space CMake build.

Nodus still needs an end-to-end native run through its Free table and
centralized fftfree row to qualify this interface in service.

## Prompt History

> the real fft path is to run the audio in nodus and have the fftfree line row process the feed

> keep those concerns separate, keep fft centralized for batching and always live execution

> split the two lanes into two workers,  you can take one of them or change topic
