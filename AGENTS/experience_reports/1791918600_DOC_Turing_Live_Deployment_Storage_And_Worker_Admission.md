# Turing Live Deployment Storage and Worker Admission

**Date:** 2026-10-13
**Title:** Deployment-state paging, compiler visibility, and elastic worker admission

## Overview

The canonical `C:/dev/Powershell/turing` compiler now adopts deployment and
loop planning state into its existing compressed live-tile arena, retains
object and alias identity, reports an eleventh finalization stage, and exposes
the previously silent compiler pass boundaries through the shared terminal
event stream. Supplied source schedules and auto deployment-region work use
one admission policy. The bootstrap and orbital commands request a default
minimum of twelve ready workers and use 75-percent decoded-front elasticity.

## Steps Taken

- Traced deployment-plan ownership, live-tile conversion, NetworkX promotion,
  route retention, source-list admission, and inner region admission.
- Fixed streaming initial adoption, shared conversion memoization, late graph
  promotion, empty-map route invalidation, and decoded-front physical-headroom
  accounting.
- Added pass-level progress and the compiler-finalization stage to the existing
  progress compositor.
- Fixed stdout ownership across parallel source compiles.
- Added dynamic admission resampling and the `--min-workers` /
  `--minimum-workers` command option to both user-facing compiler commands.
- Ran only targeted, bounded compiler tests and native fixtures.

## Observed Behaviour

The first wrong writer for missing rebuilt loop edges was a stale retained
route after the last key of a tiled mapping was deleted. Reinsertion updated
the obsolete root, so lookup and iteration disagreed. Invalidating that route
at the deletion site restored the authoritative graph relationship without a
downstream repair.

The original four-source torture probe was configured with a 2-GiB aggregate
and per-source budget and zero elasticity. It spent most of its time in the
minimum-footprint regime and was stopped at the user's request. It is not a
throughput result. A separate one-source bounded probe completed three exact
LLVM native comparisons. The latest focused admission/default gate passed 24
tests in 8.80 seconds. The identity audit retained zero structural findings.

The scheduler previously chose a synchronous serial loop whenever its first
capacity sample returned one. It now retains the parallel scheduler, resamples
every 0.5 seconds while the first job runs, and admits more work when capacity
returns. A configured minimum is bounded by ready work and an explicit maximum
but overrides memory-estimated admission.

## Lessons Learned

A pressure test with elasticity deliberately disabled can demonstrate exact
minimum-footprint behavior but cannot characterize normal compiler throughput.
Compiler planning records can be paged safely only at their ownership boundary
while preserving the exact Python objects and shared mappings. Progress needs
to follow those same boundaries; generic heartbeats do not identify the slow
pass. Worker availability must remain a live decision instead of being frozen
by the first sample.

## Next Steps

- Run the user-owned bootstrap and orbital commands with the new default
  twelve-worker floor and 75-percent elasticity, then inspect their durable
  progress and failure receipts.
- Profile the now-visible final collection phase before changing its existing
  conservative O(V*S + E + B) mark/compaction organization.

## Prompt History

> "try to fix it please, make sure you're right that we can evict these as we make them to compressed storage, then we'll let the active cache sort out anything getting used"

> "you need to put an 11th stage in we need eyes on what the fuck happens after everything finishes but it just sits around doing fucking nothing apparent but crawling very very slowly up in memory, we need eyes on that, we also need logging to happen in all the silent stages, before we're ready for me to try a big one again, and then, anything you haven't figured out yet. you do not have to demonstrate 24 overlapping threads because as you pointed out and I pointed out, we can't figure that out. as well as, all the spinning plates in the air have to be landed in working code"

> "no, I told you to put logging into all silent segments, that's a job that requires finding them and then filling out logging"

> "Kill no process, leave any python running alone"

> "please kill it, the clamped down with no elastic is a dire, dire siituation , possibly worse than paging"

> "okay, let's fix this, I want a command line option to set minimal workers, that passes through from the orbital script and from the compiler script, that requests 12 minimum workers and we need to fix the lack of easing back open as you pointed out. also, I don't want that to be 25 I mean I want that 25 value right ther ein the code a 75"
