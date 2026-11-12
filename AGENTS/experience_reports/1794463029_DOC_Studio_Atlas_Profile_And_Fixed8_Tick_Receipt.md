# Studio atlas profile and eight-band calibration receipt — 2026-11-12 UTC

The text baker's Studio REST adapter now reads a saved scene/photo contract from
`spectral-analyzer/configs/studio_ink_atlas_photo.json` (or
`--studio-atlas-contract`). It uploads the requested text mesh as the private
OBJ subject, leaving the slate and emitting ring in the saved fixture scene.
The subject is still entered into the existing ink atlas catalog. This is not
an account-facing baker endpoint or a general object-pack service. The broader
`render_assets.plan_display_scene_bake` already models page, part, token and
rotating conditions; production `plan_ink_atlas_bake` remains text-only and
fixed-view. HTML parsing/layout is not present in this path.

The previous Development QR registration jobs `e706b8fa...` and `264e9c5b...`
completed with all-zero eight-band HDR and marker. The scene prepared 630
positive 850 nm emitting triangles and 1,140 positive 850 nm source roots, so
the authored source was present. A one-sweep 512×512 replay of the same scene
produced 55,388 visible pixels and a nonzero marker. Ten sweeps inside one
held state interval produced zero. Ten separate native Studio tick intervals
produced a nonzero final marker. The worker now splits eight-band timed
intervals at shutter substep boundaries before its existing ExposureBarrier
program; this preserves field preparation and sensor integration order at each
tick. A two-lane, 512×512, ten-tick local replay completed in 51.797 s with
138,741 visible pixels. The repository's QR perspective fitter scored
approximately 1.0 for both main and peanut markers. This costs more than the
failed one-field dispatch, so performance optimization needs matched output
evidence before changing the ordering again.

The logged-in Development page reused the old black job by content digest
until the calibration subject radius changed from 10.0 to 10.1 mm. That
distinct request produced registration job `1e9bcedf01...`, then started a
120-tick watched job `7cd0157d4e...` over 12 simulated seconds, with separate
RGB and peanut lanes. The
current watcher links are:

- https://stichka.duckdns.org:8766/marble_calibration_watch.html?digest=7cd0157d4e7c4ab3f6d261d75c95fc98360fd839bc807a63f91c440d3320f966&lane=0
- https://stichka.duckdns.org:8766/marble_calibration_watch.html?digest=7cd0157d4e7c4ab3f6d261d75c95fc98360fd839bc807a63f91c440d3320f966&lane=1

The watched exposure completed at epoch 120 with 250,100 visible pixels and
120 calibration-history entries in 616.297 wall seconds. The final main and
peanut previews and marker planes are account-scoped assets in the Development
server's `srv/state-dev/content/users/account/3/assets/marble_photos/7cd0157d4e7c4ab3f6d261d75c95fc98360fd839bc807a63f91c440d3320f966/`.
Both previews have legible markers and usable measured chart regions. The main
RGB preview visibly shows the QR as white and pink as well; the requested
invisible infrared-only presentation is therefore not established by this
receipt.

`tests/test_studio_atlas_rest.py` passed 4/4 and
`tests/test_marble_preview_worker_status.py` passed 2/2. A local Studio worker
rendered the text subject “MAKE LIGHT” at 512×512, 0.1 s, one shutter tick,
ISO 100 using the saved baker scene. Files are in
`C:/Users/alber/Pictures/Studio/make-light-20261111/`. The local command path
does not place this render into the account's remote asset library. Chrome's
ChatGPT extension blocked the file chooser upload because file URL access is
disabled; this was not bypassed.

## Next steps

Adapt the existing display-scene planner's part and rotating-condition requests to the Studio
profile/subject contract before claiming a multi-view game-asset service. The
current adapter only schedules fixed-view text. The compound-camera table
preview and the native T5 camera exposure are separate routes; neither is
evidence that the fast Studio sensor already traces through a compound lens.

## Prompt history

Owner: “Let's rework it slightly and let it receive a saved studio profile or
contract instead of the manual scene, then we'll make it an offered service,
and let's break ground on a more serious attempt at making proper asset studies
or game oriented packs for pre rendering any object, though it should retain
the text rendering, it is going to eventually be used to render html documents”

Owner also requested seeing the fast method through thick lens work, working
calibration QR, and a Studio exposure of an inspirational text subject.
