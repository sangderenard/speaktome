# Studio atlas REST baker and native T5 scene boundary

## Findings

- The glyph manufacturer is `spectral-analyzer/live_spectral_text_demo.py` plus
  `camera_software/render_assets.py`. The planner already orders glyphs before
  token sequences and full strings, with a shared catalog/object library.
- Turing's `POST /api/marble_photo` admits imported-object snapshots and
  actual authored fixtures. A timed `exposure_state_frames` bundle uses model
  shutter ticks instead of wall-clock completion.
- Studio's scene builder already produces the same `MaterialDatabase` MatBuf
  layout accepted by native T5, and exposes triangle material indices and
  emissive flags. The T5 execution/latch and Studio sensor worker are still
  separate.

## Work completed

- Added `camera_software/studio_atlas_rest.py`: the existing atlas order's
  glyph/slate geometry becomes a validated imported slate object plus authored
  glyph and emissive ring fixtures. It submits an MRP1 timed photo through the
  authenticated account REST paths and reads linear HDR.
- Added `InkAtlasStudioRestRenderer` to the existing text demo, selected by
  `--studio-atlas-url`. The BDPT renderer remains the default. A completed
  Studio frame enters the same sprite, catalog, and object library path, with
  explicit separate shutter and BDPT order times in the receipt.
- Added `camera_designer/studio_t5_scene.py`, extracting the exact triangle,
  spectral material, emitter and frequency arrays for the native tracer from
  an already prepared Studio scene. This is an input boundary, not a claimed
  T5 Studio render.
- Added `tests/test_studio_atlas_rest.py` and
  `docs/STUDIO_INK_ATLAS_REST_BAKER.md`.

## Verification

- `python -m pytest tests/test_render_assets.py tests/test_studio_atlas_rest.py -q`:
  23 passed before the final T5 scene bridge test, then the focused Studio
  suite: 3 passed.
- A real `A` MRP1 payload passed Turing `marble_photo.decode_upload` using
  imported source validation and Spectral `prepare_photo_geometry`: 2,276
  scene triangles and 4,176 prepared source rows. No new GPU render was run.
- A fake authenticated REST completion exercised sprite extraction, catalog
  entry and planner completion.

## Limits and next steps

- The Studio atlas adapter has not been exercised against a logged-in live
  REST server or photographed on a GPU. No performance claim is established.
- Native T5 is not selectable from Studio REST yet. Wire the actual native
  sensor registration, T5 dispatch and HDR/status publication to this shared
  scene boundary. Keep Studio model shutter ticks separate from T5 sampling
  counts, and preserve live/exposure account ownership. The existing official
  T5 performance issue is `0a8749aaa365495c9f320c59f2e2201a`.
- Calibration should eventually choose a sensor sampling budget. A completed
  Studio timed frame is not a proof of image convergence.

## Prompt History

> While you're handling this find the glyph baker from spectral analyzer and make a version of it that uses the studio rest api for ray delivery so we can use the fast system for asset bakes, i remember that asset baker has adequate settings for the bdpt exposure requirements, giving it the studio option enormously dramatically reduces the complexity of an asset pipeline

> It's a smallish ui it takes text and squares i think maybe other things and it will make a table of individual and then up to token and complete works it's for thenodus rasterizer eventually and game directional light sprite baking

> They aren't the same concept of time

> The studio is supposed to have shutter speed settings that don't use the second of ray accumulation already

> We need to get T5 the old system integrated inti the rest api contracts for marble studio so we can serve either algorithm for live or exposures, the calibration system when it's proven will handle how we conceive of the camera settings to things like ray counts needed

> The original tools can all be made into nodus tool rows
