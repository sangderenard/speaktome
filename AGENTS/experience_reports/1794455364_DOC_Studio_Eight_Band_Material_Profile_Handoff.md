# Studio eight-band material profile handoff

**Date:** 2026-11-11

## Overview

The Studio photo path now admits an eight-band sensor basis: 650, 550, 450,
590, 365, 395, 850, and 940 nm. The first three preserve the existing RGB
film response. The calibration recipe requests this basis; ordinary photo UI
still selects fixed:3 by default, with fixed:8 available explicitly.

Base Studio light materials now reference the existing Spectral
`led_cool_white` parametric emitter source through the same named material
profile path as shop lights. Filter-off is an explicit saved control for both
base and shop fixtures; its native reference has `filter: null`, preserving the
unmodified material emission spectrum. This is independent of sensor bands.

For the game's optical-media records that author only a subset of wavelengths,
fixed:8 declares `game-transparent-outside-authored-v1`. At an unauthored
frequency the native material band has reflection 0, transmission 1, diffuse
fraction 0, real index 1, and extinction 0. Authored bands use the original
Fresnel and absorption projection. The optical policy appears in geometry
stats and the effective configuration. This rule concerns the material's
optical boundary; separately declared achromatic extinction grids remain
separate scene data.

## Steps Taken

- Added fixed:8 to Turing request validation, Studio UI and JS packer, Spectral
  worker, native emitter adapter, film stack, HDR channels, and NIR marker PNG.
- Added profile-backed base fixture records and filter-off controls without
  replacing the existing material and emitter databases.
- Exported the web bundle and copied 17 changed files to the Development web
  directory, backing up prior copies under
  `turing/build/meowdoku/dev_web_backup_20261111T214715`.
- Authenticated to local Development port 8768 and requested its supervised
  restart. Direct Development page returned HTTP 200 and contained the fixed:8
  option and filter-off controls. The external 8766 route requires the user's
  Development selection; unauthenticated requests there can reach stable.

## Observed Behaviour

- Turing photo/QR tests: 29 passed. Studio scene builder and exposure map Node
  checks passed. Spectral worker, native emitter, named profile, and optical
  media tests: 53 passed.
- Direct native worker on a saved two-camera payload completed fixed:8 at
  512x512 (short sparse exposure) and 128x128 (one-second smoke exposure).
  Both sensor shaders linked and produced PNG/HDR. HDR contained eight ordered
  wavelength planes. The 128px marker was nonzero, but this older saved scene
  predates the full-card invisible QR, so it is not a registration proof.
- `meowdoku_marble_shop_bridge_check.mjs` fails before this change in its VM
  fixture because `lightField` is undefined. The modular light-head check also
  sees pre-existing timed-exposure fields in a strict deep equality. These
  checks were not used as evidence of this change.

## Next Steps

Run the new full-card 850 nm QR calibration through the logged-in external
Development Studio, verify the custom perspective solver registers both
sensor lanes, and show the actual watcher image. The source-level and smoke
exposures do not establish successful chart registration or a calibrated fit.

## Prompt History

> "let's make the default 8 wavelength cover some critical hue angles and the two popular ir bands, any popular uv bands, we might not have much left over"

> "this also means that our filter color NEEDS to be able to be OFF, and the base emitters MUST use a parametric material emission provile"

> "no don't connect the eigh t band default to a parametric emission profile. use the emissive profiles in the material system for material that emits parametrically so it can be lowered to any distribution, for the material of the lights, unles a filter is used"

> "if we can make a spectral profile for materials sure, but otherwise, just let the objects be transparent to higher frequencies we use gaussean sensitivity already, let's be honest abou tthe game material physics"

> "kind of okay? maybe? better to just use the engine as it is where it will already do that kinda? or maybe we don't have it absorptive, it's just that you can't let the material have a boundary or refract"

> "right? if you obey physics and your materials only define so many frequencies, unless you have a conversion algorithm things are transparent, may as well be neutrinos"
