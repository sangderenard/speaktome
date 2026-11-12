# Studio eight-band false-colour display

**Date:** 2026-11-11

## Overview

The fixed:8 worker previously saved all eight raw sensor planes but made its
PNG from only the first three. The owner requested compressed wavelength scale
for spectral photographs and several selectable artistic false-colour views.

## Steps Taken

- Added a display-only projection of 365–940 nm into the existing camera
  software's 380–700 nm RGB display palette. Equal-band display white balance
  is explicitly artistic, not CIE colorimetry.
- Added selectable infrared-red, ultraviolet-violet, and visible-reference
  channel views. The original eight planes remain in `linear_spectral_exposure`.
- Connected the default preview, live in-memory retouch, saved-HDR retouch,
  and Development Studio control. Published only the changed JS file to
  `srv/state-dev/web` after a backup in `turing/build/meowdoku/`.
- Checked Development's three saved photo jobs were complete, then used its
  authenticated lifecycle restart on port 8768. A second restart loaded the
  final retouch receipt update. The boot changed from `591320ee8365` to
  `e6d368f800a4`; the page and served JS returned 200.

## Observed Behaviour

- Spectral projection and color-profile tests: 50 passed. Turing photo tests:
  27 passed. Exposure calibration browser-independent Node check passed.
- A mock GL snapshot saved all eight planes and its projected linear RGB.
- An earlier saved 128×128 fixed:8 HDR had nonzero samples in every band;
  all four new projections returned finite RGB from that real sensor data.
  No new full live eight-band exposure was taken during this change.

## Next Steps

Take a logged-in Development fixed:8 photograph, change the false-colour view
after completion, and inspect the actual picture and HDR metadata. Artist
shader effects for UV and IR remain a separate, unimplemented request.

## Prompt History

> "no, do compressed wavelength scale for spectral"

> "Yes but ideally we want to offer several different types of those photographies, false color strategies used by artists, even shader effects for ir and uv"
