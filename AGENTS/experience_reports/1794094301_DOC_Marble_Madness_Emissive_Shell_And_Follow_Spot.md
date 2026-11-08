# Marble Madness emissive shell and follow spot

**Date:** 2026-11-07

## Overview

Marble Madness now has a visible follow-spot switch and controls for the marble exterior's emitted color, strength, and light response (lit, glow only, or both). The changes use the existing Phong/PBR material tables and the existing spotlight emitter.

## Steps Taken

- Added the controls to the Marble Madness panel in `turing/examples/meowdoku/web/index.html`.
- Applied the selected emission to the default glass shell and to the shell/coated exterior of a selected Studio design as its material records are installed. The interior retains its authored records.
- Made the follow spot conditional and positioned its lamp and aim through the case transform so it stays mounted when the case moves.
- Checked `node --check examples/meowdoku/web/app.js` and `git diff --check` for the two changed web files.

## Observed Behaviour

The JavaScript syntax and diff whitespace checks passed. Browser rendering is being checked in the parent task as the related moving-case and plate changes land.

## Lessons Learned

Saved marble designs append their own material records through `installSavedParticle`; overriding only the static catalogue would leave selected Studio shells unchanged. The exterior material ID is provided by `snapshot.design.shell` and the saved material index.

## Next Steps

None for this scoped change; the parent task is integrating and checking the whole Marble Madness page.

## Prompt History

> "make the follow spot able to be toggled and also let the emissive properties of the marble base material be configurable in the marble madness game so users can choose to glow instead of be lit or do both"

> Parent assignment: "Implement latest user request: toggle Marble Madness follow spotlight and configurable emissive properties of marble base material, allowing glow only or lit only or both, using existing shader/material table rather than stand-in."
