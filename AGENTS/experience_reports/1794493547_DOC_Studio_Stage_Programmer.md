# Studio stage programmer desk

**Date:** 2026-11-12 UTC

## Work

Added a physical programmer desk to the shared Studio scene builder. Its closed cuboid parts enter the Phong furniture mesh, the captured current Studio scene, and the ray-traced photo scene. A saved `stage-programmer-enabled` lighting-profile control includes or excludes the whole desk; older profiles migrate to hidden. Four mesh keys open the programmer, fire the existing camera Flash, toggle the existing rack All out control, and set the existing shop dimmer to 15%. Five mesh faders operate the existing lighting rack bus dimmers. A centered modal temporarily hosts the existing Spectral N×N camera mixer and lighting rack controls; it restores their original DOM positions on close, preserving their state and handlers.

The desk follows product-table and room bounds. The existing product table and light support/center-crane controls remain their own saved profile settings. Sample pads, arpeggiator, rate, and music-file processing were explicitly deferred until Spectral Analyzer itself is integrated.

## Verification

`node tests/meowdoku_stage_programmer_check.mjs` passed geometry, toggle, and click mapping. `node tests/meowdoku_studio_scene_builder_check.mjs` passed the shared scene geometry and page-delegation checks; its detached test context was updated for the existing photo-scene cache. JavaScript syntax checks passed. `web_export.py --skip-boards` republished the static Studio page. Public DNS GET requests to the Studio HTML, programmer module, and scene builder returned HTTP 200 with the new dialog/module present. No server restart or GPU exposure was needed for this static client and scene-builder change.

## Prompt History

> last thing I want to suggest is we put a little desk somewhere in the studio and put a stage light programmer console in it, make a ui that takes over center screen to work the controls on the machine, which should be built into the mesh so clicking activates them even if you're not zoomed in editing the programmer

> just to give you an idea where this goes, you're going to be making all the stage light control and desk be included or not based on toggles in the controls, like the product stage and if it's been implemented in any way the hoist

> also yes it would be that system, that existing one, plus like uh, arpegiator on the samples like a sample pad a programmer, rate, music file input that gets processed - not something we would add yet we'd wait until we integrated signal analyzer itself

## Next Steps

Integrate Spectral Analyzer before adding sample pads, arpeggiation, rate, or music-file processing to the programmer.
