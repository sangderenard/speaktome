# Meowdoku game profiles, play links, and Tip Jar rendering

## Observation

The page had one set of module-level visual and particle settings across all games. Tip Jar is served as a Fountain3D game with a `tipjar` marker, so it inherited the Fountain saved-particle picker and could inherit a selected designed marble or the separate marble preview. An empty Tip Jar in a fresh development tab showed no ball. The Fountain3D-to-Tip-Jar transition exposed the inheritance path.

## Change

The existing share serializer now also backs a stored configuration for each selected game. Switching saves the outgoing game's server controls, local controls, top scene and marble controls, camera, and runtime gravity under `meowdoku.gameConfig.<game>`. A new game restores its own record. The logical `tipjar` key stays distinct from the Fountain3D engine key. A `Share game` link uses the existing validated, content-addressed share record, opens an independent playable session, and suppresses the development controls. The older `Share` setup preview and `Share live` watch link retain their separate behaviors.

Tip Jar no longer exposes the saved marble picker or draws the free marble preview. Switching away from a designed particle releases its uploaded mesh and resets the particle look before the next game's configuration is applied. Browser checks showed a clean empty Tip Jar after leaving a designed Fountain3D run, and the saved design restored when returning to Fountain3D. The development play link opened with the configured full marble mesh and compact controls. The share server and page continue to exclude player identity, location, credentials, and session keys.

The first play-link screenshot exposed `ReferenceError: L is not defined` in the existing designed-mesh vertex-array setup. Binding the existing `INSTANCE_LOCATIONS` table there restored visible instanced grains; a reload showed the marble field rendering with no new reference error.

The same browser then exposed a steady-state cost, separate from load time: 4,096 copies of the 2,112-triangle low marble ran around 1 fps. The existing shader already rotates one shared geometry per instance. A 40-triangle distant proxy derived from the saved marble recipe keeps an outer shell and inner dye core; the same 4,096-particle compiled run reached about 30 fps. The detailed full, medium, and low levels remain available when the instance count permits. This proxy does not preserve subpixel ribbons or confetti geometry.

## Limits and next steps

The per-game record covers spec controls and the enumerated page controls in `share.js`; settings outside those inputs need an explicit schema entry to be portable. Tip Jar remains backed by the Fountain3D engine and still exposes its broader controls in the regular development page. The play layout is a presentation boundary; the server's existing access tier remains the authorization boundary. Verify the public URL on another device and review which additional controls belong in each game's public contract.

## Prompt History

> there's state confusion when engines are being used multiple places, we need to start pairing each game with a unique configuration, generalizing the controls at the top of the page, and we need to add a new kind of link, one that's not watch or picture but a share link that shares the non dev portion of a page to someone, with all the configurations on it from the user making the link, so that other people as their own user of the site can enter that page, but not be surrounded by all the dev material settings we're working on cleaning up and compartmentalizing contracts on

> part of that state confusion is there's a black ball in our empty tip jar I don't know what that's about

> cant you have different rotational versions of the same geometry in a shader without it dying? isn't that what we do with the emoji? why is the fountain unresponsive when trying to load a marble mesh

> the marble runs perfectly smooth in the demo in the marble studio

> okay faggot when the problem is not load time I'll be back here to scream at you for not listening to me again
