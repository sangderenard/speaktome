# Global CSF, Scarcity, And Active-Seed Rhizome

**Date:** 1784474790
**Title:** Conserved waste lifecycle and material-driven growth

## Overview

Completed a conserved fluid lifecycle around the graph's single global CSF bath. Every cousin heart exchanges with that same bath. Factories can explicitly declare waste products that enter CSF. Only the current active seed owns and operates one graph-level rhizome, which cleans non-solvent material from CSF and slowly exudes it into soil as salts. Re-rooting transfers rhizome responsibility without duplicating or deleting stored material.

Growth demand now follows continuous material scarcity: each node compares its required opposite-side ion concentration with a configured target and accumulates sprout or air-root interest in proportion to the shortfall.

## Steps Taken

- Added `MaterialFactory.waste_outputs` and routed declared waste into global CSF.
- Activated conservative default CSF exchange and lymph-return rates.
- Added one graph-level rhizome store and active-seed owner identity.
- Updated seed, restore, and reroot paths so the current anchor owns the rhizome.
- Added CSF-to-rhizome cleanup and rhizome-to-soil salt exudation.
- Preserved rhizome contents and ownership through fluid persistence.
- Published rhizome contents and owner in live snapshots.
- Added browser controls for ion scarcity target, cousin-heart/CSF exchange, lymph return, rhizome cleanup, and soil exudation.
- Added CSF and rhizome telemetry to the heart HUD.
- Replaced binary ion-presence growth demand with fractional concentration scarcity.
- Added focused conservation, cousin-link, waste, persistence, HUD, and configuration regressions.
- Ran Python compilation, JavaScript parsing, focused regressions, and `git diff --check`.
- Restarted the Flux Radar backend.

## Observed Behaviour

- Main and cousin hearts both deposit into and draw from the same `bath` object.
- Declared metabolic waste enters CSF instead of remaining in node circulation.
- The active seed pumps a configured fraction of all non-solvent CSF material into the singular rhizome.
- The rhizome exudes a slower configured fraction into soil; forward ions there remain available to existing root uptake.
- Solvent is not pumped into the rhizome.
- Rerooting changes `rhizome_owner_id` to the new active seed while preserving the store exactly.
- Growth interest scales continuously from zero to full scarcity rather than treating any nonzero ion amount as sufficient.

## Lessons Learned

The browser should contribute geometry and interaction, such as ring proximity, while the backend remains authoritative for conserved material balances. CSF and rhizome stores are graph state, not display state; this prevents refreshes or disconnected clients from creating or destroying matter.

## Next Steps

Decide whether soil salts should remain a backend environmental pool visualized by rings, or whether ring contact should permit explicit bidirectional exchange with that pool. The client should report contact/proximity only; the backend should approve and conserve every transfer.

## Prompt History

> "make sure the lymph/csf is set up, that waste products dump to them, and then let's transition the growth rate decisions to scarcity of material and then we have to decide what the plant does. I think we may need to have the ability to transport in the web client or something, I'm open to your ideas. if the csf ion pumps itself clean, where do those ions go? out the roots as salts and then available to the ring? grow a rhizome to hold ions and pump from the csf into it?"

> "okay that's a plan, make sure the seed cousins are all linked by CSF, it's a global fluid, and we'll make the single active seed responsible for a rhizome"

> "your rg commands have been timing out getting nothing done, I don't know why, doesn't seem to be any error? but it just sits for like 20 minutes"

> "your commands don't do anything, you can't seem to do anything, do I need to do anything to help you"

> "try now, also, the server is running in the background I don't know if that makes a difference, you could also try committing and pushing to see if that calms things down"
