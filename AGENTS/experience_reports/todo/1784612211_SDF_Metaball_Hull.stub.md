# TODO Stub

**Date:** 1784612211
**Title:** Build the SDF/metaball hull for flux_radar

See `AGENTS/experience_reports/1784612211_DOC_SDF_Metaball_Hull_Plan_Not_Yet_Built.md`
for the full plan. Short version:

- [ ] Check whether `npm run build` (vite) actually works in
      `speaktome/flux_radar/` -- resolve the `dist/` vs `webgl/` build
      output confusion first (server serves from `webgl/webgl_renderer.js`).
- [ ] Replace `rebuildHulls`/the node half of `rebuildNodes` in
      `src/webgl_renderer.ts` with a raymarched SDF/metaball fullscreen
      pass. Keep `rebuildLumens` (edge tubes) as-is.
- [ ] Every visual parameter must come from a real snapshot field: radius
      from volume/pressure, blend softness from hull_permeability, color
      from the full solubles blend (not just dominant), pseudopodia reach
      from forward/backward_growth_interest toward the growing child. No
      procedural noise.
- [ ] Decide a max-blob-count cap and a prioritization rule for which
      nodes render as full blobs if the live graph exceeds it.
- [ ] Benchmark: never measured whether a per-pixel loop over the blob
      cap is cheap enough at the demo's canvas size.
