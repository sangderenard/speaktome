# Exact thick-lens aperture T2/T4 handoff

The canonical four-group lens's exact T2 payload traverses the physical
aperture inside one call, so T1 never checks that segment against the authored
T4 ScaleContext. See
`../1794583458_DOC_Thick_Lens_Wave_Context_And_3D_Kernel.md` and the strict
expected-failure test in `spectral-analyzer/tests/test_wave_t4_pipeline.py`.

Preserve exact conic traversal and create an ordered yield at the physical
aperture region in both directions. T4 must receive the complete wave/Jones,
path-length, wavelength, subpath ID and PDF state; its output must resume the
remaining exact surfaces before T3/T5. No post-lens T4 approximation can prove
the interior aperture interaction. Qualify a real light and sensor pair, then
join T3-produced records to T5 in a Nodus pipeline.
