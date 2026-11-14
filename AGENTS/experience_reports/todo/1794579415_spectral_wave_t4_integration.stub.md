# Complete Nodus T4 wave integration

Build a production Nodus row around Spectral's existing fftfree and arena
contracts, using the four SymPy-compiled wave products recorded in
`1794579415_DOC_Compiled_Wave_Kernels_And_Nodus_Row.md`. Preserve f64/f32
precision identity, aperture masks and absorbed power, Jones mapping, and
branch-state continuation. Then test a thick-lens aperture, multi-arena
batching, and sustained device timing. The current 512×512 Nodus trials prove
the four math kernels only; they do not close T4 continuation or renderer
integration.
