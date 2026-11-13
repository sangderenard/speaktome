# Nodus T4 cross-arena batch

Port the existing `wave_t4::angular_spectrum_step_wide` ownership and
state contract to a Nodus row. Group only arenas with compatible geometry
and step parameters, retain each arena's persistent fields and per-slice
progress, and measure the batch. Do not revive the known unsafe fftfree
threading paths. See
`1794573920_DOC_Nodus_BDPT_Side_Record_Pipeline_And_T4_Batch_Boundary.md`.
