# Physically calibrated marble contacts

The current Marble Madness and Studio presets are bounded effective DEM contact frequencies (150–195, ceiling 200) at `DT=1/360`, not Young moduli. To claim physical glass, metal, or ceramic stiffness, derive geometry-dependent contact stiffness and collision duration, choose an integration/contact method stable at that scale, and validate impact and rolling outcomes. Keep optical roughness independent of contact coefficients. See `AGENTS/experience_reports/1794095474_DOC_Marble_Effective_Contact_Stiffness.md`.
