# Canonical vapor/dust catalog receipt

The existing spectral-analyzer material catalog and MaterialDatabase now carry
six optional participating-medium definitions. Three molecular vapors keep
unknown spectra explicit; three coarse dry-dust references expose a sourced,
labeled geometric-extinction approximation only with explicit caller opt-in.
The real SPH density sampler feeds the existing extinction field; there is no
parallel catalog, substitute fluid solver or optical sprite-alpha shortcut.

94 targeted renderer tests and 2 Turing export tests pass. The existing actual
SPH publication check also matches the independent kernel sum with the new
canonical quartz coefficient, with unchanged state. No GPU or deployment was
performed. The native field only adds attenuation; phase, in-scattering,
Brownian dynamics, live field publication and bowl liquid boundaries remain
separate unfinished integrations.

See spectral-analyzer/docs/20261111-participating-medium-catalog.md for source
URLs, units, model assumptions, exact numbers and commands. Turing only needed
its catalog export regression count/assertions updated; source export already
preserves the optional schema. Keep material grain density distinct from
suspended species mass per carrier volume.
