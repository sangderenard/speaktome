# Nodus dynamic Fortran row adoption

When Turing's compiled-program API exposes rank-known runtime extents and
dynamic output shape provenance, extend Nodus's Fortran artifact importer to
allocate `AbstractTensorHandle` outputs from those extents. Remove the current
explicit dynamic-Fortran refusal only after an end-to-end variable-length
execution test passes.
