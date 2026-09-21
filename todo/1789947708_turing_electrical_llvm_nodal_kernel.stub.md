# Turing electrical LLVM nodal-kernel continuation

- Use `ComplexTensorCircuit.nodal_form()` as the existing topology bake seam.
- Express the fixed-shape complex system as planar real/imaginary
  `Precision[2]` AbstractTensor columns.
- Lower only the numeric nodal solve through
  `src.compiler.fortran_c_shell.lower_ast_source_to_ssa`.
- Prove pivoting, near-singular behavior, and voltage/current parity against
  Torch on the actual four-lane dewar circuit before runtime substitution.
- Retain dynamic graph discovery, empirical laws, batteries, and thermal
  publication in their existing owning systems.
