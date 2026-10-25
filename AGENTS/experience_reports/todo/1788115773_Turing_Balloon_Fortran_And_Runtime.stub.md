# Turing balloon Fortran parity and runtime qualification

- Implement interoperable Fortran `PointerArray` handling or direct native
  stack/cat lowering.
- Fill Fortran coverage for imported `llvm.memcpy`, `index_assign_double`, and
  `matmul_double` calls used by the balloon closure.
- Bind and execute the complete C/LLVM balloon artifact through the canonical
  vehicle runtime and qualification/drive loop, then compare exact outputs.

See `1788115773_DOC_Turing_Balloon_Native_Helper_Linking.md`.
