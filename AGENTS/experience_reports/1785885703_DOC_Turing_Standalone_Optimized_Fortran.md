# Turing standalone optimized Fortran policy

**Date:** 2026-08-04
**Title:** Aggressive generated-Fortran optimization with static GNU runtimes

## Overview

Centralized the generated Fortran toolchain settings used by both Turing's
shared-library compiler and the common Fortran/C native shell. GNU builds now
use high optimization and LTO and produce standalone Windows artifacts without
requiring the compiler's runtime DLL directory on `PATH`.

## Steps Taken

1. Added one compiler-aware toolchain policy rather than application-specific
   fluid build flags.
2. Enabled `-O3`, `-march=native`, `-flto`, loop unrolling, and frame-pointer
   omission for GNU Fortran and its C host.
3. Enabled static `libgfortran` and `libgcc`, plus MinGW `-static` archive
   selection for `libquadmath` and transitive GNU support libraries.
4. Kept `-ffast-math` disabled to preserve SSA IEEE, NaN, infinity, signed-zero,
   and reassociation behavior.
5. Added the `strndup` compatibility symbol required by MSYS2 GCC 16's static
   `libgfortran.a` on MinGW.
6. Removed the compiler directory from the runtime launch environment and
   tested an executable with only Windows `System32` on `PATH`.
7. Added PE import-table regression checks for GNU runtime DLLs.

## Observed Behaviour

- The standalone 96 by 64 MAC-fluid executable compiled successfully and ran
  two frames with only `System32` on `PATH`.
- Its PE imports are exactly `GDI32.dll`, `KERNEL32.dll`, `msvcrt.dll`, and
  `USER32.dll`; it imports no `libgfortran`, `libquadmath`, `libgcc_s`, or
  `libwinpthread` DLL.
- The standalone fluid executable is 1,018,365 bytes.
- The focused standalone/display suite passed 9 tests, and the final
  toolchain/display selection passed 4 tests.

## Lessons Learned

`-static-libgfortran` is a request, not proof of a standalone result. The
selected runtime archive and MinGW CRT must actually link, the executable must
be launched without the toolchain directory masking dependency mistakes, and
the PE import table must be inspected. CPU-native optimization and runtime
standalone status are also separate: `-march=native` removes old-CPU
portability even though it removes redistributable-library requirements.

## Next Steps

None required for the GNU/MinGW policy. Other Fortran compilers intentionally
reject standalone mode until their static-runtime flags are verified.

## Prompt History

> "this didn't get libfortran statically linked what's going on with that, don't do anything just explain it"

> "can you make the fortran settings aggressively optimized and made standalone"
