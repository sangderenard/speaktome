# Compile machine runtime into web and native shells

Lower the dream simulator's Python `BinaryMachineRuntime` into a browser-owned
Wasm/JavaScript runtime that installs `TuringMachineRuntime.loadBinary`, accepts
external shell ticks, and publishes live `TMSNAP01` generations. Reproduce the
same external-clock and triple-slot ownership ABI in the C/Fortran shell using
native atomics. Preserve the subject binary as runtime machine input; do not
turn its instructions into cards or application SSA.
