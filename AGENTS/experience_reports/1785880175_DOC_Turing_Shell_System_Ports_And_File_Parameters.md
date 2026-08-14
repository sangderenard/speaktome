# Turing shell system ports and file parameters

**Date:** 2026-08-04

## Overview

Work occurred in the adjacent `turing` repository. The existing shell IO
mailbox ABI was extended to describe named file parameters and external
references uniformly across HTML, native Python, and generated C/Fortran
shells. The binary-machine coordinator was renamed to make clear that Python
is its authoring language rather than a packaged interpreter.

## Steps Taken

- Added `SystemPort`, typed fields, directions, and explicit external-reference
  domains to `turing-shell-io-requirements`.
- Added distinct `bundle_references` and `host_references` shell capabilities.
  Web profiles provide only bundle references; native profiles may provide host
  references.
- Added external resolve/call/release request and completion rings to the
  physical `turing-shell-io-abi`.
- Added `u8`/`uint8_t` to the compiled API type vocabulary.
- Made HTML render file ports as byte-exact file inputs rather than numerical
  feed editors, and exposed `window.TuringSystemPorts` for file handlers and
  explicitly registered bundle references.
- Made HTML emission reject host-system external-reference ports.
- Added native CLI file-port parsing and stable contiguous byte/length resources
  to the generated Python native launcher.
- Added generated C file argument parsing, capacity checks, byte loading, and
  bound length delivery for C shells around Fortran.
- Declared the dream simulator's `subject-binary` input as a file system port.
- Renamed `BinaryMachineRuntime` to `BinaryMachineProgram`, retaining a narrow
  compatibility import for older callers.

## Observed Behaviour

- A real generated C shell loaded four exact bytes through
  `--file-subject-binary`, called compiled Fortran with the byte pointer and
  length, and returned the expected calculation.
- The generated dream HTML displayed the `subject-binary` file port and retained
  interior WebGL2 display ownership without a shader error in headless Chrome.
- Focused shell, dream, machine, and shader tests passed: 90 passed.
- Python compilation checks and `git diff --check` passed.

## Lessons Learned

File parameters are initial named resources, while the existing file broker
rings represent later dynamic open/read/write operations. They share byte-span
semantics but should not be conflated.

The external-reference domain must be part of the port identity. This allows
the current HTML policy to mean only another Turing bundle without silently
granting access to host libraries, while leaving stable request/completion
records for the future cross-bundle API project.

## Next Steps

- Define the cross-bundle identity, version negotiation, signature, discovery,
  export-table, and lifecycle rules.
- Teach the compiled browser machine program to register its `subject-binary`
  file handler automatically after lowering.
- Connect guest-binary external references to PE import/export loading and
  native host references to a separately capability-gated host-call policy.

## Prompt History

> what do you mean python machine runtime, you put the python function kicking everything off into the compiler and it works

> Sure, sure, clean this stuff up, keep your eyes out, and let's look at the general shell abi/contracts/etc, try to understand how we offer file parameters in the html shells make sure we have file handlers for native, and then in the shell abi/api get our system ports for external references formalized

> html shell will have external references but only in the sense of other bundles, that's a larger api project i want you to know we need to be prepared for as well while you do this
