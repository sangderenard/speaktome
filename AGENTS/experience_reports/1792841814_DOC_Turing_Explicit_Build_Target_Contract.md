# Turing explicit build-target contract

## Scope

Added one compiler/decompiler/executor target contract so platform facts are
resolved once and passed through repository SSA. This advances the application
product work without treating backend language names as platform identities.

## Receipt

- Extended `NativeTargetContext` into the owning
  `turing.build-target.v1` record: stable identity, architecture, operating
  system, ABI, object formats, LLVM triple, SDK, minimum OS, pointer width,
  byte order, and accepted machine-state SSA dialects.
- Added the closed registry and exact targets `ios-arm64`,
  `android-arm64-v8a`, and `x86-64-portable` alongside the existing Windows
  and internal machine-SSA targets.
- `lower_ast_source_to_ssa(..., build_target=...)` resolves the target before
  source ingestion, passes that exact record through whole-program lowering,
  uses it for retained-native versus repository/machine-state selection, and
  publishes it in `IRModule.metadata['build_target']`.
- `build_target_from_metadata` verifies the module receipt against the owning
  registry before a backend consumes it; a changed triple or ABI is refused.
- Retained-code and implementation selection APIs now accept either the typed
  record or its registered identity.
- LLVM consumes the verified receipt, emits its target triple, and passes the
  triple to the object compiler instead of allowing the host to override it.
- The installed Zig/LLVM lane emitted a real arm64 Mach-O object for
  `ios-arm64` (magic `cffaedfe`) and an ELF object for
  `android-arm64-v8a`. Host-only `-march=native` is not applied to an explicit
  cross target; other conflicting architecture flags are refused.
- Final iOS shared linking is refused by name because it requires the
  iPhoneOS SDK and signing on an Xcode host.

`x86-64-portable` deliberately accepts the decoded
`turing.machine-state-ssa.amd64.v1` dialect and no native object format. One
native binary cannot truthfully span PE/COFF, ELF, and Mach-O ABIs; the portable
lane runs or re-emits the semantic machine-state program instead.

## Validation

- Build-target, retention, and focused LLVM compilation tests: 19 passed.
- The wider build-target, retention, and machine-target run: 45 passed and one
  recorded baseline failure. The failure is
  `test_fortran_and_desktop_glsl_print_the_same_numeric_program`, already
  listed in `TEST_BASELINE_AND_HAZARDS.md` because it expects literal `cos(`
  after the Fortran lane moved to its table implementation.
- `git diff --check` passed with line-ending warnings only.

Pre-existing changes in the baseline receipt, LLVM backend/tests, site bundle,
and unrelated worktree files were preserved.

## Prompt history

> We're going to need ios-arm64 as an explicit build target type for Turing or nodus compiling, some generalized parameter or contract that can be passed to backends or the decompiler/executor

> Makes sure we have super generic run on anything x86 as well
