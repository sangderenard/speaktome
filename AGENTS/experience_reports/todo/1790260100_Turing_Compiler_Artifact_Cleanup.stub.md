# Turing compiler artifact cleanup

Use `1790260100_AUDIT_Turing_Compiler_Disposable_Artifacts.md` as the cleanup
boundary. Persistent scripts, cited evidence, worktree preservation, and the
native voxel-fluid checkpoint have been moved out of `build/`. The stale build,
cache, log, and media inventory has been removed. After the active Python
processes exit, remove the remaining loaded bootstrap DLL and CFFI cache binary,
the live woodshop log, and current-run identity logs. Preserve retained evidence,
checkpoints, worktree state, and live LLVM pieces.
