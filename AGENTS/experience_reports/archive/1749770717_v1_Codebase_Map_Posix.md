# Codebase Map Posix Path

**Date/Version:** 1749770717 v1
**Title:** Codebase_Map_Posix

## Prompt History
- "we need to make sure that when this runs as a kind of pyproject.toml mirror, installing things, that it resolves to the actual location of tools. AGENTS/tools might be what it should do, but I'm not sure that's not platform dependent"
- Root `AGENTS.md` guidance to add experience reports and run `AGENTS/validate_guestbook.py`.

## Summary
Adjusted `AGENTS/tools/update_codebase_map.py` to emit POSIX-style paths in `codebase_map.json`. This prevents backslash separators on Windows which previously caused inconsistent paths like `AGENTS\\tools`.

## Steps Taken
1. Modified the script to call `as_posix()` when storing each path.
2. Regenerated `AGENTS/codebase_map.json` using the updated script.
3. Created this report and validated the guestbook.

## Observed Behaviour
- New map shows forward slashes regardless of the operating system, ensuring cross-platform installs can locate `AGENTS/tools` reliably.

## Next Steps
- Verify environment setup scripts correctly interpret POSIX paths on Windows.
