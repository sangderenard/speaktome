# Keep Garmin FIT SDK references local

Removed only `DataProcessor/nonuploading resources/SDK/` from the Git index and added the exact root-anchored ignore rule. The previous `nonuploading-resources/` rule used a different spelling and did not match this directory. No license text, application code, other reference material, or local SDK file was changed.

## Verification

- Baseline: SpeakToMe REST worktree commit `91add50378fae73baeefe4505a57007fc9a9b0a3`.
- All 1,973 SDK files were tracked and recoverable in that commit before removal.
- Used `git rm -r --cached` only for the SDK directory.
- Compared SHA-256 hashes before and after: all 1,973 local files remain unchanged.
- Tracked SDK count is now zero; `git check-ignore -v` matches the new rule.
- This removes the directory from this branch's next tree. It does not rewrite Git history or alter other worktrees.
- Application tests are not applicable to this index/ignore-only cleanup. The guestbook filename validator is run without `--apply` so unrelated reports remain untouched.

## Prompt History

User: "oh those files can be removed from the repo that's just a document referencing things"

User: "just gitignore them"

User: "oh your're right remove them from git properly"
