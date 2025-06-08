**Experience Report: Git LFS Lock-In Caused by Malformed package-lock.json from Automatic Changelog**

**Date:** 2025-06-07
**Prepared by:** Albert Stichka

---

### Summary

During the ongoing development of the `speaktome` project, a critical error was encountered related to Git LFS (Large File Storage) and the automatic changelog system. The issue stemmed from an anomalous state in which a seemingly standard text-based file, `package-lock.json`, was treated as a large binary object by Git LFS. This caused Git push failures, rendering the repository partially unusable and blocking Codex agent integration.

---

### Root Cause

The files `package-lock.json` and `package.json`, normally small and manageable, were registered with Git LFS and reported as missing during pushes. This was **not** due to their size but because:

* Git LFS **still tracked them as pointer files**, expecting large binary blobs.
* The associated LFS blob objects were **missing from the local `.git/lfs/objects` cache**.
* Attempts to push resulted in errors like:

  ```
  Git LFS upload failed: (missing) package-lock.json (SHA...)
  ```

Investigation revealed this LFS lock-in happened **right after the automatic changelog system committed changes** — most likely by a GitHub automation agent or a Codex test runner. There’s a possibility the changelog failed mid-commit or erroneously wrapped these files into LFS.

Further manual inspection of the downloaded files confirmed they were **ordinary JSON files of unremarkable size**. This confirms the error was due to LFS tracking metadata, not file content.

---

### Timeline

* **June 6, 2025**: Codex and GitHub actions began malfunctioning.
* **June 7, 2025**: Git push fails with Git LFS errors on `package-lock.json`.
* **Log Analysis**: Confirmed missing SHA blob for `package-lock.json` and no legitimate size justification.
* **LFS Deactivation Attempted**: Partial success; push still blocked by incomplete LFS object references.
* **Forensic Tracking**: Confirmed the LFS tracking originated from a changelog automation update.
* **Mitigation Deployed**: LFS removed from repo, files manually re-added and sanitized.

---

### Mitigation Steps Taken

1. Removed `.gitattributes` to break LFS tracking.
2. Cleared LFS filters from `.git/config`:

   ```
   git config --unset-all filter.lfs.*
   ```
3. Manually re-added clean versions of `package.json` and `package-lock.json`.
4. Verified via `git lfs ls-files` that no files were tracked.
5. Forced repository push to replace corrupted history.
6. Installed `git-filter-repo` via `pipx` to manage LFS metadata.
7. Encountered permissions issues during cleanup; resolved using `sudo`.
8. **Uninstalled `git-lfs` via `apt`** to eliminate system interference with Git operations.

---

### Lessons Learned

* **Git LFS is stateful** even after removal — pointers in history can silently poison your repo.
* **Automation systems (like changelog agents or Codex hooks) can create structural corruption** if improperly aborted or misconfigured.
* **Don't assume large file errors are about size**; inspect LFS state and pointer integrity.
* **Codex may be impacted** by such failures, especially if running under assumptions of clean `venv` and valid Git state.
* **System-installed tools may conflict** with project-specific expectations — full control requires local, user-managed environments.

---

### Recommendations

* **Audit automation agents** that touch repo state (Codex, GitHub changelogs, etc.).
* **Block LFS tracking of JSON/text files** with `.gitattributes` and Git hooks.
* **Create a pre-push hook to detect orphaned LFS pointers.**
* **Prefer manual changelog updates or sandboxed automation** with LFS entirely disabled.
* **Avoid global LFS installs when local environments suffice.**

---

### Artifact

The corrupted `package-lock.json` and `package.json` files were downloaded and confirmed to be ordinary JSON data of typical size. Archived copies are available for forensic comparison.

---

**End of Report**
