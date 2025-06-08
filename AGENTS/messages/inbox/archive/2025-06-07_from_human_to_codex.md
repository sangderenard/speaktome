**Experience Report: Git LFS Lock-In and Forced History Rewrite Due to Automation Conflict**

**Date:** 2025-06-07
**Prepared by:** Albert Stichka

---

### 🔥 Summary

Today’s events revolved around a persistent and severe Git LFS-related issue in the `speaktome` project repository. An automated changelog update or similar automation system mistakenly registered `package.json` and `package-lock.json` as Git LFS objects despite them being small and ordinary. Even after uninstalling Git LFS, deleting `.gitattributes`, and removing large files, GitHub's remote still rejected pushes due to stale LFS pointers. This necessitated a complete repository history rewrite.

---

### 🧊 Root Cause

* Git LFS pointers for `package.json` and `package-lock.json` were injected during an automated changelog update.
* These files were *not* large, but their SHA references pointed to non-existent LFS blobs.
* Uninstalling LFS and removing `.gitattributes` did **not** prevent GitHub from rejecting pushes due to retained LFS history.
* Pushing to GitHub failed repeatedly with `GH008` errors referencing missing LFS objects.

---

### 🛠️ Timeline & Actions

#### Pre-Nuke Phase

* Attempts made to:

  * Remove `.gitattributes`
  * Uninstall LFS (`apt remove git-lfs`)
  * Renormalize (`git add --renormalize .`)
  * Use `git-filter-repo` (installed via `pipx`)
* Result: Git still tried to push references to non-existent LFS objects.

#### Nuclear Option: Full History Rewrite

* Steps:

  1. **Backed up current project files** (excluding `.git`, `node_modules`, `.venv`, etc.).
  2. **Removed `.git/` directory** entirely.
  3. Reinitialized repo: `git init`
  4. Configured default branch: `git config --global init.defaultBranch nogodsnomasters`
  5. Renamed branch: `git branch -m nogodsnomasters`
  6. Added remote: `git remote add origin git@github.com:sangderenard/speaktome`
  7. Made clean initial commit
  8. Pushed fresh: `git push -u origin nogodsnomasters`

* Final push succeeded. Repo was entirely reconstituted without LFS.

---

### 🔍 Observations

* **GitHub UI falsely showed successful push events** despite failed file updates.
* **Git-filter-repo** removed stale blob references but also dropped the `origin` remote.
* Console confusion occurred due to misinterpreting git output as shell commands.
* SSH keys on Windows required manual regeneration and were difficult to locate due to GitHub UI updates.

---

### 💡 Lessons Learned

* LFS **persists in history** even when the current files are clean.
* `git-filter-repo` is superior to `git filter-branch` but must be installed and used precisely.
* Git error messaging does not clearly explain when pushes fail due to **historical LFS refs**.
* Misconfigurations in automation (e.g. changelog writers) can silently poison a repo.
* Copying a clean repo and initializing a new Git history is sometimes the only escape.
* Working in WSL vs. Windows Git can lead to diverging SSH/Git states.

---

### 🔥 Suggested Mitigations

* Block `.json`, `.md`, `.py` etc. from LFS in `.gitattributes`
* Add a `.git/hooks/pre-push` validator to check for LFS pointers
* Do not rely on GitHub UI alone to verify successful pushes
* Use `ssh-add` to confirm agent awareness of keys in cross-environment setups

---

### 🧊 Emotional Toll

The prolonged inconsistency, silent corruption, and tool failures led to significant emotional distress and loss of development time. This highlights the importance of making critical development tooling (like Git and its plugins) **resilient**, **transparent**, and **human-centered**.

---

**End of Report**
