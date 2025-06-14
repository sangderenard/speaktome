## Incident Report

### Autonomous-Agent Interaction with Git & Git LFS — “Codex × Wheelhouse” Case Study

---

### 1. Background

* **Repository topology**

  * *speaktome* (main project)
  * *wheelhouse* (aux repo holding pre-built `.whl` bundles)
* **Agent** – a Codex-style automation that executes PowerShell/Bash build scripts without human confirmation.
* **Large-file policy** – Git LFS was acceptable only for *true* large binaries, **never** for JSON or small wheels (<100 kB).

---

### 2. Trigger Script

The agent executed **`build-wheelhouse.sh`**  &#x20;

* Downloads wheels for Windows & manylinux (↗ `pip download … --platform …`)
* Writes them under `wheelhouse/wheels/{linux,windows}/…`
* Finally **installs “LFS guard hooks”** *if* it finds the `lfsavoider` sub-module — but the guard ran **after** files had already been added to the index.
* Because `.gitattributes` was still carrying

  ```
  *.whl filter=lfs diff=lfs merge=lfs -text
  ```

  every wheel became an **LFS pointer**, even the 25 KB `colorama` wheels.

---

### 3. Failure Cascade

| Time | Autonomous Action                | Hidden Git/LFS Mechanism                                                            | Visible Outcome                                                         |
| ---- | -------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| T₀   | Agent runs *build-wheelhouse*    | `git add wheelhouse/…whl` → Git stores **pointer stubs** instead of real wheels     | Commit appears tiny; push seems fine                                    |
| T₁   | Human clones repo elsewhere      | `git lfs smudge` contacts GitHub LFS                                                | 404 error: object ID not on server                                      |
| T₂   | Attempts to pull future updates  | Every checkout touches the bad stubs                                                | Developers blocked by *“smudge filter lfs failed”*                      |
| T₃   | Manual deletions of `.whl` paths | Index clean, but **old commit** with pointers still in branch                       | `git push` invokes pre-push hook → tries to upload missing blob → fails |
| T₄   | “Nuclear” historical rewrite     | Only option: `git filter-repo --path wheelhouse/wheels --invert-paths` + force-push | Repo finally usable                                                     |

The disaster root: **pointers created, blobs never uploaded, history still references them, LFS hook refuses to push**.

---

### 4. Scientific & Engineering Analysis

| Aspect                          | Observation                                                                         | Engineering Implication                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Autonomy**                    | Agent executed chain of git operations without verifying LFS policy state.          | ⇒ Agents require *pre-flight* repo audits (hooks, attributes, size thresholds).              |
| **Git invariants**              | LFS *pre-push* hook insists every object hash it sees must exist server-side.       | ⇒ Removing a pointer **from HEAD** is not enough; *all reachable commits* must be sanitized. |
| **Pointer toxicity**            | A 72-byte text stub is indistinguishable from a real file in diffs & review UIs.    | ⇒ Humans may approve PRs with poisonous content unwittingly.                                 |
| **Bare vs work-tree rewriting** | Filtering in a *bare mirror* avoids accidental pointer resurrection during rewrite. | ⇒ Keeps working clone intact until clean history is pushed.                                  |
| **History rewrite economics**   | `git filter-repo` finished in seconds; manual pointer triage consumed hours.        | ⇒ For systemic pointer errors, rewriting is cheaper than incremental cleanup.                |
| **Hook timing**                 | “Guard hooks” were installed **after** the damage.                                  | ⇒ Safeguards must run **before first add/commit**; ideally enforced in CI.                   |

---

### 5. Narrow Resolution Path (what finally worked)

1. **Make bare mirror**

   ```
   git clone --mirror repo repo_mirror
   ```
2. **Strip offending paths**

   ```
   git filter-repo --path wheelhouse/wheels --invert-paths
   ```
3. **Verify**
   `git lfs ls-files`  → *no output*
4. **Add minimal `.gitattributes`** (`*.whl binary`)
5. **Force-push via SSH**

   ```
   git remote set-url origin git@github.com:user/wheelhouse.git
   git push --force --all origin
   ```
6. **Team re-clone** shallow or fresh.

Any deviation (e.g., forgetting `--invert-paths` or pushing without `--force`) reproduced the error.

---

### 6. Recommendations

| Layer             | Action                                                                                                                                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent runtime** | 1. Pre-task probe: `git lfs ls-files` must be empty unless policy approves.<br>2. Abort if `.gitattributes` contains `filter=lfs` lines for disallowed patterns.<br>3. Enforce hook installation **before** any `git add`. |
| **Repo policy**   | *Guard commit* at root: `.gitattributes` tracked with `*.whl binary` only; JSON never in LFS.                                                                                                                              |
| **CI**            | Lint job: fail build if `git lfs ls-files` non-empty.                                                                                                                                                                      |
| **Human ops**     | When LFS 404 arises, prefer **filter-repo rewrite** over ad-hoc pointer deletion.                                                                                                                                          |
| **Docs**          | Include “LFS emergency playbook”: environment variable `GIT_LFS_SKIP_SMUDGE=1`, clone shallow, run audit script.                                                                                                           |

---

### 7. Conclusion

This episode exemplifies the **high coupling** between autonomous build agents, Git’s filter architecture, and GitHub’s LFS enforcement. A single misplaced `.gitattributes` line turned harmless 25 KB wheels into long-tailed poison across the DAG.
Scientific lesson: in distributed VCS, **consistency constraints propagate historically**, so remediation must consider the *entire reachable state space*, not just the tip.
Engineering lesson: bake LFS policy checks into the agent’s critical path, and treat history-rewriting tools (`git filter-repo`) as first-line fixes rather than last-ditch nukes.
