
# Agent Forking and Push Strategy

## Overview

This document defines the auto-forking commit strategy enforced by Git hooks in the `speaktome` development system.
It is designed to ensure safety, auditability, and recoverability for changes made by automated agents or users.

---

## 🔁 Pre-Commit / Pre-Push Hook Logic

### Shared Enforcement Rules:
- Always recursively update submodules before commit or push
- Always sync to Google Cloud Storage (if configured)
- Ensure clean working tree

### Fork-on-Failure Behavior:
When a `commit` or `push` fails due to conflicts or policy:

1. A branch is auto-created using:
   ```
   auto/fork/YYYYMMDD-<commit-hash>
   ```
2. The commit is pushed to this branch.
3. Human maintainers may inspect, approve, and merge.

---

## 🌿 Fork Naming Convention

- `auto/fork/20250614-a1b2c3d`
- Includes timestamp and short commit hash

---

## ✅ Merge Strategy

### Auto Merge
- If a forked commit is strictly additive and non-conflicting, it can be safely rebased.

### Manual Merge
- For divergent history or conflict, humans review and merge with:
  ```bash
  git checkout main
  git merge auto/fork/20250614-a1b2c3d
  ```

---

## 🚫 Agent Restrictions

Automated agents **should not disable or bypass hooks**.

---

## 🔐 GCS Sync

The system supports a `sync-gcs.ps1` and `sync-gcs.sh` for mirroring artifacts to cloud. This must be completed **before** any commit or push is allowed.

---

## 🧩 Compatibility

Supports:
- PowerShell (.ps1)
- Bash (.sh)
- Works on Windows, Linux, Mac

---

## 🧠 Future Plans

- Per-agent fork tagging
- Change logging with diffs
- Epoch-aware metadata merge tools


---

## ☁️ GCS Integration: Ephemeral Installer Strategy

To avoid Git LFS limitations, GCS CLI binaries are not stored in the repository.
Instead, each agent or developer must retrieve the installer using a manifest-based approach.

### ✅ Manifest Location

The following file should list versioned, secure download links:

- `wheelhouse/gcs_installer_manifest.txt`

Example line format:
```
https://dl.google.com/cloudsdk/channels/rapid/downloads/google-cloud-cli-456.0.0-linux-x86_64.tar.gz
```

### 🛠️ Auto-Installer Script Behavior

1. Read `gcs_installer_manifest.txt`
2. Pick the latest stable URL
3. Download the archive to a temporary folder
4. Install (ephemeral; not committed)
5. Use `gcloud` as needed, then discard

### 🔐 Git Hygiene

Installers **must never be committed** to any repository.
Use `.gitignore` or `.gitattributes` to enforce this for:
- `*.tar.gz`, `*.zip`, etc.
- `google-cloud-sdk/` directories

### 🔧 Platform Compatibility

Include links in manifest for:
- Windows (.zip)
- Mac (.tar.gz)
- Linux (.tar.gz)

The hook scripts will detect the OS/shell and trigger the right install method.

---

