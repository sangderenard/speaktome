# Environment Setup Scripts: Options and Usage Guide

This document details all options, flags, and usage patterns for the environment setup scripts in this repository. It covers both PowerShell (`setup_env.ps1`, `setup_env_dev.ps1`) and Bash (`setup_env.sh`, `setup_env_dev.sh`) scripts, ensuring parity and clarity for all users.

---

## Table of Contents
- [Overview](#overview)
- [Script List](#script-list)
- [All Options and Flags](#all-options-and-flags)
- [Usage Scenarios](#usage-scenarios)
  - [Basic Interactive Setup](#basic-interactive-setup)
  - [Developer Setup](#developer-setup)
  - [Headless Setup (Non-Interactive)](#headless-setup-non-interactive)
  - [Headless Setup Without Torch](#headless-setup-without-torch)
- [Avoiding Torch and Torch-Dependent Codebases/Groups](#avoiding-torch-and-torch-dependent-codebasesgroups)
- [Examples](#examples)
- [Notes](#notes)

---

## Overview

The environment setup scripts automate the creation of a Python virtual environment, installation of dependencies, and selection of codebases/groups for development or testing. They support both interactive and headless (non-interactive) modes, and can be configured to skip installation of PyTorch and any codebase/group that requires it.

---

## Script List

- **PowerShell:**
  - `setup_env.ps1` — Main environment setup (Windows)
  - `setup_env_dev.ps1` — Developer-oriented setup (Windows)
- **Bash:**
  - `setup_env.sh` — Main environment setup (Linux/macOS)
  - `setup_env_dev.sh` — Developer-oriented setup (Linux/macOS)

Each project root also includes quick-start wrappers `setup_env.sh` and
`setup_env.ps1`. These wrappers change to the repository root and invoke the
installation-choice Python tool in headless mode with default arguments.

---

## All Options and Flags

| Option/Flag         | Both PowerShell & Bash | Description                                                                                 |
|--------------------|-------------------------|-----------------------------------------------------------------------------------------|
| `-NoVenv`          | `-NoVenv`               | Do not create or use a Python virtual environment.                                      |
| `-notorch`         | `-notorch`              | Skip installing PyTorch and any codebase/group that requires it.                        |
| `-Codebases VAL`   | `-Codebases VAL`        | Specify codebases to install/select (comma-separated).                                  |
| `-Groups VAL`      | `-Groups VAL`           | Specify groups to install/select (comma-separated).                                     |
| `-headless`        | `-headless`             | Run in non-interactive mode.                                                            |
| `-FromDev`         | `-FromDev`              | Internal: Indicates script was called from a dev setup script.                          |

> **Both PowerShell and Bash scripts use identical single-dash flags for consistency across operating systems.**

---

## Usage Scenarios

### Basic Interactive Setup

**Windows (PowerShell):**
```powershell
# In repo root - Interactive mode (no defaults selected)
./setup_env.ps1
```

**Linux/macOS (Bash):**
```bash
# In repo root - Interactive mode (no defaults selected)
bash setup_env.sh
```

- Runs in interactive mode with NO codebases/groups pre-selected.
- User must choose from available options defined in `AGENTS/codebase_map.json`.
- Installs torch by default unless `-notorch` is specified.

### Developer Setup

**Windows (PowerShell):**
```powershell
./setup_env_dev.ps1
```

**Linux/macOS (Bash):**
```bash
bash setup_env_dev.sh
```

- Runs standard setup, then presents a developer menu for documentation/tools.
- Accepts all the same flags as the main setup script.

### Headless Setup (Non-Interactive)

**Windows (PowerShell):**
```powershell
./setup_env.ps1 -headless -codebases projecta,projectb -groups groupx
```

**Linux/macOS (Bash):**
```bash
bash setup_env.sh -headless -codebases=projecta,projectb -groups=groupx
```

- No interactive prompts. Must specify codebases and/or groups explicitly.
- Available codebases and groups are defined in `AGENTS/codebase_map.json`.

### Headless Setup Without Torch

**Windows (PowerShell):**
```powershell
./setup_env.ps1 -headless -notorch -codebases projectc,projectd -groups groupy
```

**Linux/macOS (Bash):**
```bash
bash setup_env.sh -headless -notorch -codebases=projectc,projectd -groups=groupy
```

- Skips torch installation and any codebase/group that requires torch.
- Use with `-codebases` or `-groups` to specify what to install instead.

---

## Avoiding Torch and Torch-Dependent Codebases/Groups

- Use the `-notorch` flag to skip torch installation.
 - Pass `-torch` for the CPU build or `-gpu` for the GPU variant. These map to optional groups defined in `pyproject.toml`.
- In headless mode, auto-selection will avoid torch-dependent codebases/groups if `-notorch` is set.
- Interactive menus hide those choices when torch is skipped.

---

## Examples

**Interactive setup (no defaults, user selects from menu):**
```powershell
./setup_env.ps1
```
```bash
bash setup_env.sh
```

**Headless setup with specific codebases and groups:**
```powershell
./setup_env.ps1 -headless -codebases projecta,projectb -groups groupx
```
```bash
bash setup_env.sh -headless -codebases=projecta,projectb -groups=groupx
```

**No venv, no torch, specific codebases and groups:**
```powershell
./setup_env.ps1 -novenv -notorch -codebases projectc,projectd -groups groupy
```
```bash
bash setup_env.sh -novenv -notorch -codebases=projectc,projectd -groups=groupy
```

**Developer setup with specific group:**
```powershell
./setup_env_dev.ps1 -groups groupx
```
```bash
bash setup_env_dev.sh -groups=groupx
```

---

## Notes

- All flags use single-dash format for consistency across both PowerShell and Bash.
- The `-notorch` flag is the authoritative way to avoid torch and torch-dependent codebases/groups.
- For interactive setup, no codebases/groups are pre-selected - user must choose from the menu.
- For headless setup, codebases and/or groups must be specified explicitly.
- Available codebases and groups are defined in `AGENTS/codebase_map.json`.
- The scripts will create a `.venv` directory by default unless `-NoVenv` is used.
- Selections are recorded to the file specified by the `SPEAKTOME_ACTIVE_FILE` environment variable (or a default path).

---

For further help, see the script headers or run with `-?` (PowerShell) or `--help` (Bash).
