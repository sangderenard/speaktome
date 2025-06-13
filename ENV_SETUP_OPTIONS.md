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
  - [Torch Installation Options](#torch-installation-options)
- [Examples](#examples)
- [Notes](#notes)

---

## Overview

The environment setup scripts automate the creation of a Python virtual environment, installation of dependencies, and selection of codebases/groups for development or testing. Torch is skipped unless requested with `-torch` or `-gpu`.

**Default Behavior:** Torch groups are skipped unless explicitly requested with
`-torch` or `-gpu`. This avoids any network access to the PyTorch wheel index on
systems without connectivity. Pass one of those flags only if torch is truly
needed.

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
installation-choice Python tool with default arguments.

---

## All Options and Flags

| Option/Flag         | Both PowerShell & Bash | Description                                                                                 |
|--------------------|-------------------------|-----------------------------------------------------------------------------------------|
| `-NoVenv`          | `-NoVenv`               | Do not create or use a Python virtual environment.                                      |
| `-Codebases VAL`   | `-Codebases VAL`        | Specify codebases to install/select (comma-separated).                                  |
| `-Groups VAL`      | `-Groups VAL`           | Specify groups to install/select (comma-separated).                                     |
| `-torch`           | `-torch`                | Install the CPU Torch optional dependency group.                                        |
| `-gpu`             | `-gpu`                  | Install the GPU Torch optional dependency group.                                        |
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
- Torch is not installed unless `-torch` or `-gpu` is specified.

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

---

## Torch Installation Options

Pass `-torch` for the CPU build or `-gpu` for the GPU variant. These map to optional groups defined in `pyproject.toml`. When neither flag is provided, torch-related dependencies are skipped.

---

## Examples

**Interactive setup (no defaults, user selects from menu):**
```powershell
./setup_env.ps1
```
```bash
bash setup_env.sh
```

**No venv, CPU torch, specific codebases and groups:**
```powershell
./setup_env.ps1 -novenv -torch -codebases projectc,projectd -groups groupy
```
```bash
bash setup_env.sh -novenv -torch -codebases=projectc,projectd -groups=groupy
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
- For interactive setup, no codebases/groups are pre-selected - user must choose from the menu.
- Available codebases and groups are defined in `AGENTS/codebase_map.json`.
- The scripts will create a `.venv` directory by default unless `-NoVenv` is used.
- Selections are recorded to the file specified by the `SPEAKTOME_ACTIVE_FILE` environment variable (or a default path).

---

For further help, see the script headers or run with `-?` (PowerShell) or `--help` (Bash).
