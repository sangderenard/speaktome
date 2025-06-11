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

---

## All Options and Flags

| Option/Flag         | PowerShell Param         | Bash Flag           | Description                                                                                 |
|--------------------|-------------------------|---------------------|---------------------------------------------------------------------------------------------|
| `-NoVenv`          | `-NoVenv`               | `--no-venv`         | Do not create or use a Python virtual environment.                                           |
| `-NoTorch`         | `-NoTorch`              | `--notorch`         | Skip installing PyTorch and any codebase/group that requires it.                             |
| `-Codebases VAL`   | `-Codebases VAL`        | `--codebases=VAL`   | Specify codebases to install/select (comma-separated or repeated).                           |
| `-Groups VAL`      | `-Groups VAL`           | `--groups=VAL`      | Specify groups to install/select (comma-separated or repeated).                              |
| `-headless`        | `-headless`             | `--headless`        | Run in non-interactive mode (auto-select codebases/groups if possible).                      |
| `-FromDev`         | `-FromDev`              | `--from-dev`        | Internal: Indicates script was called from a dev setup script.                               |
| `-ml`              | `-ml`                   | `--ml`              | Install full ML extras (transformers, torch_geometric, etc).                                 |
| `-gpu`             | `-gpu`                  | `--gpu`             | Force GPU-enabled torch install (if not using -NoTorch).                                     |
| `-prefetch`        | `-prefetch`             | `--prefetch`        | Prefetch models or data (if supported by codebases/groups).                                  |
| `-NoExtras`        | `-NoExtras`             | `--noextras`        | Minimal install, skip optional extras.                                                       |
| `-Extras`          | `-Extras`               | `--extras`          | Install all extras.                                                                         |

> **PowerShell scripts only support single-dash flags (e.g., `-NoTorch`, `-NoVenv`, `-Codebases`). Do not use double-dash flags with PowerShell scripts.**
> **Bash scripts only support double-dash GNU-style flags (e.g., `--notorch`, `--no-venv`, `--codebases`). Do not use single-dash flags with Bash scripts.**

---

## Usage Scenarios

### Basic Interactive Setup

**Windows (PowerShell):**
```powershell
# In repo root
./setup_env.ps1
```

**Linux/macOS (Bash):**
```bash
# In repo root
bash setup_env.sh
```

- Prompts for codebase/group selection unless run with `-headless`/`--headless` or explicit `-Codebases`/`--codebases`/`-Groups`/`--groups`.
- Installs torch by default unless `-NoTorch`/`--notorch` is specified.

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
./setup_env.ps1 -headless -Codebases projectA,projectB
```

**Linux/macOS (Bash):**
```bash
bash setup_env.sh --headless --codebases=projectA,projectB
```

- No interactive prompts. Codebases/groups are auto-selected or provided via flags.

### Headless Setup Without Torch

**Windows (PowerShell):**
```powershell
./setup_env.ps1 -headless -NoTorch
```

**Linux/macOS (Bash):**
```bash
bash setup_env.sh --headless --notorch
```

- Skips torch installation and any codebase/group that requires torch.
- Use with `-Codebases`/`--codebases`/`-Groups`/`--groups` to further restrict selection.

---

## Avoiding Torch and Torch-Dependent Codebases/Groups

- Use the `-NoTorch` (PowerShell) or `--notorch` (Bash) flag to skip torch installation.
- The setup scripts will also skip any codebase or group that requires torch, either directly or as a dependency.
- In headless mode, auto-selection will avoid torch-dependent codebases/groups if `-NoTorch`/`--notorch` is set.
- In interactive mode, torch-dependent options will be hidden or disabled if possible.

---

## Examples

**Minimal install, no venv, no torch:**
```powershell
./setup_env.ps1 -NoVenv -NoTorch -NoExtras
```
```bash
bash setup_env.sh --no-venv --notorch --noextras
```

**Developer setup, only for groupX, with torch:**
```powershell
./setup_env_dev.ps1 -Groups groupX
```
```bash
bash setup_env_dev.sh --groups=groupX
```

**Headless, all extras, skip torch:**
```powershell
./setup_env.ps1 -headless -Extras -NoTorch
```
```bash
bash setup_env.sh --headless --extras --notorch
```

**Interactive, GPU torch, specific codebases:**
```powershell
./setup_env.ps1 -gpu -Codebases projectA,projectB
```
```bash
bash setup_env.sh --gpu --codebases=projectA,projectB
```

---

## Notes

- All options are case-insensitive in PowerShell, case-sensitive in Bash.
- The `-NoTorch`/`--notorch` flag is the authoritative way to avoid torch and torch-dependent codebases/groups.
- For advanced codebase/group selection, use the developer menu or pass explicit flags.
- The scripts will create a `.venv` directory by default unless `-NoVenv`/`--no-venv` is used.
- Selections are recorded to the file specified by the `SPEAKTOME_ACTIVE_FILE` environment variable (or a default path).

---

For further help, see the script headers or run with `-?` (PowerShell) or `--help` (Bash).
