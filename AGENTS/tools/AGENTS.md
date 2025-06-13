# Agent Tools

## Quick Setup

See `ENV_SETUP_OPTIONS.md` for all environment setup instructions.

This codebase provides helper scripts for managing the repository and coordinating agent work.

## Optional Dependency Groups

The root `pyproject.toml` defines optional groups `dev`, `cpu-torch`, and `gpu-torch`.
These are all skipped by default and may be installed through `dev_group_menu.py`.

## Non-Interactive Setup

No additional steps are required beyond running the main setup script.

### Avoid Premature Imports

Do **not** import helper modules (like `header_utils`) within environment setup
scripts. These scripts run before dependencies are installed, so importing a
package that is provided by those dependencies will fail. Instead, rely solely
on standard library modules and the logic in `auto_env_setup` to bootstrap the
environment.
