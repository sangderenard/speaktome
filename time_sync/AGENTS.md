# Time Sync Utilities

## Quick Setup

Run `setup_env_dev.sh` to install this codebase. Use pip extras like `time_sync[gui]` to enable the demo window.

This project provides simple functions for adjusting system time using an environment variable offset. All modules must follow the coding standards from `../AGENTS/CODING_STANDARDS.md`.

## Optional Dependency Groups

An optional "gui" group installs `pygame` for the demo window:

```bash
python -m pip install time_sync[gui]
```

## Non-Interactive Setup

No additional steps are required beyond running the main setup script.
