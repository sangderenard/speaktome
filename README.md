# SpeakToMe Multi‑Project Repository

This repository hosts several independent codebases that share a common virtual environment. Each codebase has its own documentation and optional dependency groups defined in its `pyproject.toml`.

## Available Codebases

- **speaktome** — main beam search controllers and utilities.
- **laplace** — Laplace builder and DEC utilities.
- **tensorprinting** — experimental Grand Printing Press package.
- **timesync** — system clock offset helpers.
- **AGENTS/tools** — shared helper scripts for repository management.

See `AGENTS/CODEBASE_REGISTRY.md` for the canonical list.

## Environment Setup

All environment configuration is handled by the workflow described in
`ENV_SETUP_OPTIONS.md`.

## Legacy Modules

The helper ``AGENTS.tools.headers.header_utils`` remains in the tree for historical
reference only. **Do not import it**. Scripts should read ``ENV_SETUP_BOX``
directly from the environment instead of relying on this module.

Run ``python -m AGENTS.tools.headers.run_header_checks`` to automatically repair,
validate and test file headers across the repository.
