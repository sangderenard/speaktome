# SpeakToMe Multi‑Project Repository

This repository hosts several independent codebases that share a common virtual environment. Each codebase has its own documentation and optional dependency groups defined in its `pyproject.toml`.

## Available Codebases

- **speaktome** — main beam search controllers and utilities.
- **laplace** — Laplace builder and DEC utilities.
- **tensor_printing** — experimental Grand Printing Press package.
- **time_sync** — system clock offset helpers.
- **AGENTS/tools** — shared helper scripts for repository management.

See `AGENTS/CODEBASE_REGISTRY.md` for the canonical list.

## Environment Setup

All environment configuration is handled by the headless workflow described in
`AGENTS/HEADLESS_SETUP_GUIDE.md`.
