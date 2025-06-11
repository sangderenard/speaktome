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

Run `setup_env.sh` (or the PowerShell equivalent) from the repository root to create `.venv` and install selected codebases. Previous revisions referenced a separate menu script for group selection; that method is deprecated.

### Non‑Interactive Example

```bash
bash setup_env_dev.sh --from-dev
```

Add additional codebases by listing them with `--codebases speaktome,laplace`.

Each project directory contains a `README.md` and `AGENTS.md` with more detailed guidance.
