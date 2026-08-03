# SpeakToMe multi-project repository

SpeakToMe is a Python research workspace whose codebases share environment and agent tooling but remain conceptually distinct. The central application explores beam-search controllers and “wide truth” region estimation; sibling packages cover tensor abstraction, DEC/Laplace utilities, printing, and time synchronization.

This directory is itself one project within the larger `C:\dev\Powershell` workspace. Instructions here do not make the parent directory a monorepo.

## Codebases

- **speaktome** — main beam-search controllers and utilities.
- **laplace** — Laplace builder and DEC utilities.
- **tensorprinting** — experimental Grand Printing Press package.
- **timesync** — system-clock offset helpers.
- **AGENTS/tools** — shared repository-management helpers.

See [`AGENTS/CODEBASE_REGISTRY.md`](AGENTS/CODEBASE_REGISTRY.md) for the canonical registry and [`AGENTS_FILESYSTEM_MAP.md`](AGENTS_FILESYSTEM_MAP.md) for orientation.

## Environment and tests

Do not install packages manually into an arbitrary interpreter. Follow [`ENV_SETUP_OPTIONS.md`](ENV_SETUP_OPTIONS.md) and [`AGENTS_DO_NOT_PIP_MANUALLY.md`](AGENTS_DO_NOT_PIP_MANUALLY.md).

Testing conventions, including the distinction between `tests/` and `testing/`, are in [`AGENTS_TESTING_ADVICE.md`](AGENTS_TESTING_ADVICE.md).

## Design context

For the C++ simulation ethos, read [`CRT_Vector_Manifesto.md`](CRT_Vector_Manifesto.md). For the beam-search direction—parallel forward/backward beams, independent batched/threaded workers, and “wide truth” region estimation—read [`VISION_FORWARD_BACKWARD_DIFFUSION.md`](VISION_FORWARD_BACKWARD_DIFFUSION.md).

## Legacy tooling

`AGENTS.tools.headers.header_utils` remains for historical reference only; do not import it. Scripts should read `ENV_SETUP_BOX` from the environment. Run `python -m AGENTS.tools.headers.run_header_checks` to repair, validate, and test repository headers.
