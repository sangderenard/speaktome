# Cooperative Environment and Codebase Definition

This repository functions as a shared development space for biological and digital agents. Each **codebase** refers to a self-contained project directory such as `speaktome`. All codebases are listed in `CODEBASE_REGISTRY.md`.

Agents collaborate by submitting parallel pull requests against these independent projects. Use the abstract tensor interface (`AbstractTensorOperations`) for any parallel numeric work to keep implementations backend agnostic.

Conflicts should be rare when tasks remain isolated. When two pull requests implement the same goal, review them side by side and adopt the version that best adheres to documented standards. Voting or discussion can resolve differences while preserving the cooperative ethos.

## Virtual Environment Expectations

All repository scripts must run from the Python interpreter in `.venv`. The
developer setup (`setup_env_dev.sh` or `setup_env_dev.ps1`) installs every
required dependency. If you encounter an `ImportError`, check the following:

1. Ensure the `.venv` environment is active. Running outside of the virtual
   environment is the most common cause of missing packages.
2. Confirm `setup_env_dev` finished successfully. A partial run can leave the
   environment incomplete.

It should not be possible to correctly use the provided virtual environment and
lack a required library. When a package truly seems absent, verify that
`setup_env.sh` does not already install it, then add the dependency to
`pyproject.toml` so future setups fetch it automatically. (For example, `torch`
is installed by default and should never be missing.)
