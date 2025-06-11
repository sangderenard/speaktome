# Cooperative Environment and Codebase Definition

This repository functions as a shared development space for biological and digital agents. Each **codebase** refers to a self-contained project directory such as `speaktome`. All codebases are listed in `CODEBASE_REGISTRY.md`.

Agents collaborate by submitting parallel pull requests against these independent projects. Use the abstract tensor interface (`AbstractTensor`) for any parallel numeric work to keep implementations backend agnostic.

Conflicts should be rare when tasks remain isolated. When two pull requests implement the same goal, review them side by side and adopt the version that best adheres to documented standards. Voting or discussion can resolve differences while preserving the cooperative ethos.

## Virtual Environment Expectations

All repository scripts run from the `.venv` interpreter created by the setup
scripts. For detailed instructions see `ENV_SETUP_OPTIONS.md`.
