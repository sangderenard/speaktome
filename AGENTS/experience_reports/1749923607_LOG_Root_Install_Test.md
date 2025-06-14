# Log Report

**Date:** 1749923607
**Title:** Root install dry-run

## Command
`POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --dry-run -vv`

## Log
```text
Using virtualenv: /workspace/speaktome/.venv
Checking keyring availability: Unavailable
Updating dependencies
Resolving dependencies...
   1: fact: speaktome-root is 0.1.0
   1: derived: speaktome-root
   0: Duplicate dependencies for torch
   0: Merging requirements for torch
   1: fact: speaktome-root depends on tools (0.1.0)
   1: fact: speaktome-root depends on testenv (0.1.0)
   1: fact: speaktome-root depends on pytest (>=8.0)
   1: fact: speaktome-root depends on torch (2.3.1)
   1: fact: speaktome-root depends on speaktome (0.1.0)
   1: fact: speaktome-root depends on fontmapper (0.1.0)
   1: fact: speaktome-root depends on laplace (0.1.0)
   1: fact: speaktome-root depends on tensors (0.1.0)
   1: fact: speaktome-root depends on timesync (0.1.0)
   1: fact: speaktome-root depends on tensorprinting (0.1.0)
   1: selecting speaktome-root (0.1.0)
   1: derived: tensorprinting (0.1.0) @ file:///workspace/speaktome/tensorprinting
```

## Prompt History
```
you ignored part of my request please comment on the issue installing the root, please dump it to log trying to, before it couldn't find the folder and we need to know it can find the monorepo root
```
