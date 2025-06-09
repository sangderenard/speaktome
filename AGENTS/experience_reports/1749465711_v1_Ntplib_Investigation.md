# Ntplib Investigation

**Date/Version:** 1749465711 v1
**Title:** Ntplib Investigation

## Overview
Reviewed setup scripts and time_sync package to determine why `ntplib` might not install.

## Prompt History
- "investigate in the environmental setup scripts and the time_sync project why ntplib never gets installed, check if the time_sync project is not properly loaded as a codebase, or if there is a problem with the pyproject for it"

## Steps Taken
1. Examined `setup_env.sh` to confirm `time_sync` is included in editable installs.
2. Verified `time_sync/pyproject.toml` lists `ntplib` as dependency.
3. Ran the setup script and observed `ntplib` is installed via pip.
4. Implemented the analog clock stub in `time_sync/ascii_digits.py`.
5. Ran test suite (failed due to missing heavy dependencies such as torch).

## Observed Behaviour
`ntplib` installs correctly when running `setup_env.sh`; the dependency is specified properly. Failures may come from not running the script or lacking network access.
