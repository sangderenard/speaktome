# Documentation Report

**Date:** 1749896072
**Title:** Poetry-core installation checks in setup scripts

## Overview
Both `setup_env.sh` and its PowerShell counterpart `setup_env.ps1` verify whether the `poetry.core.masonry.api` module is available. If it cannot be imported, these scripts attempt to install `poetry-core>=1.5.0` using pip before continuing. The developer-oriented `setup_env_dev.ps1` includes the same logic. This automatic check ensures editable installs work even when the backend is missing.

## Prompt History
```
who in the scripts is responsible for poetry core and can you dump the failure to a markdown log experience report and create a new experience report type and guidance for logs, which should try pretty logger but if that can't be loaded, the log experience reports should just be raw log files made markup so github tracks them, their template should just explain there are to be piped output from repeatable errors or requested analysis
```
