# Documentation Report

**Date:** 1749920678
**Title:** Disable hub package installation

## Summary
- added `package-mode = false` in root `pyproject.toml`
- prevents Poetry from installing nonexistent `speaktome-hub` package

## Prompt History
```
Error: The current project could not be installed: No file/folder found for package speaktome-hub
```
