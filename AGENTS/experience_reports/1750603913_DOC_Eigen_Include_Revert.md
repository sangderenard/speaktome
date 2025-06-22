# Eigen Include Revert

**Date:** 1750603913
**Title:** Restore submodule include paths

## Overview
Headers were changed to use system-style Eigen includes. The project bundles
Eigen as a submodule, so include paths should reference the local `eigen/`
directory. This commit reverts those changes.

## Prompt History
```
Eigen is a submodule located at speaktome/asciioscilliscope/eigen, so I believe the imports should be eigen/Eigen or eigen/unsupported, This is not a project using system libraries. I appreciate your work but it's not exactly apropriate to change the includes to pretend they are system libraries not bundled vendored submodules
```
