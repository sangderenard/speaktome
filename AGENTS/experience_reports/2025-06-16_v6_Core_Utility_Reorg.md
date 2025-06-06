# Core Utility Reorg

**Date/Version:** 2025-06-16 v6
**Title:** Core Utility Reorg

## Overview
Continued the domain reorganization by moving remaining utility and algorithm modules into
subpackages. The `core` package now houses beam search components and tensor abstractions,
while general helpers live under `util`. Import paths and tests were updated accordingly.

## Prompt History
```
continue moving files and adjusting imports until the root directory is clean of utilities
```

## Steps Taken
1. Created `speaktome/core` and `speaktome/util` packages.
2. Relocated beam search and helper modules into these packages.
3. Updated all relative imports and tests.
4. Verified the full test suite passes.

