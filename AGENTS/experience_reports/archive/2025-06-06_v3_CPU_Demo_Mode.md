# CPU Demo Mode Implementation

**Date/Version:** 2025-06-06 v3
**Title:** Added CPU-only fallback for minimal lookahead

## Overview
Following feedback about allowing exploration without installing PyTorch,
this update introduces a small NumPy-based demo. When PyTorch is
unavailable the main entry point now delegates to `cpu_demo.main` which
runs a simplified lookahead algorithm using a `TokenVocabulary` helper
and a generic `topk` function.

## Key Changes
- Optional imports guard against missing PyTorch.
- `config.py` exposes `TORCH_AVAILABLE` and provides a dummy device
  object when PyTorch is absent.
- New modules `array_utils.py`, `token_vocab.py`, and `cpu_demo.py`
  implement the NumPy path.
- `speaktome.main` automatically launches the demo mode if PyTorch is not
  installed.

## Next Steps
This demo bypasses retirement and outer-loop logic. It only demonstrates
lookahead with randomly sampled tokens. Future iterations may expand the
vocabulary handling and allow scoring strategies to plug into either the
NumPy or PyTorch pipeline.
