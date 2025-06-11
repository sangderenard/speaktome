# Batch_Kernel_Only

**Date/Version:** 1749681354 v1

## Prompt History
- User: "look for parallelization optimization in the clock_demo, particularly where we are judging similarities to a token set, I want to repeat interleave and compare every input to every output at once in one operation that is as efficient as its backend is when we insist it work in parallel as torch would"
- User: "lets go ahead and require that any kernel function in parallel we don't actually care about serial versions of this idea"

## Summary
Converted `draw_diff` to always operate on batches by requiring the provided kernel to accept an array of subunits and return a list of characters. Removed the single-subunit path and updated `flexible_subunit_kernel` along with tests. The default kernel remains `default_subunit_batch_to_chars` which uses `AsciiKernelClassifier.classify_batch` for parallel character selection.

## Testing
- `python -m py_compile time_sync/ascii_kernel_classifier.py time_sync/draw.py tests/test_draw.py`
- `pytest -q` *(fails: environment not configured)*
- `PYTHONPATH=. python AGENTS/validate_guestbook.py`
