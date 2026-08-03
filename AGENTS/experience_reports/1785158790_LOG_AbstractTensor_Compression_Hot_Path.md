# AbstractTensor Compression Hot Path

**Date:** 1785158790
**Title:** Remove Python-carried JPEG coefficient scans and trusted-path scalar synchronization

## Command

`py -3.11 -m pytest -q tests/test_tensor_compression_coefficient_events.py tests/test_tensor_compression_entropy_symbols.py tests/test_tensor_compression_huffman.py tests/test_tensor_compression_jpeg_frame.py tests/test_tensor_compression_avi.py tests/test_glsl_backend.py`

## Log

```text
144 passed, 1 warning in 48.96s
```

The JPEG coefficient hot path now derives AC zero runs by AbstractTensor rank,
scatter, and adjacent compact-position differencing. Fixed standard JPEG tables
and known-valid encoder payloads bypass host `.item()` validation, while public
general-purpose APIs retain validation by default.

## Prompt History

> "start converting compression code to abstract tensor not python or anything else, abstract tensor only"

> "proceed"
