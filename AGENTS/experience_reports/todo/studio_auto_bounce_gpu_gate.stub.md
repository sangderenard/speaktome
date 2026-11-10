# Studio Auto bounce GPU validation

Source and CPU/static checks are complete. When no production photo/live GPU job is active, run Spectral's tests/marble_bounce_probe_gpu_check.py directly with the default Python311 interpreter, then inspect the bounded three-case receipt. Coordinate one GPU owner with the live-history worker. No production restart or publication until this gate and the integration owner's final batch checks pass.
