import pytest
import logging
import time
from pathlib import Path


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "stub: placeholder test requiring implementation")

    log_dir = Path("testing/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # remove old log files, keeping the 10 most recent
    logs = sorted(log_dir.glob("pytest_*.log"), key=lambda p: p.stat().st_mtime)
    for old in logs[:-10]:
        try:
            old.unlink()
        except OSError:
            pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pytest_{timestamp}.log"

    handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    config._speaktome_log_handler = handler


def pytest_unconfigure(config: pytest.Config) -> None:
    handler = getattr(config, "_speaktome_log_handler", None)
    if handler:
        logging.getLogger().removeHandler(handler)
        handler.close()
