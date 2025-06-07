import pytest
import logging
import time
from pathlib import Path
import sys


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # Our logger logs immediately, so flush is a no-op.
        pass

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "stub: placeholder test requiring implementation")

    log_dir = Path("testing/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # remove old log files, keeping the 10 most recent
    logs = sorted(log_dir.glob("pytest_*.log"), key=lambda p: p.stat().st_mtime)
    for old in logs[:-10]:
        try:
            old.unlink()
        except OSError:  # pragma: no cover
            pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pytest_{timestamp}.log"

    # Get the root logger and set its level to capture INFO and higher messages
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) # Ensure handler processes INFO messages

    root_logger.addHandler(file_handler)
    config._speaktome_log_handler = file_handler

    # Store original stdout and redirect sys.stdout to our logger
    config._original_stdout = sys.stdout
    sys.stdout = StreamToLogger(root_logger, logging.INFO)


def pytest_unconfigure(config: pytest.Config) -> None:
    # Restore stdout
    original_stdout = getattr(config, "_original_stdout", None)
    if original_stdout:
        sys.stdout = original_stdout

    handler = getattr(config, "_speaktome_log_handler", None)
    if handler:
        logging.getLogger().removeHandler(handler)
        handler.close()
