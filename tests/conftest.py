"""PyTest configuration with faculty-aware logging.

This configuration file establishes a consistent logging setup for all tests
and announces the active :class:`~speaktome.faculty.Faculty` tier.  The output
is written to ``testing/logs`` so future agents may trace prior sessions.
"""

import pytest
import logging
import time
from pathlib import Path
import sys
import os  # For FORCE_ENV

# Import faculty components for logging
from speaktome.faculty import DEFAULT_FACULTY, FORCE_ENV, Faculty

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

    def isatty(self):  # pragma: no cover - mimic stdout interface
        """Return False as this stream is not an interactive tty."""
        return False

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

    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate logging if pytest is run multiple times in one session/process
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
        
    root_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) # Ensure handler processes INFO messages

    root_logger.addHandler(file_handler)
    config._speaktome_log_handler = file_handler

    # --- Enhanced Log Header & Faculty Information ---
    log_header_intro = f"""
================================================================================
SPEAKTOME PROJECT - PYTEST LOG
================================================================================
Test Session Timestamp: {timestamp}

Project Testing Strategy:
-------------------------
This project utilizes a 'faculty-based' approach to manage dependencies and
compute resources. Tests may behave differently or be skipped based on the
detected faculty. The active faculty for this test run is reported below.

Faculties (from lowest to highest):
  - PURE_PYTHON: No optional numerical/ML libraries. Core logic.
  - NUMPY: NumPy available. Enables basic numerical operations.
  - TORCH: PyTorch available. Enables GPU acceleration and core ML models.
  - PYGEO: PyTorch Geometric available. Enables GNN features.

How to Help / Interpret This Log:
---------------------------------
1. Identify the first FAILED test or critical ERROR message.
2. Note the active faculty. Failures might be due to missing optional
   dependencies if a higher faculty is expected by the test.
3. If a test is SKIPPED, the reason will indicate if it's due to faculty
   requirements or if it's a 'stub' test needing implementation.
   Check 'testing/stub_todo.txt' for a list of stubs.
4. For AGENTS: Look for messages prefixed with [AGENT_ACTIONABLE_ERROR],
   [AGENT_TASK], or [AGENT_INFO]. These indicate specific issues or tasks.
5. The goal is an iterative cycle: run pytest, fix first error, repeat.
   Developer entry point: use 'bash setup_env.sh', then follow CLI guidance after activating the environment.

Active Faculty for this Session:
--------------------------------
"""
    root_logger.info(log_header_intro)

    forced_faculty_env = os.environ.get(FORCE_ENV)
    if forced_faculty_env:
        try:
            # Attempt to use the forced faculty for logging, but DEFAULT_FACULTY reflects actual runtime for tests
            forced_faculty_val_log = Faculty[forced_faculty_env.upper()]
            root_logger.info(f"[FACULTY_INFO] Environment variable {FORCE_ENV} is SET to '{forced_faculty_env}'.")
            root_logger.info(f"[FACULTY_INFO] Tests will attempt to run as if faculty is {forced_faculty_val_log.name}.")
            root_logger.info(f"[FACULTY_INFO] Actual auto-detected faculty (used by tests unless overridden in code): {DEFAULT_FACULTY.name}.")
        except KeyError:
            root_logger.warning(f"[FACULTY_WARNING] Environment variable {FORCE_ENV} is SET to an INVALID value '{forced_faculty_env}'.")
            root_logger.info(f"[FACULTY_INFO] Using auto-detected faculty for tests: {DEFAULT_FACULTY.name}.")
    else:
        root_logger.info(f"[FACULTY_INFO] Environment variable {FORCE_ENV} is NOT set.")
        root_logger.info(f"[FACULTY_INFO] Using auto-detected faculty for tests: {DEFAULT_FACULTY.name}.")
    
    root_logger.info("--------------------------------------------------------------------------------")
    # --- End Enhanced Log Header & Faculty Information ---

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
