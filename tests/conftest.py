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
import importlib.util
from io import StringIO

# Import faculty components for logging
from speaktome.faculty import DEFAULT_FACULTY, FORCE_ENV, Faculty

spec = importlib.util.spec_from_file_location(
    "pretty_logger",
    Path(__file__).resolve().parents[1] / "AGENTS" / "tools" / "pretty_logger.py",
)
pretty_mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(pretty_mod)
PrettyLogger = pretty_mod.PrettyLogger
# --- END HEADER ---

class StdoutTee:
    """Duplicate writes to the original stdout and a logger."""

    def __init__(self, stream, logger, log_level=logging.INFO) -> None:
        self.stream = stream
        self.logger = logger
        self.log_level = log_level

    def write(self, buf: str) -> None:  # pragma: no cover - passthrough
        self.stream.write(buf)
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self) -> None:  # pragma: no cover - passthrough
        self.stream.flush()

    def isatty(self) -> bool:  # pragma: no cover - mimic stdout interface
        return getattr(self.stream, "isatty", lambda: False)()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-stubs",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.stub",
    )

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
    md_file = log_dir / f"pytest_{timestamp}.md"

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

    pretty_logger = PrettyLogger("pytest-pretty")
    md_handler = logging.FileHandler(md_file, mode="w", encoding="utf-8")
    md_handler.setFormatter(logging.Formatter("%(message)s"))
    pretty_logger.logger.addHandler(md_handler)
    config._speaktome_pretty_logger = pretty_logger

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
    with pretty_logger.context("PyTest Session Header"):
        pretty_logger.info("Test Session Timestamp: " + timestamp)

    forced_faculty_env = os.environ.get(FORCE_ENV)
    with pretty_logger.context("Faculty Information"):
        if forced_faculty_env:
            try:
                forced_faculty_val_log = Faculty[forced_faculty_env.upper()]
                msg = (
                    f"Environment variable {FORCE_ENV} is SET to '{forced_faculty_env}'."
                )
                pretty_logger.info(msg)
                root_logger.info(f"[FACULTY_INFO] {msg}")
                msg = (
                    f"Tests will attempt to run as if faculty is {forced_faculty_val_log.name}."
                )
                pretty_logger.info(msg)
                root_logger.info(f"[FACULTY_INFO] {msg}")
                msg = (
                    "Actual auto-detected faculty (used by tests unless overridden in code): "
                    f"{DEFAULT_FACULTY.name}."
                )
                pretty_logger.info(msg)
                root_logger.info(f"[FACULTY_INFO] {msg}")
            except KeyError:
                warn_msg = (
                    f"Environment variable {FORCE_ENV} is SET to an INVALID value '{forced_faculty_env}'."
                )
                pretty_logger.info(warn_msg)
                root_logger.warning(f"[FACULTY_WARNING] {warn_msg}")
                msg = f"Using auto-detected faculty for tests: {DEFAULT_FACULTY.name}."
                pretty_logger.info(msg)
                root_logger.info(f"[FACULTY_INFO] {msg}")
        else:
            msg = f"Environment variable {FORCE_ENV} is NOT set."
            pretty_logger.info(msg)
            root_logger.info(f"[FACULTY_INFO] {msg}")
            msg = f"Using auto-detected faculty for tests: {DEFAULT_FACULTY.name}."
            pretty_logger.info(msg)
            root_logger.info(f"[FACULTY_INFO] {msg}")
    
    root_logger.info("--------------------------------------------------------------------------------")
    # --- End Enhanced Log Header & Faculty Information ---
    from dump_headers import dump_headers
    buf = StringIO()
    _stdout = sys.stdout
    sys.stdout = buf
    dump_headers(Path("speaktome"))
    dump_headers(Path("AGENTS/tools"))
    sys.stdout = _stdout
    with pretty_logger.context("Module Headers"):
        for line in buf.getvalue().splitlines():
            pretty_logger.info(line)

    # Store original stdout and tee writes to both stdout and the log
    config._original_stdout = sys.stdout
    sys.stdout = StdoutTee(config._original_stdout, root_logger, logging.INFO)


def pytest_unconfigure(config: pytest.Config) -> None:
    # Restore stdout
    original_stdout = getattr(config, "_original_stdout", None)
    if original_stdout:
        sys.stdout = original_stdout

    handler = getattr(config, "_speaktome_log_handler", None)
    if handler:
        logging.getLogger().removeHandler(handler)
        handler.close()

    p_logger = getattr(config, "_speaktome_pretty_logger", None)
    if p_logger:
        for h in p_logger.logger.handlers[:]:
            p_logger.logger.removeHandler(h)
            h.close()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Optionally skip tests marked ``stub``."""
    if not config.getoption("--skip-stubs"):
        return

    skip_stub = pytest.mark.skip(reason="stub skipped via --skip-stubs")
    for item in items:
        if 'stub' in item.keywords:
            item.add_marker(skip_stub)


@pytest.fixture(scope="session")
def pretty_logger(request: pytest.FixtureRequest):
    """Access the PrettyLogger configured for the session."""
    return request.config._speaktome_pretty_logger


@pytest.fixture(autouse=True)
def pretty_context(request: pytest.FixtureRequest, pretty_logger):
    """Wrap each test in a pretty logging context."""
    with pretty_logger.context(request.node.nodeid):
        yield

