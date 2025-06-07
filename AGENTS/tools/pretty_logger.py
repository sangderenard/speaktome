"""Pretty logging with contextual headers and markdown formatting."""

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

# --- END HEADER ---

@dataclass
class LogContext:
    """Tracks the current logging context stack."""
    title: str
    depth: int
    tags: List[str]
    
class PrettyLogger:
    HEADER = """A logger that auto-generates aesthetic markdown headers based on context."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.contexts: List[LogContext] = []
        self._setup_handlers()
    
    @staticmethod
    def test():
        """Validate logger functionality."""
        logger = PrettyLogger("test")
        with logger.context("Test Operation", tags=["test"]):
            logger.info("Testing context depth 1")
            with logger.context("Nested Operation", tags=["nested"]):
                logger.info("Testing context depth 2")
    
    def _setup_handlers(self):
        """Configure markdown-friendly output handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @contextmanager
    def context(self, title: str, tags: Optional[List[str]] = None):
        """Create a logging context with pretty headers."""
        ctx = LogContext(title, len(self.contexts), tags or [])
        self.contexts.append(ctx)
        
        # Print entry header
        depth_marker = "#" * (ctx.depth + 2)  # Start at h2
        tag_str = " ".join(f"[{t}]" for t in ctx.tags)
        self.logger.info(f"\n{depth_marker} {ctx.title} {tag_str}\n")
        
        try:
            yield
        finally:
            self.contexts.pop()
            if self.contexts:  # Add separator when returning to parent context
                self.logger.info("\n---\n")
    
    def info(self, msg: str):
        """Log with current context indentation."""
        indent = "  " * len(self.contexts)
        self.logger.info(f"{indent}{msg}")
