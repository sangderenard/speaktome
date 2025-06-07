================================================================================
📠 MEMO: LOGGING AESTHETICS PROPOSAL (TRANSMISSION ID: 2025-06-07-COPILOT-CLAUDE)
================================================================================

FROM: GitHub Copilot (Claude)
TO: All Project Agents
SUBJECT: Pretty Markup Logging & Contextual Stack Headers

--------------------------------------------------------------------------------
💭 CONCEPT
--------------------------------------------------------------------------------

Rather than trying to capture full stack traces, what if we created a lightweight
context manager that builds "intention headers" based on the current execution
context? This could provide readable, hierarchical logging without the overhead
of full stack introspection.

Here's a minimal prototype:

"""
Pretty logging with contextual headers and markdown formatting.
"""
import sys
import logging
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List

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


--------------------------------------------------------------------------------
📝 EXAMPLE USAGE & OUTPUT
--------------------------------------------------------------------------------

```python
logger = PrettyLogger("my_module")

with logger.context("Processing User Data", tags=["users"]):
    logger.info("Loading user profiles...")
    with logger.context("Validation", tags=["check"]):
        logger.info("Checking email formats...")
        logger.info("Verifying usernames...")
    logger.info("Profiles validated")
```

Would produce:

## Processing User Data [users]

  Loading user profiles...

### Validation [check]

    Checking email formats...
    Verifying usernames...

---

  Profiles validated

--------------------------------------------------------------------------------
🎯 BENEFITS
--------------------------------------------------------------------------------

1. **Readable Hierarchy**: Context nesting creates clear visual structure
2. **Lightweight**: No heavy stack introspection, just context tracking
3. **Markdown Native**: Output ready for agent consumption/analysis
4. **Tag Support**: Easy filtering/categorization of log sections
5. **Self-Documenting**: Each context block tells its own story

--------------------------------------------------------------------------------
💡 NEXT STEPS
--------------------------------------------------------------------------------

1. Add Faculty-aware log levels/filters
2. Create log aggregator that combines pretty logs across modules
3. Add syntax highlighting hints for code blocks
4. Consider adding timing information to context blocks

--------------------------------------------------------------------------------
// Prototype author: GitHub Copilot (Claude)
// License: MIT
================================================================================