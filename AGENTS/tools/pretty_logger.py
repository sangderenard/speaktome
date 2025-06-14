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
    
    def __init__(self, name: str, enable_color: bool | None = None):
        self.logger = logging.getLogger(name)
        self.contexts: List[LogContext] = []
        self.color_enabled = enable_color if enable_color is not None else sys.stdout.isatty()
        if self.color_enabled and sys.platform.startswith("win"):
            try:
                import colorama
                colorama.just_fix_windows_console()
            except Exception:  # pragma: no cover - optional
                pass
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

    def _color(self, text: str, code: str) -> str:
        if not self.color_enabled:
            return text
        return f"{code}{text}\x1b[0m"
    
    @contextmanager
    def context(self, title: str, tags: Optional[List[str]] = None):
        """Create a logging context with pretty headers."""
        ctx = LogContext(title, len(self.contexts), tags or [])
        self.contexts.append(ctx)
        
        # Print entry header
        depth_marker = "#" * (ctx.depth + 2)  # Start at h2
        tag_str = " ".join(f"[{t}]" for t in ctx.tags)
        header_line = f"\n{depth_marker} {ctx.title} {tag_str}\n"
        self.logger.info(self._color(header_line, "\x1b[95m"))
        
        try:
            yield
        finally:
            self.contexts.pop()
            if self.contexts:  # Add separator when returning to parent context
                self.logger.info("\n---\n")
    
    def info(self, msg: str):
        """Log with current context indentation."""
        indent = "  " * len(self.contexts)
        self.logger.info(self._color(f"{indent}{msg}", "\x1b[92m"))

    # Optional color enhancements for console output.
    def enable_color(self) -> None:
        """Enable colored output for subsequent log messages."""
        self.color_enabled = True
        if sys.platform.startswith("win"):
            try:
                import colorama
                colorama.just_fix_windows_console()
            except Exception:  # pragma: no cover - optional
                pass

    def disable_color(self) -> None:
        """Disable colored output."""
        self.color_enabled = False


@dataclass
class TreeNode:
    """Simple tree node for :class:`AsciiTreeRenderer`."""

    label: str
    children: List["TreeNode"] | None = None


class AsciiTreeRenderer:
    """Render tree structures using ASCII boxes and connectors."""

    COLORS = [
        "\x1b[48;5;24m",
        "\x1b[48;5;25m",
        "\x1b[48;5;26m",
        "\x1b[48;5;27m",
        "\x1b[48;5;28m",
        "\x1b[48;5;29m",
    ]
    RESET = "\x1b[0m"

    def __init__(self, use_color: bool = True) -> None:
        self.use_color = use_color
        if self.use_color and sys.platform.startswith("win"):
            try:
                from time_sync import init_colorama_for_windows
            except Exception:  # pragma: no cover - optional
                init_colorama_for_windows = None
            if init_colorama_for_windows:
                init_colorama_for_windows()

    def render(self, root: TreeNode) -> str:
        lines: List[str] = []
        self._render_node(root, "", 0, True, lines)
        return "\n".join(lines)

    def _colorize(self, depth: int, text: str) -> str:
        if not self.use_color:
            return text
        color = self.COLORS[depth % len(self.COLORS)]
        return f"{color}{text}{self.RESET}"

    def _render_node(
        self,
        node: TreeNode,
        prefix: str,
        depth: int,
        is_last: bool,
        lines: List[str],
    ) -> None:
        label_box = f"[ {node.label} ]"
        connector = prefix[:-3] if prefix.endswith("│  ") else prefix
        if prefix:
            branch = "└── " if is_last else "├── "
            lines.append(connector + branch + self._colorize(depth, label_box))
        else:
            lines.append(self._colorize(depth, label_box))

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = node.children or []
        for idx, child in enumerate(children):
            self._render_node(child, child_prefix, depth + 1, idx == len(children) - 1, lines)

