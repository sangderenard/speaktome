"""Tests for the update_agents_toc utility."""
from pathlib import Path
import json

from AGENTS.tools.update_agents_toc import generate_toc, replace_toc

# --- END HEADER ---


def test_generate_and_replace(tmp_path: Path) -> None:
    agents_dir = tmp_path / "AGENTS"
    agents_dir.mkdir()
    (agents_dir / "AGENTS.md").write_text("# Header\n")
    (agents_dir / "a1.md").write_text("x")
    (agents_dir / "a2.md").write_text("x")

    desc = {"a1.md": "desc1", "AGENTS.md": "self"}
    toc = generate_toc(agents_dir, desc)
    replace_toc(agents_dir / "AGENTS.md", toc)
    data = (agents_dir / "AGENTS.md").read_text()
    assert "desc1" in data
    assert "unknown" in data
