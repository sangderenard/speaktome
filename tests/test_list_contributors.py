"""Tests for list_contributors script handling LFS pointers."""
from pathlib import Path
from AGENTS.tools.list_contributors import generate_credits
# --- END HEADER ---

def test_skips_lfs(tmp_path: Path) -> None:
    """LFS pointer files are ignored."""
    users_dir = tmp_path / "users"
    users_dir.mkdir()
    (users_dir / "valid.json").write_text('{"name": "A", "created_by": "B"}', encoding="utf-8")
    (users_dir / "lfs.json").write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    result = generate_credits(users_dir)
    assert "A" in result
    assert "lfs.json" not in result
