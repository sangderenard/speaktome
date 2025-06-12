try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from fontmapper import ascii_preview
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from pathlib import Path


def test_ascii_preview_basic():
    font_path = Path("fontmapper/FM16/consola.ttf")
    result = ascii_preview([str(font_path)], font_size=12, complexity_level=0)
    assert isinstance(result, str)
    assert len(result) > 0
