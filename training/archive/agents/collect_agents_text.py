from __future__ import annotations
import shutil
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parents[2] / 'AGENTS'
OUTPUT_DIR = Path(__file__).resolve().parent

EXTS = {'.md', '.txt', '.py', '.sh', '.ps1'}


def collect() -> None:
    for path in AGENTS_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in EXTS:
            dest = OUTPUT_DIR / path.relative_to(AGENTS_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


if __name__ == '__main__':
    collect()
