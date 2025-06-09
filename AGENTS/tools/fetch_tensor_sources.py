"""Download third-party sources for tensor models.

This script ensures required third-party libraries exist for
`speaktome/tensors/models`. If a dependency directory contains only
 the placeholder README shipped with the repository, the script clones
its source from GitHub using ``git``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# --- END HEADER ---

THIRD_PARTY = {
    "blis": "https://github.com/flame/blis.git",
    "blasfeo": "https://github.com/giaf/blasfeo.git",
    "sleef": "https://github.com/shibatch/sleef.git",
    "cephes": "https://github.com/jeremybarnes/cephes.git",
    "cnpy": "https://github.com/rogersce/cnpy.git",
    "klib": "https://github.com/attractivechaos/klib.git",
    "uthash": "https://github.com/troydhanson/uthash.git",
    "marisa-trie": "https://github.com/s-yata/marisa-trie.git",
    "jsmn": "https://github.com/zserge/jsmn.git",
    "cJSON": "https://github.com/DaveGamble/cJSON.git",
    "pcg-c": "https://github.com/imneme/pcg-c.git",
    "xoshiro-c": "https://github.com/brycelelbach/xoshiro-c.git",
    "stb": "https://github.com/nothings/stb.git",
    "docopt.c": "https://github.com/docopt/docopt.c.git",
    "linenoise": "https://github.com/antirez/linenoise.git",
}


def run(cmd: list[str]) -> None:
    """Run a command and print it."""
    print("Executing", " ".join(cmd))
    subprocess.run(cmd, check=True)


def needs_clone(path: Path) -> bool:
    """Return True if ``path`` is missing real content."""
    if not path.exists():
        return True
    entries = [p for p in path.iterdir() if p.name != "README.md"]
    return len(entries) == 0


def ensure_repo(name: str, url: str, base: Path) -> None:
    """Clone ``url`` into ``base/name`` if needed."""
    dest = base / name
    if needs_clone(dest):
        if dest.exists():
            for item in dest.iterdir():
                item.unlink()
        else:
            dest.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", url, str(dest)])
    else:
        print(f"{name} already present; skipping")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    third_party = root / "third_party"
    for name, url in THIRD_PARTY.items():
        ensure_repo(name, url, third_party)


if __name__ == "__main__":
    main()
