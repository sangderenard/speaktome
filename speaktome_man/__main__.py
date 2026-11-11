"""``python -m speaktome_man [SECTION] NAME`` prints a page; ``-k WORD`` searches; no argument lists every page."""
from __future__ import annotations

import sys

from . import SECTIONS, ManError, index, listing, page


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(listing(index()))
        return 0
    if args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "-k":
        if len(args) != 2:
            print("usage: python -m speaktome_man -k WORD", file=sys.stderr)
            return 2
        rows = index(args[1])
        print(listing(rows) if rows else f"{args[1]}: nothing appropriate")
        return 0 if rows else 1
    section = args.pop(0) if len(args) == 2 and args[0] in SECTIONS else None
    if len(args) != 1:
        print("usage: python -m speaktome_man [SECTION] NAME", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(page(args[0], section).render())
    except ManError as error:
        print(error.message + (f"; nearest: {', '.join(error.nearest)}" if error.nearest else ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
