"""Select a compiled OpenRGB light-field effect without changing its engine."""

from __future__ import annotations

import argparse
from typing import Sequence


EFFECTS = ("spirograph", "three-body", "studio")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("effect", choices=EFFECTS)
    args, effect_argv = parser.parse_known_args(argv)
    if args.effect == "studio":
        from openrgb_field_studio import main as effect_main
    elif args.effect == "spirograph":
        from openrgb_dt_spirograph import main as effect_main
    else:
        from openrgb_dt_three_body import main as effect_main
    return effect_main(effect_argv)


if __name__ == "__main__":
    raise SystemExit(main())
