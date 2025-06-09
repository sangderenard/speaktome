"""ASCII art utilities for rendering clock faces."""

from __future__ import annotations

import datetime as _dt
from typing import List

# --- END HEADER ---

_DIGITS = {
    '0': [
        " ### ",
        "#   #",
        "#   #",
        "#   #",
        " ### ",
    ],
    '1': [
        "  #  ",
        " ##  ",
        "  #  ",
        "  #  ",
        " ### ",
    ],
    '2': [
        " ### ",
        "    #",
        " ### ",
        "#    ",
        "#####",
    ],
    '3': [
        "#####",
        "    #",
        " ### ",
        "    #",
        "#####",
    ],
    '4': [
        "#   #",
        "#   #",
        "#####",
        "    #",
        "    #",
    ],
    '5': [
        "#####",
        "#    ",
        "#### ",
        "    #",
        "#### ",
    ],
    '6': [
        " ### ",
        "#    ",
        "#### ",
        "#   #",
        " ### ",
    ],
    '7': [
        "#####",
        "    #",
        "   # ",
        "  #  ",
        "  #  ",
    ],
    '8': [
        " ### ",
        "#   #",
        " ### ",
        "#   #",
        " ### ",
    ],
    '9': [
        " ### ",
        "#   #",
        " ####",
        "    #",
        " ### ",
    ],
    ':': [
        "     ",
        "  #  ",
        "     ",
        "  #  ",
        "     ",
    ],
    ' ': [
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
    ],
}


def compose_ascii_digits(text: str) -> str:
    """Return ``text`` rendered as large ASCII digits."""
    rows = ["" for _ in range(5)]
    for ch in text:
        patt = _DIGITS.get(ch, _DIGITS[' '])
        for i in range(5):
            rows[i] += patt[i] + "  "
    return "\n".join(rows)


# ########## STUB: ascii_analog_clock ##########
# PURPOSE: Placeholder for analog ASCII clock rendering.
# EXPECTED BEHAVIOR: Will draw an analog clock face representing the given time.
# INPUTS: datetime object.
# OUTPUTS: multiline string of ASCII art.
# KEY ASSUMPTIONS/DEPENDENCIES: None.
# TODO:
#   - Compute hand angles for hour and minute hands.
#   - Render a circular face with hands positioned accordingly.
# NOTES: Implementation is complex and not required for initial release.
# ###########################################################################
def print_analog_clock(time: _dt.datetime) -> None:
    raise NotImplementedError("ascii_analog_clock stub")
