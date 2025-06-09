"""ASCII art utilities for rendering clock faces."""

from __future__ import annotations

import datetime as _dt
import math
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
    '.': [
        "     ",
        "     ",
        "     ",
        "     ",
        "  #  ",
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


def print_digital_clock(time: _dt.datetime) -> None:
    """Print ``time`` in HH:MM:SS format as ASCII art digits."""

    digits = compose_ascii_digits(time.strftime("%H:%M:%S"))
    print(digits)


def print_analog_clock(time: _dt.datetime) -> None:
    """Print a simple ASCII analog clock."""

    size = 11
    center = size // 2
    grid: List[List[str]] = [[" " for _ in range(size)] for _ in range(size)]

    # Draw circular outline using a rough stencil of the hour marks
    circle_points = [
        (0, center),
        (1, center + 3),
        (2, center + 4),
        (center, size - 1),
        (size - 3, center + 4),
        (size - 2, center + 3),
        (size - 1, center),
        (size - 2, center - 3),
        (size - 3, center - 4),
        (center, 0),
        (2, center - 4),
        (1, center - 3),
    ]

    for y, x in circle_points:
        if 0 <= y < size and 0 <= x < size:
            grid[y][x] = "#"

    def angle_to_point(angle: float, length: int) -> tuple[int, int]:
        rad = math.radians(angle)
        dy = int(round(-length * math.cos(rad)))
        dx = int(round(length * math.sin(rad)))
        return center + dy, center + dx

    hour_angle = (time.hour % 12 + time.minute / 60) * 30
    minute_angle = time.minute * 6

    hy, hx = angle_to_point(hour_angle, 3)
    my, mx = angle_to_point(minute_angle, 4)

    if 0 <= hy < size and 0 <= hx < size:
        grid[hy][hx] = "H"
    if 0 <= my < size and 0 <= mx < size:
        grid[my][mx] = "M"

    grid[center][center] = "*"

    for row in grid:
        print("".join(row))

