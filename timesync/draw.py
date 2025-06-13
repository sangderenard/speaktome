#!/usr/bin/env python3
"""Terminal drawing helpers for ASCII frame diffs."""
from __future__ import annotations

import sys
import numpy as np
from colorama import Style, Fore, Back
from timesync.ascii_kernel_classifier import AsciiKernelClassifier
# --- END HEADER ---


# Default ASCII ramp used if no specific ramp is provided to drawing functions.
DEFAULT_DRAW_ASCII_RAMP = " .:░▒▓█"


def flexible_subunit_kernel(
    subunit_data: np.ndarray,
    ramp: str,
    mode: str = "ascii",
) -> str | np.ndarray:
    """Return a representation for ``subunit_data`` based on ``mode``.

    Parameters
    ----------
    subunit_data:
        Pixel array representing one character cell. Shape ``(H, W, 3)``.
    ramp:
        ASCII ramp string ordered from darkest to brightest.
    mode:
        ``"ascii"``  -> return a single character.
        ``"raw"``    -> return the pixel array untouched.
        ``"hybrid"`` -> return a small image of the ASCII character.
    """
    if mode == "raw":
        return subunit_data
    char = default_subunit_batch_to_chars(
        np.expand_dims(subunit_data, axis=0), ramp
    )[0]
    if mode == "ascii":
        return char
    if mode == "hybrid":
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return subunit_data
        h, w = subunit_data.shape[:2]
        img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_size = max(1, min(w, h))
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        y = (h - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), char, fill=(255, 255, 255), font=font)
        return np.array(img)
    raise ValueError(f"Unknown mode: {mode}")

# Module-level cache for classifier and ramp
_classifier_cache = {
    "ramp_char_size": None, # Cache key will be a tuple (ramp, char_width, char_height)
    "classifier": None,
}

def default_subunit_batch_to_chars(
    subunit_batch: np.ndarray,
    ramp: str = DEFAULT_DRAW_ASCII_RAMP,
    char_width: int = 16, # Add parameters for desired char_size
    char_height: int = 16, # These will be the actual cell_w, cell_h from clock_demo
) -> list[str]:
    """Return characters for ``subunit_batch`` using a cached classifier."""
    # Removed: (subunit_height, subunit_width) = get_char_cell_dims(),
    # Removed: print(subunit_height, subunit_width)
    # Removed: exit()
    # Now use the passed-in char_width and char_height

    cache_key = (ramp, char_width, char_height)
    if _classifier_cache["ramp_char_size"] != cache_key or _classifier_cache["classifier"] is None:
        # AsciiKernelClassifier expects char_size as (width, height)
        classifier = AsciiKernelClassifier(ramp, char_size=(char_width, char_height))
        # font_size is for rendering reference characters, which are then scaled to char_size
        classifier.set_font(font_path="fontmapper/FM16/consola.ttf", font_size=16, char_size=(char_width, char_height))
        _classifier_cache["ramp_char_size"] = cache_key
        _classifier_cache["classifier"] = classifier
    else:
        classifier = _classifier_cache["classifier"]

    result = classifier.classify_batch(subunit_batch)
    return result["chars"]


def draw_text_overlay(
    row: int,
    col: int,
    text: str,
    color_code: str = Style.RESET_ALL # Default to terminal default color
) -> None:
    """Draws text directly to the terminal at a specific 1-indexed row and column."""
    # Use \x1b[<row>;<col>H for cursor positioning
    sys.stdout.write(f"\x1b[{row};{col}H{color_code}{text}{Style.RESET_ALL}")
    sys.stdout.flush()

def draw_diff(
    changed_subunits: list[tuple[int, int, np.ndarray]],  # List of (y_pixel, x_pixel, subunit_pixel_data)
    char_cell_pixel_height: int = 1,  # The height of a character cell in pixels
    char_cell_pixel_width: int = 1,  # The width of a character cell in pixels
    subunit_to_char_kernel: callable[[np.ndarray, str, int, int], list[str]] = default_subunit_batch_to_chars, # Update signature
    active_ascii_ramp: str = DEFAULT_DRAW_ASCII_RAMP,  # The ASCII ramp to use for character conversion
    base_row: int = 1, 
    base_col: int = 1
) -> None:
    """Render changed subunits using a kernel to map subunit data to characters.
    
    `y_pixel`, `x_pixel` in `changed_subunits` are 0-indexed top-left pixel coordinates of the subunit.
    `char_cell_pixel_height` and `char_cell_pixel_width` define the dimensions of a single
    character cell in terms of pixels. The `subunit_data` for each entry in `changed_subunits`
    is expected to match these dimensions (or be a part of a larger image from which these
    coordinates are derived). The `subunit_to_char_kernel` is expected to accept the entire
    batch of subunits and return a list of ASCII characters for display.
    `base_row`, `base_col` are 1-indexed for ANSI terminal compatibility.
    """
    if char_cell_pixel_height <= 0: char_cell_pixel_height = 1
    if char_cell_pixel_width <= 0: char_cell_pixel_width = 1
    if not changed_subunits:
        return
    subunit_batch = np.stack([data for _, _, data in changed_subunits], axis=0)
    # Pass the actual char_cell_pixel_width and char_cell_pixel_height to the kernel
    chars = subunit_to_char_kernel(subunit_batch, active_ascii_ramp, char_cell_pixel_width, char_cell_pixel_height)

    for (y_pixel, x_pixel, subunit_data), char_to_draw in zip(changed_subunits, chars):
        char_y = y_pixel // char_cell_pixel_height
        char_x = x_pixel // char_cell_pixel_width

        # Determine average color of the subunit for foreground/background
        if subunit_data.ndim == 3 and subunit_data.shape[2] == 3:  # RGB
            avg_color = np.mean(subunit_data, axis=(0, 1)).astype(int)
            r, g, b = avg_color[0], avg_color[1], avg_color[2]
            ansi_color_bg = f"\x1b[48;2;{r};{g};{b}m"
            ansi_color_fg = "\x1b[38;2;255;255;255m"
        else:
            ansi_color_bg = ""
            ansi_color_fg = ""

        terminal_row = base_row + char_y
        terminal_col = base_col + char_x

        sys.stdout.write(
            f"\x1b[{terminal_row};{terminal_col}H{ansi_color_bg}{ansi_color_fg}{char_to_draw}\x1b[0m"
        )
    sys.stdout.flush()


from typing import List, Tuple

def get_changed_subunits(
    old_frame: np.ndarray,
    new_frame: np.ndarray,
    subunit_height: int,
    subunit_width: int,
    loss_threshold: float = 0.0,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Compares two frames and returns subunits from the new_frame that have changed.

    The frames are expected to be NumPy arrays, typically representing image data
    with shape (height, width) or (height, width, channels).

    Args:
        old_frame: The previous frame.
        new_frame: The current frame.
        subunit_height: The height of the rectangular subunits to check.
        subunit_width: The width of the rectangular subunits to check.
        loss_threshold:
            Mean absolute difference threshold for considering a subunit
            changed. A value of ``0.0`` means any difference will be
            returned.

    Returns:
        A list of tuples. Each tuple contains:
        - y_coord (int): The y-coordinate of the top-left corner of the changed subunit
                         in the original frame.
        - x_coord (int): The x-coordinate of the top-left corner of the changed subunit
                         in the original frame.


    Raises:
        ValueError: If old_frame and new_frame do not have the same shape,
                    or if subunit dimensions are not positive.
    """
    if old_frame.shape != new_frame.shape:
        raise ValueError("Old and new frames must have the same shape.")
    if not (subunit_height > 0 and subunit_width > 0):
        raise ValueError("Subunit height and width must be positive integers.")

    frame_height, frame_width = new_frame.shape[0], new_frame.shape[1]
    changed_subunits_list: List[Tuple[int, int, np.ndarray]] = []

    for y in range(0, frame_height, subunit_height):
        for x in range(0, frame_width, subunit_width):
            # Determine the actual end coordinates for the current subunit,
            # handling cases where the subunit might extend beyond frame boundaries.
            y_end = min(y + subunit_height, frame_height)
            x_end = min(x + subunit_width, frame_width)

            # Extract the current subunit from both old and new frames.
            current_subunit_old = old_frame[y:y_end, x:x_end]
            current_subunit_new = new_frame[y:y_end, x:x_end]

            # Short-circuit: if the slice is smaller than requested, mark it
            # changed without comparing to ``old_frame``.
            if (
                current_subunit_new.shape[0] != subunit_height
                or current_subunit_new.shape[1] != subunit_width
            ):
                changed_subunits_list.append((y, x, current_subunit_new.copy()))
                continue

            # Compare the subunits using a loss metric. ``loss_threshold``
            # represents the mean absolute difference required to flag a
            # change. Multi-channel data is handled by computing the mean
            # across all channels.
            loss = np.mean(
                np.abs(
                    current_subunit_old.astype(np.int16)
                    - current_subunit_new.astype(np.int16)
                )
            )
            if loss > loss_threshold:
                changed_subunits_list.append((y, x, current_subunit_new.copy()))


    return changed_subunits_list
