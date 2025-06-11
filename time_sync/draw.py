#!/usr/bin/env python3
"""Terminal drawing helpers for ASCII frame diffs."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import sys
    import numpy as np
    from colorama import Style, Fore, Back
    from time_sync.ascii_kernel_classifier import AsciiKernelClassifier
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
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
    char = default_subunit_to_char_kernel(subunit_data, ramp)
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
    "ramp": None,
    "classifier": None,
}

def default_subunit_batch_to_chars(
    subunit_batch: np.ndarray,
    ramp: str = DEFAULT_DRAW_ASCII_RAMP,
) -> list[str]:
    """Return characters for ``subunit_batch`` using a cached classifier."""
    if _classifier_cache["ramp"] != ramp or _classifier_cache["classifier"] is None:
        classifier = AsciiKernelClassifier(ramp)
        classifier.set_font("fontmapper/FM16/consola.ttf", 16, (16, 16))
        _classifier_cache["ramp"] = ramp
        _classifier_cache["classifier"] = classifier
    else:
        classifier = _classifier_cache["classifier"]

    result = classifier.classify_batch(subunit_batch)
    return result["chars"]


def default_subunit_to_char_kernel(
    subunit_data: np.ndarray,
    ramp: str = DEFAULT_DRAW_ASCII_RAMP,
) -> str:
    """Map a single subunit's pixel data to an ASCII character."""
    return default_subunit_batch_to_chars(np.expand_dims(subunit_data, axis=0), ramp)[0]

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
    changed_subunits: list[tuple[int, int, np.ndarray]], # List of (y_pixel, x_pixel, subunit_pixel_data)
    char_cell_pixel_height: int = 1, # The height of a character cell in pixels
    char_cell_pixel_width: int = 1,  # The width of a character cell in pixels
    subunit_to_char_kernel: callable[[np.ndarray], str] = default_subunit_to_char_kernel,
    active_ascii_ramp: str = DEFAULT_DRAW_ASCII_RAMP, # The ASCII ramp to use for character conversion
    base_row: int = 1, 
    base_col: int = 1
) -> None:
    """Render changed subunits using a kernel to map subunit data to characters.
    
    `y_pixel`, `x_pixel` in `changed_subunits` are 0-indexed top-left pixel coordinates of the subunit.
    `char_cell_pixel_height` and `char_cell_pixel_width` define the dimensions of a single
    character cell in terms of pixels. The `subunit_data` for each entry in `changed_subunits`
    is expected to match these dimensions (or be a part of a larger image from which these
    coordinates are derived). The `subunit_to_char_kernel` converts this `subunit_data`
    (representing one character cell's worth of pixels) into a single display character.
    `base_row`, `base_col` are 1-indexed for ANSI terminal compatibility.
    """
    if char_cell_pixel_height <= 0: char_cell_pixel_height = 1
    if char_cell_pixel_width <= 0: char_cell_pixel_width = 1

    # Batch convert all subunits to characters when using the default kernel
    if subunit_to_char_kernel is default_subunit_to_char_kernel:
        subunit_batch = np.stack([data for _, _, data in changed_subunits], axis=0)
        chars = default_subunit_batch_to_chars(subunit_batch, active_ascii_ramp)
    else:
        chars = [subunit_to_char_kernel(data, active_ascii_ramp) for _, _, data in changed_subunits]

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
    subunit_width: int
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

    Returns:
        A list of tuples. Each tuple contains:
        - y_coord (int): The y-coordinate of the top-left corner of the changed subunit
                         in the original frame.
        - x_coord (int): The x-coordinate of the top-left corner of the changed subunit
                         in the original frame.
        - subunit_data (np.ndarray): A NumPy array representing the pixel data of
                                     the changed subunit, extracted from new_frame.
                                     The shape of subunit_data will be
                                     (actual_subunit_height, actual_subunit_width, ...)
                                     where actual dimensions might be smaller than
                                     subunit_height/subunit_width if at the frame edge.

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

            # Compare the subunits.
            # np.any() checks if any element is True after element-wise comparison.
            # This handles multi-channel (e.g., RGB) data correctly.
            if np.any(current_subunit_old != current_subunit_new):
                # If a change is detected, append the coordinate and a copy of
                # the subunit from the new_frame to the list.
                changed_subunits_list.append((y, x, current_subunit_new.copy()))

    return changed_subunits_list
