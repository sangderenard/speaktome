#!/usr/bin/env python3
"""Terminal drawing helpers for ASCII frame diffs."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import sys
    import numpy as np
    from time_sync.time_sync.ascii_digits import ASCII_RAMP_BLOCK
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def default_subunit_to_char_kernel(
    subunit_data: np.ndarray, ramp: str = ASCII_RAMP_BLOCK
) -> str:
    """
    Stub kernel function to map a subunit of pixel data to a single character.
    This basic stub averages the subunit and returns a block character based on luminance.
    """
    if subunit_data.size == 0:
        return " " # Should not happen with valid subunits
    
    # Assuming subunit_data is (H, W, 3) for RGB
    # For simplicity, convert to grayscale and average luminance
    if subunit_data.ndim == 3 and subunit_data.shape[2] == 3:
        # A simple grayscale conversion: (R+G+B)/3
        luminance_map = np.mean(subunit_data, axis=2)
    elif subunit_data.ndim == 2: # Already grayscale
        luminance_map = subunit_data
    else: # Fallback for unexpected shapes
        return "?"
        
    avg_luminance = np.mean(luminance_map)
    
    # Simple mapping to a character based on average luminance
    # ASCII_RAMP_BLOCK = " .:░▒▓█" (example)
    
    char_index = min(len(ramp) - 1, int((avg_luminance / 255) * len(ramp)))
    return ramp[char_index]


def draw_diff(
    changed_subunits: list[tuple[int, int, np.ndarray]], # List of (y, x, subunit_pixel_data)
    subunit_to_char_kernel: callable[[np.ndarray], str] = default_subunit_to_char_kernel,
    base_row: int = 1, 
    base_col: int = 1
) -> None:
    """Render changed subunits using a kernel to map subunit data to characters.
    
    `y`, `x` in `changed_subunits` are 0-indexed pixel coordinates of the subunit's top-left.
    `base_row`, `base_col` are 1-indexed for ANSI terminal compatibility.
    """
    # This function now assumes that each subunit corresponds to one character cell.
    # The (y,x) coordinates from get_changed_subunits are pixel coordinates.
    # We need to determine which character cell these correspond to.
    # For simplicity, this example assumes subunit_height and subunit_width
    # from get_changed_subunits implicitly define the character cell size.
    # A more robust solution would pass subunit_height/width or derive char_x, char_y.

    for y_pixel, x_pixel, subunit_data in changed_subunits:
        # This is a simplification: assumes (y_pixel, x_pixel) can directly map to terminal rows/cols.
        # In reality, you'd divide y_pixel by subunit_height and x_pixel by subunit_width
        # to get the character cell coordinates if subunits are larger than 1x1 pixel.
        # For this example, let's assume each subunit IS a character cell's pixel data.
        # The (y_pixel, x_pixel) are thus the top-left of the character cell.
        
        char_to_draw = subunit_to_char_kernel(subunit_data)
        
        # Determine average color of the subunit for foreground/background
        # This is a very basic approach; more sophisticated coloring could be used.
        if subunit_data.ndim == 3 and subunit_data.shape[2] == 3: # RGB
            avg_color = np.mean(subunit_data, axis=(0, 1)).astype(int)
            r, g, b = avg_color[0], avg_color[1], avg_color[2]
            # Example: use average color as background, fixed foreground (e.g., white)
            # Or, derive foreground based on contrast with average background.
            # For this example, let's set background to average and print the char.
            # A more advanced kernel could return (char, fg_color, bg_color).
            ansi_color_bg = f"\x1b[48;2;{r};{g};{b}m"
            ansi_color_fg = "\x1b[38;2;255;255;255m" # White foreground
        else: # Grayscale or other
            ansi_color_bg = "" # No specific background color
            ansi_color_fg = "" # No specific foreground color

        terminal_row = base_row + y_pixel # Assuming y_pixel is the target row
        terminal_col = base_col + x_pixel # Assuming x_pixel is the target col

        sys.stdout.write(f"\x1b[{terminal_row};{terminal_col}H{ansi_color_bg}{ansi_color_fg}{char_to_draw}\x1b[0m")
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
