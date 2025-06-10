"""ASCII art utilities for rendering clock faces."""

from __future__ import annotations

import datetime as _dt
import math
from typing import List, Tuple, Optional
# --- END HEADER ---

from colorama import Fore, Style
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    from .theme_manager import ThemeManager # Assuming theme_manager.py is in the same directory
except ImportError:
    PIL_AVAILABLE = False

# ASCII ramp for gradient effect (more dense = brighter/more opaque)
ASCII_RAMP_BLOCK = " .:░▒▓█" # Using block characters for a more solid look
ASCII_RAMP_DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Palette of Colorama colors and their approximate RGB values
# (R, G, B) values are 0-255
COLORAMA_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    (Fore.BLACK, (0, 0, 0)),
    (Fore.RED, (128, 0, 0)),
    (Fore.GREEN, (0, 128, 0)),
    (Fore.YELLOW, (128, 128, 0)),
    (Fore.BLUE, (0, 0, 128)),
    (Fore.MAGENTA, (128, 0, 128)),
    (Fore.CYAN, (0, 128, 128)),
    (Fore.WHITE, (192, 192, 192)), # Often light gray
    (Fore.LIGHTBLACK_EX, (128, 128, 128)), # Often dark gray
    (Fore.LIGHTRED_EX, (255, 0, 0)),
    (Fore.LIGHTGREEN_EX, (0, 255, 0)),
    (Fore.LIGHTYELLOW_EX, (255, 255, 0)),
    (Fore.LIGHTBLUE_EX, (0, 0, 255)),
    (Fore.LIGHTMAGENTA_EX, (255, 0, 255)),
    (Fore.LIGHTCYAN_EX, (0, 255, 255)),
    (Fore.LIGHTWHITE_EX, (255, 255, 255)),
]
# For convenience, a "bright" version of the palette using Style.BRIGHT
COLORAMA_PALETTE_BRIGHT: List[Tuple[str, Tuple[int, int, int]]] = [
    (Style.BRIGHT + Fore.BLACK, (128, 128, 128)), # Bright black is grey
    (Style.BRIGHT + Fore.RED, (255, 0, 0)),
    (Style.BRIGHT + Fore.GREEN, (0, 255, 0)),
    (Style.BRIGHT + Fore.YELLOW, (255, 255, 0)),
    (Style.BRIGHT + Fore.BLUE, (0, 0, 255)),
    (Style.BRIGHT + Fore.MAGENTA, (255, 0, 255)),
    (Style.BRIGHT + Fore.CYAN, (0, 255, 255)),
    (Style.BRIGHT + Fore.WHITE, (255, 255, 255)),
]
# We'll primarily use the bright palette for more vibrant foregrounds.
ACTIVE_COLOR_PALETTE = COLORAMA_PALETTE_BRIGHT

# Keep old digits for a potential fallback if PIL is not available, though not used in current impl.
_DIGITS_OLD = {
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


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    if not PIL_AVAILABLE:
        return None
    common_fonts = ["DejaVuSansMono.ttf", "cour.ttf", "Consolas.ttf", "Menlo.ttf", "LiberationMono-Regular.ttf"]
    for font_name in common_fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    try: # Fallback to a generic sans-serif if monospace not found
        return ImageFont.truetype("arial.ttf", size)
    except IOError: # PIL's default bitmap font if no ttf found
        try:
            return ImageFont.load_default() # This font is very small
        except IOError:
            return None


def _find_closest_color(rgb_pixel: Tuple[int, int, int], palette: List[Tuple[str, Tuple[int, int, int]]]) -> str:
    """Finds the closest color in the palette to the given RGB pixel."""
    if not palette:
        return Fore.WHITE # Fallback
    
    min_dist = float('inf')
    closest_color_code = palette[0][0]
    r1, g1, b1 = rgb_pixel

    for color_code, (r2, g2, b2) in palette:
        dist = (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2
        if dist < min_dist:
            min_dist = dist
            closest_color_code = color_code
    return closest_color_code

def _image_to_ascii_colored(
    image: Image.Image, # Expects RGBA image
    ramp: str = ASCII_RAMP_BLOCK,
    ascii_bg_fill: str = Fore.BLACK + Style.DIM + " ", # Colorama string for background char
    target_ascii_width: int = 20,
    target_ascii_height: int = 14,
    bg_alpha_threshold: int = 10, # Alpha below this is considered background
) -> str:
    if not PIL_AVAILABLE:
        return "Pillow library not found."

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Resize image so each pixel corresponds to an ASCII character cell
    img_for_ascii = image.resize((target_ascii_width, target_ascii_height), Image.Resampling.LANCZOS)

    ascii_art_rows = []
    ramp_len = len(ramp)

    for y_idx in range(img_for_ascii.height):
        line_chars = []
        for x_idx in range(img_for_ascii.width):
            r, g, b, a = img_for_ascii.getpixel((x_idx, y_idx))

            if a < bg_alpha_threshold:
                # Background pixel (transparent or nearly transparent)
                colored_char = ascii_bg_fill[0:-1] + ramp[0] + Style.RESET_ALL if ascii_bg_fill else ramp[0]
            else:
                # Foreground pixel
                char_color_code = _find_closest_color((r, g, b), ACTIVE_COLOR_PALETTE)
                luminance = int(0.299 * r + 0.587 * g + 0.114 * b) # Standard luminance calc
                char_index = min(ramp_len - 1, int((luminance / 255) * ramp_len))
                char = ramp[char_index]
                colored_char = char_color_code + char + Style.RESET_ALL
            line_chars.append(colored_char)
        ascii_art_rows.append("".join(line_chars))
    return "\n".join(ascii_art_rows)


def compose_ascii_digits(
    text: str,
    font_size: int = 32,
    target_ascii_width: int = 60,
    target_ascii_height: int = 12,
    text_color_on_image: Tuple[int,int,int,int] = (255,255,255,255), # White opaque for main text
    shadow_color_on_image: Tuple[int,int,int,int] = (0,0,0,100), # Semi-transparent black for shadow
    outline_color_on_image: Tuple[int,int,int,int] = (0,0,0,200), # Opaque black for outline
    outline_thickness: int = 1,
    shadow_offset: Tuple[int, int] = (2, 2), # (dx, dy) for shadow from text
    backdrop_image_path: Optional[str] = None,
    final_ascii_bg_fill: str = Fore.BLACK + Style.DIM + " ", # Used for transparent areas
) -> str:
    if not PIL_AVAILABLE:
        return f"Pillow not available. Text: {text}"

    font = _get_font(font_size)
    if not font: # Handle case where load_default also fails
        return f"Font not found (size {font_size}). Text: {text}"

    # Determine text bounding box to size the image correctly
    try: # For TrueType/FreeType fonts
        text_bbox = font.getbbox(text)
        text_x_offset, text_y_offset = text_bbox[0], text_bbox[1]
        text_render_width, text_render_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError: # Fallback for bitmap fonts (like load_default())
        text_render_width, text_render_height = font.getsize(text) # Deprecated but works for bitmap
        text_x_offset, text_y_offset = 0, 0 # Bitmap fonts usually render from (0,0) relative to anchor

    # Image dimensions need to accommodate text and shadow
    img_width = text_render_width + abs(shadow_offset[0]) + outline_thickness * 2 + 4 # Extra padding
    img_height = text_render_height + abs(shadow_offset[1]) + outline_thickness * 2 + 4 # Extra padding

    # Create base image: either backdrop or new transparent image
    if backdrop_image_path:
        try:
            image = Image.open(backdrop_image_path).convert("RGBA")
            image = image.resize((img_width, img_height), Image.Resampling.LANCZOS)
        except FileNotFoundError:
            return f"Backdrop image not found: {backdrop_image_path}"
        except Exception as e:
            return f"Error loading backdrop: {e}"
    else:
        image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Position for drawing text (top-left of the text rendering area)
    # Adjust for shadow offset: if shadow is positive, text needs to be at (0,0) relative to its box
    # and shadow at (offset_x, offset_y). If shadow is negative, text needs to be at (abs(offset_x), abs(offset_y))
    # Also adjust for outline thickness
    padding = 2 + outline_thickness
    base_x = padding + (abs(shadow_offset[0]) if shadow_offset[0] < 0 else 0) - text_x_offset
    base_y = padding + (abs(shadow_offset[1]) if shadow_offset[1] < 0 else 0) - text_y_offset

    # 1. Draw Shadow
    shadow_pos_x = base_x + shadow_offset[0]
    shadow_pos_y = base_y + shadow_offset[1]
    # Draw shadow with its own (potentially blurred) outline if desired, or just plain
    if shadow_color_on_image[3] > 0: # Only draw if shadow is visible
        draw.text((shadow_pos_x, shadow_pos_y), text, font=font, fill=shadow_color_on_image)

    # 2. Draw Main Text with Outline (on top of shadow)
    # Pillow's stroke is centered on the text path, so text is drawn over half the stroke.
    if isinstance(font, ImageFont.FreeTypeFont) and outline_thickness > 0 and outline_color_on_image[3] > 0:
        draw.text((base_x, base_y), text, font=font, fill=text_color_on_image,
                  stroke_width=outline_thickness, stroke_fill=outline_color_on_image)
    else: # Fallback for bitmap fonts or no outline
        draw.text((base_x, base_y), text, font=font, fill=text_color_on_image)

    return _image_to_ascii_colored(
        image,
        ramp=ASCII_RAMP_BLOCK,
        ascii_bg_fill=final_ascii_bg_fill,
        target_ascii_width=target_ascii_width,
        target_ascii_height=target_ascii_height,
    )


def print_digital_clock(
    time: _dt.datetime,
    backdrop_image_path: Optional[str] = None,
    theme_manager: Optional[ThemeManager] = None,
    **override_params
) -> None:
    """Print ``time`` in HH:MM:SS format as pretty, colored ASCII art."""
    digits_str = time.strftime("%H:%M:%S")
    
    if not PIL_AVAILABLE:
        print(Fore.RED + "Pillow library not installed. Cannot display enhanced digital clock." + Style.RESET_ALL)
        print(Style.BRIGHT + Fore.CYAN + digits_str + Style.RESET_ALL) # Simple fallback
        return

    # Configuration for the digital clock appearance
    # Start with defaults, override with override_params (from JSON), then theme (for colors)
    params = {
        "font_size": 40,
        "target_ascii_width": 60,
        "target_ascii_height": 7,
        "text_color_on_image": (100, 220, 255, 255),  # Light Blue text
        "outline_color_on_image": (0, 50, 100, 255),   # Dark Blue outline
        "outline_thickness": 2,
        "shadow_color_on_image": (20, 20, 20, 150),   # Dark grey shadow
        "shadow_offset": (2,2),
        "final_ascii_bg_fill": Fore.BLACK + " "
    }
    params.update(override_params)

    if theme_manager and theme_manager.current_theme.palette:
        palette = theme_manager.current_theme.palette
        params["text_color_on_image"] = tuple(palette.get("text", params["text_color_on_image"]))
        params["outline_color_on_image"] = tuple(palette.get("outline", params["outline_color_on_image"]))
        params["shadow_color_on_image"] = tuple(palette.get("shadow", params["shadow_color_on_image"]))

    # Create the raw Pillow image using compose_ascii_digits's logic
    # We need to call the internal image creation part of compose_ascii_digits
    # For now, let's assume compose_ascii_digits is refactored or we replicate its image creation here.
    # Simplified: Re-calling compose_ascii_digits which returns a string.
    # Ideally, compose_ascii_digits would have a part that returns the image.
    # Let's modify compose_ascii_digits to optionally return the image.

    # This is a placeholder for getting the raw image.
    # You'd refactor compose_ascii_digits to have an internal _create_image function
    raw_image = Image.new("RGBA", (params["target_ascii_width"]*10, params["target_ascii_height"]*10)) # Dummy
    # Actual image creation should happen here using params.
    # For a quick test, we'll just use the string output and skip theme image effects for digital.
    # TODO: Refactor compose_ascii_digits to allow image manipulation before ASCII conversion.
    
    # For now, we'll directly call compose_ascii_digits and it won't have image effects from theme_manager
    # but will use the ASCII ramp.
    final_image_str = compose_ascii_digits(
        digits_str,
        backdrop_image_path=backdrop_image_path,
        **params
    )
    print(final_image_str) # This call needs to be updated if compose_ascii_digits changes ramp


def print_analog_clock(
    time: _dt.datetime,
    canvas_size_px: int = 120, # Pixel dimensions of the internal canvas for drawing
    target_ascii_diameter: int = 22, # Approximate diameter of the clock in ASCII characters
    face_color_on_image: Tuple[int,int,int,int] = (70, 70, 70, 255),
    marks_color_on_image: Tuple[int,int,int,int] = (220, 220, 200, 255), # Off-white
    hour_hand_color_on_image: Tuple[int,int,int,int] = (100, 220, 100, 255), # Light green
    minute_hand_color_on_image: Tuple[int,int,int,int] = (120, 255, 120, 255), # Brighter green
    second_hand_color_on_image: Tuple[int,int,int,int] = (255, 100, 100, 255), # Red
    center_dot_color_on_image: Tuple[int,int,int,int] = (255, 255, 255, 255), # White
    clock_drawing_rect_on_canvas: Optional[Tuple[int, int, int, int]] = None, # (x, y, width, height)
    backdrop_image_path: Optional[str] = None,
    final_ascii_bg_fill: str = Fore.BLACK + " ", # Black background for ASCII
    theme_manager: Optional[ThemeManager] = None,
    **override_params
) -> None:
    if not PIL_AVAILABLE:
        print(Fore.RED + "Pillow library not installed. Cannot display enhanced analog clock." + Style.RESET_ALL)
        print(Style.BRIGHT + Fore.MAGENTA + time.strftime("%H:%M:%S") + " (Analog)" + Style.RESET_ALL)
        return

    params = {
        "canvas_size_px": canvas_size_px,
        "target_ascii_diameter": target_ascii_diameter,
        "face_color_on_image": face_color_on_image,
        "marks_color_on_image": marks_color_on_image,
        "hour_hand_color_on_image": hour_hand_color_on_image,
        "minute_hand_color_on_image": minute_hand_color_on_image,
        "second_hand_color_on_image": second_hand_color_on_image,
        "center_dot_color_on_image": center_dot_color_on_image,
        "clock_drawing_rect_on_canvas": clock_drawing_rect_on_canvas,
        "final_ascii_bg_fill": final_ascii_bg_fill,
    }
    params.update(override_params)

    if theme_manager and theme_manager.current_theme.palette:
        palette = theme_manager.current_theme.palette
        # Example: map theme palette keys to analog clock parts
        params["face_color_on_image"] = tuple(palette.get("outline", params["face_color_on_image"]))
        params["marks_color_on_image"] = tuple(palette.get("text", params["marks_color_on_image"]))
        params["hour_hand_color_on_image"] = tuple(palette.get("accent1", params["hour_hand_color_on_image"]))
        params["minute_hand_color_on_image"] = tuple(palette.get("accent2", params["minute_hand_color_on_image"]))
        params["second_hand_color_on_image"] = tuple(palette.get("shadow", params["second_hand_color_on_image"])) # Reusing shadow for seconds

    # Create the base canvas: either from backdrop or a new transparent image
    if backdrop_image_path:
        try:
            base_canvas_image = Image.open(backdrop_image_path).convert("RGBA")
            base_canvas_image = base_canvas_image.resize((params["canvas_size_px"], params["canvas_size_px"]), Image.Resampling.LANCZOS)
            if theme_manager and theme_manager.current_theme.invert_backdrop:
                base_canvas_image = Image.eval(base_canvas_image, lambda x: 255 - x) # Invert
        except FileNotFoundError:
            print(Fore.RED + f"Backdrop image not found: {backdrop_image_path}" + Style.RESET_ALL)
            base_canvas_image = Image.new("RGBA", (canvas_size_px, canvas_size_px), (0,0,0,0)) # Fallback
        except Exception as e:
            print(Fore.RED + f"Error loading backdrop: {e}" + Style.RESET_ALL)
            base_canvas_image = Image.new("RGBA", (canvas_size_px, canvas_size_px), (0,0,0,0)) # Fallback
    else:
        base_canvas_image = Image.new("RGBA", (params["canvas_size_px"], params["canvas_size_px"]), (0, 0, 0, 0))

    # Determine the drawing surface and parameters
    if params["clock_drawing_rect_on_canvas"]:
        rect_x, rect_y, rect_width, rect_height = params["clock_drawing_rect_on_canvas"]
        # Create a temporary canvas for drawing just the clock elements
        clock_elements_img = Image.new("RGBA", (rect_width, rect_height), (0, 0, 0, 0))
        draw_on = ImageDraw.Draw(clock_elements_img) # Draw on this temporary image
        
        # Drawing parameters for the clock elements canvas
        current_center_x = rect_width // 2
        current_center_y = rect_height // 2
        effective_diameter_for_elements = min(rect_width, rect_height)
    else:
        # Drawing directly on the base_canvas_image
        draw_on = ImageDraw.Draw(base_canvas_image) # Draw on the main canvas
        
        current_center_x = params["canvas_size_px"] // 2
        current_center_y = params["canvas_size_px"] // 2
        effective_diameter_for_elements = params["canvas_size_px"]
        # No separate clock_elements_img needed in this case

    current_radius = effective_diameter_for_elements // 2 - (effective_diameter_for_elements // 20) # Padding

    # --- Drawing operations will use: draw_on, current_center_x, current_center_y, current_radius ---
    # --- Line widths and element sizes will scale with effective_diameter_for_elements ---

    face_stroke_width = max(1, effective_diameter_for_elements // 50)
    major_mark_stroke_width = max(1, effective_diameter_for_elements // 60)
    minor_mark_stroke_width = max(1, effective_diameter_for_elements // 100)
    hour_hand_stroke_width = max(2, effective_diameter_for_elements // 25)
    minute_hand_stroke_width = max(2, effective_diameter_for_elements // 35)
    second_hand_stroke_width = max(1, effective_diameter_for_elements // 50)
    center_dot_radius_px = max(2, effective_diameter_for_elements // 30)

    # Clock face
    draw_on.ellipse(
        (current_center_x - current_radius, current_center_y - current_radius, 
         current_center_x + current_radius, current_center_y + current_radius),
        outline=params["face_color_on_image"], width=face_stroke_width
    )

    # Hour/Minute marks
    for i in range(12):
        angle = math.radians(i * 30 - 90)
        is_major_mark = (i % 3 == 0)
        start_factor = 0.88 if is_major_mark else 0.92 # Adjusted for potentially smaller radii
        # mark_len = current_radius * (1 - start_factor) # Not directly used, but for context
        x1 = current_center_x + int(current_radius * start_factor * math.cos(angle))
        y1 = current_center_y + int(current_radius * start_factor * math.sin(angle))
        x2 = current_center_x + int(current_radius * math.cos(angle))
        y2 = current_center_y + int(current_radius * math.sin(angle))
        mark_width = major_mark_stroke_width if is_major_mark else minor_mark_stroke_width
        draw_on.line((x1, y1, x2, y2), fill=params["marks_color_on_image"], width=mark_width)

    # Hands
    h_angle = math.radians((time.hour % 12 + time.minute / 60) * 30 - 90)
    m_angle = math.radians((time.minute + time.second / 60) * 6 - 90)
    s_angle = math.radians(time.second * 6 - 90)

    h_len = current_radius * 0.55
    draw_on.line((current_center_x, current_center_y, current_center_x + int(h_len * math.cos(h_angle)), current_center_y + int(h_len * math.sin(h_angle))),
              fill=params["hour_hand_color_on_image"], width=hour_hand_stroke_width)
    m_len = current_radius * 0.75
    draw_on.line((current_center_x, current_center_y, current_center_x + int(m_len * math.cos(m_angle)), current_center_y + int(m_len * math.sin(m_angle))),
              fill=params["minute_hand_color_on_image"], width=minute_hand_stroke_width)
    s_len = current_radius * 0.8
    draw_on.line((current_center_x, current_center_y, current_center_x + int(s_len * math.cos(s_angle)), current_center_y + int(s_len * math.sin(s_angle))),
              fill=params["second_hand_color_on_image"], width=second_hand_stroke_width)

    # Center dot
    draw_on.ellipse((current_center_x - center_dot_radius_px, current_center_y - center_dot_radius_px, 
                  current_center_x + center_dot_radius_px, current_center_y + center_dot_radius_px), 
                 fill=params["center_dot_color_on_image"])

    # If clock elements were drawn on a separate image, paste them onto the base_canvas_image
    if params["clock_drawing_rect_on_canvas"]:
        rect_x, rect_y, _, _ = params["clock_drawing_rect_on_canvas"] # Get original x,y for pasting
        base_canvas_image.paste(clock_elements_img, (rect_x, rect_y), clock_elements_img) # Use alpha for transparency

    # Apply theme effects and post-processing
    image_to_convert = base_canvas_image
    ascii_ramp_to_use = ASCII_RAMP_BLOCK # Default

    if theme_manager:
        image_to_convert = theme_manager.apply_effects(image_to_convert)
        image_to_convert = theme_manager.apply_theme(image_to_convert) # Handles invert_clock
        ascii_ramp_to_use = theme_manager.get_current_ascii_ramp()

    # Convert to ASCII. Character aspect ratio heuristic: width should be ~2x height for square look.
    # So, target_ascii_width = target_ascii_diameter * 2 (approx)
    # target_ascii_height = target_ascii_diameter
    ascii_art = _image_to_ascii_colored(
        image_to_convert,
        ramp=ascii_ramp_to_use,
        ascii_bg_fill=params["final_ascii_bg_fill"],
        target_ascii_width=int(params["target_ascii_diameter"] * 1.8), # Adjust for char aspect
        target_ascii_height=params["target_ascii_diameter"],
        bg_alpha_threshold=10
    )
    print(ascii_art)


def render_ascii_to_array(text: str, **kwargs) -> np.ndarray:
    """Return ASCII art for ``text`` as a ``numpy.ndarray``."""
    ascii_str = compose_ascii_digits(text, **kwargs)
    rows = ascii_str.splitlines()
    return np.array([list(row) for row in rows], dtype="<U1")
