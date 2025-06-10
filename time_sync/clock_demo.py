#!/usr/bin/env python3
"""Demonstrate time synchronization with analog and digital clocks.

The demo runs until ``q`` or ``quit`` is received on standard input.
"""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import datetime as _dt
    import argparse
    import time
    import json
    import os
    import threading
    import sys
    import numpy as np
    from colorama import Style, Fore, Back  # For colored terminal output
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from time_sync import (
    get_offset, sync_offset,
    print_analog_clock, print_digital_clock,
    init_colorama_for_windows, reset_cursor_to_top, full_clear_and_reset_cursor,
)
from time_sync.time_sync.theme_manager import (
    ThemeManager,
    ClockTheme,
)
from time_sync.time_sync.render_backend import RenderingBackend
from time_sync.frame_buffer import PixelFrameBuffer
from time_sync.render_thread import render_loop
from time_sync.draw import draw_diff
from time_sync.time_sync.ascii_digits import (
    compose_ascii_digits,
    ASCII_RAMP_BLOCK,
)
from time_sync.draw import draw_text_overlay  # Import the new text drawing function
from PIL import Image
import queue

# Platform-specific input handling (adapted from AGENTS/tools/dev_group_menu.py)
if os.name == 'nt':  # Windows
    pass # Windows specific input will be handled directly in input_thread_fn
else:  # Unix-like
    import select
    import sys
    import termios
    import tty
    def getch_timeout(timeout_seconds: float) -> str | None:
        """Get a single character with timeout on Unix-like systems."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ready_to_read, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
            if ready_to_read:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# --- END HEADER ---



def parse_rect(rect_str: str) -> tuple[int, int, int, int] | None:
    """Parses a string 'x,y,width,height' into a tuple of ints."""
    try:
        parts = [int(p.strip()) for p in rect_str.split(',')]
        if len(parts) == 4:
            return tuple(parts) # type: ignore
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"Invalid rect format: '{rect_str}'. Expected 'x,y,width,height'.")


def load_config_from_json(config_path: str, clock_identifier: Optional[str] = None) -> dict:
    """Loads configuration from a JSON file."""
    if clock_identifier:
        config_path = f"{clock_identifier}_config.json"

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {config_path}. Using defaults.")
        except Exception as e:
            print(f"Warning: Error loading {config_path}: {e}. Using defaults.")
    return {}

def save_config_to_json(config_data: dict, clock_identifier: str) -> None:
    """Saves configuration to a JSON file."""
    config_path = f"{clock_identifier}_config.json"
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")

def interactive_configure_mode(
    clock_identifier: str, # e.g., "slow_analog", "fast_digital"
    initial_config: dict, 
    theme_manager: ThemeManager # Pass theme_manager for backdrop access
    ):
    """Interactive mode for configuring clock appearance."""
    config = initial_config.copy()
    
    clock_type = "analog" if "analog" in clock_identifier else "digital"
    is_fast_clock = "fast" in clock_identifier

    print(f"Entering {clock_identifier} clock configuration mode...")
    current_backdrop = theme_manager.current_theme.current_backdrop_path
    print(f"Using backdrop: {current_backdrop if current_backdrop else 'None'}")
    print("Press 's' to save, 'q' to quit without saving.")
    time.sleep(1)

    adjustment_step = 2
    
    while True:
        full_clear_and_reset_cursor()
        current_time = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc) # Or use time_sync.now()
        stopwatch_td = _dt.timedelta(seconds=time.perf_counter()) # Dummy stopwatch for config

        time_obj_for_config = stopwatch_td if is_fast_clock else current_time

        print(f"--- {clock_identifier.upper()} CLOCK CONFIGURATION ---")
        if clock_type == "analog":
            # Ensure rect is a tuple if it exists
            rect = config.get("clock_drawing_rect_on_canvas")
            if isinstance(rect, list) and len(rect) == 4:
                config["clock_drawing_rect_on_canvas"] = tuple(rect)
            
            # Use theme_manager's current backdrop for preview
            print_analog_clock(time_obj_for_config, 
                               backdrop_image_path=theme_manager.current_theme.current_backdrop_path, 
                               theme_manager=theme_manager, **config)
            print("\n--- Current Analog Config ---")
            rect = config.get("clock_drawing_rect_on_canvas", "Not set")
            print(f"Drawing Rect (x,y,w,h): {rect} (Use Arrow keys to move)")
            print(f"Resize Rect: Width (J/L), Height (I/K)")
            print(f"ASCII Diameter: {config.get('target_ascii_diameter', 'Default')} (+/-)")
            print(f"Canvas Size PX: {config.get('canvas_size_px', 'Default')} (c/C)")
            help_text = "[Arrows]:Move [J/L]:Width [I/K]:Height [+/-]:ASCII Dia. [c/C]:Canvas Px [s]:Save [q]:Quit"

        elif clock_type == "digital":
            print_digital_clock(time_obj_for_config, 
                                backdrop_image_path=theme_manager.current_theme.current_backdrop_path,
                                theme_manager=theme_manager, 
                                **config)
            print("\n--- Current Digital Config ---")
            print(f"ASCII Width: {config.get('target_ascii_width', 'Default')} (←/→)")
            print(f"ASCII Height: {config.get('target_ascii_height', 'Default')} (↑/↓)")
            print(f"Font Size: {config.get('font_size', 'Default')} (+/-)")
            help_text = "[←→]: ASCII Width [↑↓]: ASCII Height [+/-]: Font Size [s]:Save [q]:Quit"
        
        print(f"\n{help_text}")
        
        key = getch_timeout(0.15) # Shorter timeout for better responsiveness
        if key is None:
            continue

        if key == 'q':
            break
        elif key == 's':
            save_config_to_json(config, clock_identifier)
            print("Configuration saved. Exiting config mode.")
            time.sleep(1)
            break

        if clock_type == "analog":
            rect = list(config.get("clock_drawing_rect_on_canvas", [20, 20, 80, 80])) # Default if not set
            if key == '\x1b': # Arrow key prefix
                next_key1 = getch_timeout(0.1)
                if next_key1 == '[':
                    next_key2 = getch_timeout(0.1)
                    if next_key2 == 'D': rect[0] -= adjustment_step # Left
                    elif next_key2 == 'C': rect[0] += adjustment_step # Right
                    elif next_key2 == 'A': rect[1] -= adjustment_step # Up
                    elif next_key2 == 'B': rect[1] += adjustment_step # Down
            elif key == 'L': rect[2] += adjustment_step # Resize width + (Shift+Right)
            elif key == 'J': rect[2] -= adjustment_step # Resize width - (Shift+Left)
            elif key == 'I': rect[3] -= adjustment_step # Resize height - (Shift+Up)
            elif key == 'K': rect[3] += adjustment_step # Resize height + (Shift+Down)
            
            config["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect) # Ensure non-negative

            if key == '+': config["target_ascii_diameter"] = config.get("target_ascii_diameter", 22) + 1
            elif key == '-': config["target_ascii_diameter"] = max(5, config.get("target_ascii_diameter", 22) - 1)
            elif key == 'C': config["canvas_size_px"] = config.get("canvas_size_px", 120) + 10
            elif key == 'c': config["canvas_size_px"] = max(20, config.get("canvas_size_px", 120) - 10)

        elif clock_type == "digital":
            if key == '\x1b': # Arrow key prefix
                next_key1 = getch_timeout(0.1)
                if next_key1 == '[':
                    next_key2 = getch_timeout(0.1)
                    if next_key2 == 'D': config["target_ascii_width"] = max(10, config.get("target_ascii_width", 60) - adjustment_step) # Left
                    elif next_key2 == 'C': config["target_ascii_width"] = config.get("target_ascii_width", 60) + adjustment_step # Right
                    elif next_key2 == 'A': config["target_ascii_height"] = max(3, config.get("target_ascii_height", 7) - 1) # Up
                    elif next_key2 == 'B': config["target_ascii_height"] = config.get("target_ascii_height", 7) + 1 # Down
            elif key == '+': config["font_size"] = config.get("font_size", 40) + adjustment_step
            elif key == '-': config["font_size"] = max(8, config.get("font_size", 40) - adjustment_step)

    full_clear_and_reset_cursor()
    print(f"Exited {clock_identifier} clock configuration mode.")


def input_thread_fn(input_queue, stop_event):
    if os.name == 'nt': # Windows specific logic
        import msvcrt
        while not stop_event.is_set():
            if msvcrt.kbhit(): # Check if a key is pressed
                try:
                    key_bytes = msvcrt.getch()
                    # Handle special keys like arrows if necessary, they might be multi-byte
                    # For now, decode simply.
                    key = key_bytes.decode(errors='ignore')
                    if key: # Ensure key is not empty or None before putting
                        input_queue.put(key)
                except Exception:
                    pass # Ignore decode errors or other issues with getch
            time.sleep(0.01) # Small sleep to prevent busy-waiting at 100% CPU
    else: # Unix-like specific logic (uses the getch_timeout defined above for Unix)
        while not stop_event.is_set():
            key = getch_timeout(0.01) # Poll frequently using the Unix getch_timeout
            if key:
                input_queue.put(key)

# Path to key mappings JSON
KEY_MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), "time_sync", "key_mappings.json")

# Or for more robustness, add debugging:
def load_key_mappings(path: str = KEY_MAPPINGS_PATH) -> dict:
    try:
        with open(path, 'r') as f:
            mappings = json.load(f)
            print(f"Successfully loaded key mappings from {path}")
            return mappings
    except FileNotFoundError:
        print(f"Key mappings file not found at {path}")
        # Try alternate locations
        alt_paths = [
            os.path.join(os.path.dirname(__file__), "key_mappings.json"),
            os.path.join(os.path.dirname(__file__), "time_sync", "key_mappings.json"),
            os.path.join(os.path.dirname(__file__), "..", "time_sync", "key_mappings.json")
        ]
        for alt_path in alt_paths:
            try:
                with open(alt_path, 'r') as f:
                    mappings = json.load(f)
                    print(f"Found key mappings at alternate path: {alt_path}")
                    return mappings
            except FileNotFoundError:
                continue
        print("Could not find key_mappings.json in any expected location")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing key mappings JSON from {path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading key mappings from {path}: {e}")
        return {}
def main() -> None:
    """Run the clock demo until interrupted."""
    global PIXEL_BUFFER_SCALE, TEXT_FIELD_SCALE # Allow modification by keys

    parser = argparse.ArgumentParser(description="Live clock demo with various displays.")
    parser.add_argument(
        "--configure", 
        choices=["slow_analog", "fast_analog", "slow_digital", "fast_digital"], 
        help="Enter interactive configuration mode for a specific clock instance."
    )
    parser.add_argument(
        "--backdrops", nargs='+', type=str, 
        help="Space-separated list of backdrop image paths to cycle through. The first is default."
    )
    parser.add_argument(
        "--initial-pixel-buffer-scale", type=float, default=1.0, help="Initial scale factor for the internal render buffer resolution."
    )
    parser.add_argument(
        "--initial-text-field-scale", type=float, default=1.0, help="Initial scale factor for text field output size (characters)."
    )
    parser.add_argument(
        "--analog-backdrop", type=str, help="Path to an image file for the analog clock backdrop."
    )
    parser.add_argument(
        "--digital-backdrop", type=str, help="Path to an image file for digital clock backdrops."
    )
    parser.add_argument(
        "--analog-clock-rect", type=parse_rect, help="Rectangle 'x,y,width,height' for analog clock drawing on its canvas."
    )
    parser.add_argument(
        "--refresh-rate", type=float, default=0.1, help="Refresh rate in seconds (e.g., 0.1 for 10 FPS)."
    )
    parser.add_argument(
        "--no-analog", action="store_false", dest="show_analog", help="Do not display the analog clock."
    )
    parser.add_argument(
        "--no-digital-system", action="store_false", dest="show_digital_system", help="Do not display the digital system clock."
    )
    parser.add_argument(
        "--no-digital-internet", action="store_false", dest="show_digital_internet", help="Do not display the digital internet clock."
    )
    parser.add_argument(
        "--no-stopwatch", action="store_false", dest="show_stopwatch", help="Do not display the stopwatch."
    )
    parser.add_argument(
        "--no-offset", action="store_false", dest="show_offset", help="Do not display the time offset."
    )
    parser.add_argument(
        "--theme", choices=["cyberpunk", "matrix", "retro"], 
        default="cyberpunk", help="Color theme preset"
    )
    parser.add_argument(
        "--effects",
        choices=[
            "neon", "crisp", "dramatic", "ethereal", "crystalline", "ghostly",
            "vivid", "soft_glow", "shadowed", "vintage_bright", "dreamlike",
            "sketch",
        ],
        default="neon",
        help="Visual effects preset"
    )
    parser.add_argument(
        "--post-processing",
        choices=[
            "high_contrast", "soft", "monochrome", "vintage", "neon_glow",
            "matrix_code", "cyberdream", "night_vision", "sepia_tone",
            "cold_wave", "hot_warm", "comic_book", "vibrant", "low_light",
        ],
        default="high_contrast",
        help="Post-processing preset"
    )
    parser.add_argument(
        "--ascii-style", choices=["block", "detailed", "minimal", "dots", "shapes"],
        default="detailed", help="Initial ASCII character style" # Changed default to detailed
    )
    parser.add_argument(
        "--slow-analog-units", type=str, default="default_analog", help="Time unit set for the slow analog clock."
    )
    parser.add_argument(
        "--fast-analog-units", type=str, default="analog_fast_sms", help="Time unit set for the fast analog clock."
    )
    parser.add_argument(
        "--slow-digital-units", type=str, default="default_digital_hms", help="Time unit set for the slow digital clock."
    )
    parser.add_argument(
        "--fast-digital-units", type=str, default="digital_fast_sms", help="Time unit set for the fast digital clock."
    )
    parser.add_argument("--slow-digital-format", type=str, default="default_hms", help="Format key for slow digital clock.")
    parser.add_argument("--fast-digital-format", type=str, default="fast_s_ms", help="Format key for fast digital clock.")
    parser.set_defaults(
        show_analog=True,
        show_digital_system=True,
        show_digital_internet=True,
        show_stopwatch=True,
        show_offset=True,
    )
    args = parser.parse_args()

    key_mappings = load_key_mappings()

    PIXEL_BUFFER_SCALE = args.initial_pixel_buffer_scale
    TEXT_FIELD_SCALE = args.initial_text_field_scale
    
    # Base dimensions for 1.0 scale
    BASE_FB_ROWS = 35 # Reduced to make space for text overlays
    BASE_FB_COLS = 120

    init_colorama_for_windows()

    # Load configurations from JSON files
    # These will serve as templates; specific instances can override
    slow_analog_config = load_config_from_json("analog_config.json", "slow_analog")
    fast_analog_config = load_config_from_json("analog_config.json", "fast_analog") # Start with base analog
    slow_digital_config = load_config_from_json("digital_config.json", "slow_digital")
    fast_digital_config = load_config_from_json("digital_config.json", "fast_digital") # Start with base digital

    # Consolidate configs for easier access later if needed
    clock_configs = {
        "slow_analog": slow_analog_config, "fast_analog": fast_analog_config,
        "slow_digital": slow_digital_config, "fast_digital": fast_digital_config
    }

    if args.configure:
        # ThemeManager needs to be initialized for backdrop cycling in config mode
        temp_theme_manager = ThemeManager(presets_path=os.path.join(os.path.dirname(__file__), "time_sync", "presets", "default_themes.json"))
        if args.backdrops: temp_theme_manager.current_theme.current_backdrop_path = args.backdrops[0]
        
        interactive_configure_mode(args.configure, clock_configs[args.configure], temp_theme_manager)
        return # Exit after configuration mode


    init_colorama_for_windows()
    sync_offset()
    start = time.perf_counter()

    # Initialize ThemeManager and set current theme from args
    presets_file_path = os.path.join(os.path.dirname(__file__), "time_sync", "presets", "default_themes.json")
    # Simplified path finding, assuming ThemeManager's default is usually correct
    # or the user runs from a location where `time_sync/time_sync/presets` is valid.

    theme_manager = ThemeManager(presets_path=presets_file_path)
    if args.backdrops:
        theme_manager.current_theme.current_backdrop_path = args.backdrops[0]

    render_backend = RenderingBackend(theme_manager)

    theme_manager.set_palette(args.theme)
    theme_manager.set_effects(args.effects)
    theme_manager.set_post_processing(args.post_processing)
    # Set initial ASCII style from args, then ThemeManager can cycle it
    theme_manager.current_theme.ascii_style = args.ascii_style

    # Store current unit set and format keys, to be cycled by new controls
    current_slow_analog_units_key = args.slow_analog_units
    current_fast_analog_units_key = args.fast_analog_units
    current_slow_digital_units_key = args.slow_digital_units
    current_fast_digital_units_key = args.fast_digital_units
    current_slow_digital_format_key = args.slow_digital_format
    current_fast_digital_format_key = args.fast_digital_format
    
    # Helper to get a list of TimeUnit objects for a given set name
    def get_time_units_for_set(set_name: str) -> list:
        all_defined_units = theme_manager.get_time_unit_definitions()
        unit_names_in_set = theme_manager.presets.get("time_unit_sets", {}).get(set_name, [])
        units = [all_defined_units[name] for name in unit_names_in_set if name in all_defined_units]
        if len(units) != len(unit_names_in_set):
            print(f"Warning: Some units in set '{set_name}' were not found in definitions.")
        return units[:5] # Max 5 units

    # Helper to get a format string for a given key
    def get_format_string_for_key(key_name: str) -> str:
        return theme_manager.presets.get("digital_format_strings", {}).get(key_name, "{value}")

    # Initial unit sets and format strings will be fetched inside compose_full_frame

    # Initialize clock parameters by merging defaults with command-line args
    # These are base parameters, specific clock instances might have their own loaded configs
    base_analog_params = load_config_from_json("analog_config.json") 
    if args.analog_clock_rect:
        base_analog_params["clock_drawing_rect_on_canvas"] = args.analog_clock_rect
    # Backdrop for analog_params is now handled by theme_manager.current_theme.current_backdrop_path

    base_digital_params = load_config_from_json("digital_config.json")
    # Backdrop for digital_params is now handled by theme_manager.current_theme.current_backdrop_path

    # --- FRAMEBUFFER COMPOSITION ---
    def get_scaled_fb_dims():
        return int(BASE_FB_ROWS * PIXEL_BUFFER_SCALE), int(BASE_FB_COLS * PIXEL_BUFFER_SCALE)

    fb_rows, fb_cols = get_scaled_fb_dims()
    framebuffer = PixelFrameBuffer((fb_rows, fb_cols), diff_threshold=20) # Initialized with scaled dims
    stop_event = threading.Event()

    # Display state flags
    display_state = {
        "analog_clocks": args.show_analog,
        "slow_digital_clock": args.show_digital_system,
        "fast_digital_clock": args.show_digital_internet,
        "stopwatch": args.show_stopwatch,
        "offset_info": args.show_offset,
        "legend": True # Show legend by default
    }

    def compose_full_frame(system_time_obj, internet_time_obj, stopwatch_td_obj, offset_val):
        current_fb_rows, current_fb_cols = get_scaled_fb_dims()
        # Initialize buffer with black pixels
        buf = np.full((current_fb_rows, current_fb_cols, 3), [0,0,0], dtype=np.uint8) # This buffer is ONLY for the clocks now
        row = 0
        available_width = current_fb_cols # Pixel width
        spacer_color = [5,5,10] # Dark spacer
        spacer_height = 1

        # Determine time objects for slow/fast clocks
        slow_time_obj = internet_time_obj # Or system_time_obj
        fast_time_obj = stopwatch_td_obj
        
        current_backdrop_path = theme_manager.current_theme.current_backdrop_path
        # Fetch current unit sets and formats based on keys
        slow_analog_units = get_time_units_for_set(current_slow_analog_units_key)
        fast_analog_units = get_time_units_for_set(current_fast_analog_units_key)
        slow_digital_units = get_time_units_for_set(current_slow_digital_units_key)
        fast_digital_units = get_time_units_for_set(current_fast_digital_units_key)
        slow_digital_format = get_format_string_for_key(current_slow_digital_format_key)
        fast_digital_format = get_format_string_for_key(current_fast_digital_format_key)

        if display_state["analog_clocks"]:
            # Scale analog row pixel height based on TEXT_FIELD_SCALE (char height) and PIXEL_BUFFER_SCALE (pixel density)
            # Base character height for analog might be around 15-20 chars
            base_analog_char_h = 15 
            analog_row_pixel_height = int(base_analog_char_h * TEXT_FIELD_SCALE * (PIXEL_BUFFER_SCALE * 1.5)) # Heuristic for char cell to pixel
            analog_width_each = available_width // 2
            
            # Slow Analog Clock (Left)
            analog_slow_params = clock_configs["slow_analog"].copy()
            analog_slow_params["target_ascii_diameter"] = int(base_analog_char_h * TEXT_FIELD_SCALE)
            analog_slow_params["canvas_size_px"] = int(analog_slow_params["target_ascii_diameter"] * 7 * PIXEL_BUFFER_SCALE) # Scale canvas
            analog_slow_params["backdrop_image_path"] = current_backdrop_path

            slow_analog_arr = print_analog_clock(
                slow_time_obj,
                active_units_override=slow_analog_units,
                theme_manager=theme_manager,
                render_backend=render_backend,
                as_pixel=True,
                **analog_slow_params
            )
            if slow_analog_arr is not None:
                h, w, _ = slow_analog_arr.shape
                place_h = min(h, analog_row_pixel_height)
                place_w = min(w, analog_width_each)
                y_offset = (analog_row_pixel_height - place_h) // 2 
                x_offset = (analog_width_each - place_w) // 2 # Center horizontally
                if row + y_offset + place_h <= current_fb_rows:
                    buf[row + y_offset : row + y_offset + place_h, 
                        x_offset : x_offset + place_w] = slow_analog_arr[:place_h, :place_w]

            # Fast Analog Clock (Right)
            analog_fast_params = clock_configs["fast_analog"].copy()
            analog_fast_params["target_ascii_diameter"] = int(base_analog_char_h * TEXT_FIELD_SCALE)
            analog_fast_params["canvas_size_px"] = int(analog_fast_params["target_ascii_diameter"] * 7 * PIXEL_BUFFER_SCALE)
            analog_fast_params["backdrop_image_path"] = current_backdrop_path

            fast_analog_arr = print_analog_clock(
                fast_time_obj, # Stopwatch time
                active_units_override=fast_analog_units,
                theme_manager=theme_manager,
                render_backend=render_backend,
                as_pixel=True,
                **analog_fast_params
            )
            if fast_analog_arr is not None:
                h, w, _ = fast_analog_arr.shape
                place_h = min(h, analog_row_pixel_height)
                place_w = min(w, analog_width_each)
                y_offset = (analog_row_pixel_height - place_h) // 2
                x_offset_fast = analog_width_each + (analog_width_each - place_w) // 2 
                if row + y_offset + place_h <= current_fb_rows:
                     buf[row + y_offset : row + y_offset + place_h, 
                         x_offset_fast : x_offset_fast + place_w] = fast_analog_arr[:place_h, :place_w]
            
            row += analog_row_pixel_height
            if row < current_fb_rows: buf[row, :available_width] = spacer_color; row += spacer_height

        if display_state["slow_digital_clock"]:
            digital_slow_params = clock_configs["slow_digital"].copy() # Use specific config
            base_digital_char_h = digital_slow_params.get("target_ascii_height", 7)
            digital_row_pixel_height = int(base_digital_char_h * TEXT_FIELD_SCALE * (PIXEL_BUFFER_SCALE * 1.5)) # Heuristic
            digital_slow_params["target_ascii_height"] = int(base_digital_char_h * TEXT_FIELD_SCALE) # Text field scale
            digital_slow_params["target_ascii_width"] = int(digital_slow_params.get("target_ascii_width", 60) * TEXT_FIELD_SCALE)
            digital_slow_params["font_size"] = int(digital_slow_params.get("font_size", 40) * PIXEL_BUFFER_SCALE) # Font size scales with buffer
            digital_slow_params["backdrop_image_path"] = current_backdrop_path
            
            slow_digital_arr = print_digital_clock(
                slow_time_obj,
                active_units=slow_digital_units,
                display_format_template=slow_digital_format,
                theme_manager=theme_manager,
                render_backend=render_backend,
                as_pixel=True,
                **digital_slow_params
            )
            if slow_digital_arr is not None:
                h, w, _ = slow_digital_arr.shape
                place_h = min(h, digital_row_pixel_height)
                place_w = min(w, available_width)
                if row + place_h <= current_fb_rows:
                    buf[row:row+place_h, :place_w] = slow_digital_arr[:place_h, :place_w]
                row += place_h
            if row < current_fb_rows: buf[row, :available_width] = spacer_color; row += spacer_height

        if display_state["fast_digital_clock"]:
            digital_fast_params = clock_configs["fast_digital"].copy() # Use specific config
            base_digital_char_h = digital_fast_params.get("target_ascii_height", 7)
            digital_row_pixel_height = int(base_digital_char_h * TEXT_FIELD_SCALE * (PIXEL_BUFFER_SCALE * 1.5))
            digital_fast_params["target_ascii_height"] = int(base_digital_char_h * TEXT_FIELD_SCALE)
            digital_fast_params["target_ascii_width"] = int(digital_fast_params.get("target_ascii_width", 60) * TEXT_FIELD_SCALE)
            digital_fast_params["font_size"] = int(digital_fast_params.get("font_size", 40) * PIXEL_BUFFER_SCALE)
            digital_fast_params["backdrop_image_path"] = current_backdrop_path

            fast_digital_arr = print_digital_clock(
                fast_time_obj, # Stopwatch time
                active_units=fast_digital_units,
                display_format_template=fast_digital_format,
                theme_manager=theme_manager,
                render_backend=render_backend,
                as_pixel=True,
                **digital_fast_params
            )
            if fast_digital_arr is not None:
                h, w, _ = fast_digital_arr.shape
                place_h = min(h, digital_row_pixel_height)
                place_w = min(w, available_width)
                if row + place_h <= current_fb_rows:
                    buf[row:row+place_h, :place_w] = fast_digital_arr[:place_h, :place_w]
                row += place_h
            if row < current_fb_rows: buf[row, :available_width] = spacer_color; row += spacer_height

        # Stopwatch, Offset, and Legend are now drawn as text overlays AFTER the pixel buffer
        # So they are NOT rendered into 'buf' here.

        return buf

    def render_fn(framebuffer_ref): # Pass framebuffer to allow reinitialization
        elapsed = time.perf_counter() - start
        offset = get_offset()
        system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
        internet = system + _dt.timedelta(seconds=offset)

        h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
        m, rem = divmod(rem, 60 * 1000)
        s, ms = divmod(rem, 1000)
        stopwatch_td = _dt.timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

        # Check if framebuffer dimensions need to change due to scale factors
        new_fb_rows, new_fb_cols = get_scaled_fb_dims()
        if framebuffer_ref.buffer_shape[0] != new_fb_rows or framebuffer_ref.buffer_shape[1] != new_fb_cols:
            framebuffer_ref._resize((new_fb_rows, new_fb_cols)) # Use internal resize

        frame = compose_full_frame(system, internet, stopwatch_td, offset)
        img = Image.fromarray(frame, mode="RGB")
        img = render_backend.process(img)
        if img.mode != "RGB":
            # RenderingBackend may produce an RGBA image when effects use
            # transparency. Convert back to RGB so PixelFrameBuffer receives
            # the expected three-channel data.
            img = img.convert("RGB")
        return np.array(img)

    render_thread = threading.Thread(
        target=render_loop,
        args=(framebuffer, lambda: render_fn(framebuffer), 10, stop_event), # Pass framebuffer to render_fn
        daemon=True,
    )
    render_thread.start()

    # Start the input thread AFTER the render thread
    full_clear_and_reset_cursor()
    input_buffer = ""
    input_queue = queue.Queue()
    input_thread = threading.Thread(target=input_thread_fn, args=(input_queue, stop_event), daemon=True)
    input_thread.start()

    # Calculate the starting row for text overlays (1-indexed for terminal)
    # This is the row immediately after the pixel buffer ends.
    text_overlay_start_row = get_scaled_fb_dims()[0] + 1
    current_text_overlay_row = text_overlay_start_row

    # For periodic keyframe redraw
    KEYFRAME_INTERVAL_SECONDS = 30  # Redraw everything every 30 seconds
    last_keyframe_time = time.perf_counter()

    try:
        while True:
            # --- Drawing Phase ---
            # The render_thread updates the framebuffer's render buffer.
            # The main loop gets the diff from the framebuffer and draws it.
            
            # Get changed pixels from the framebuffer
            diff_pixels = framebuffer.get_diff_and_promote()
            changed = [
                (y, x, np.array([[color]], dtype=np.uint8))
                for y, x, color in diff_pixels
            ]
            # Draw the changed pixels (the clocks)
            # If diff_pixels is empty but a redraw was forced, 'changed' will also be empty.
            # The draw_diff will still be called, but it won't do much if 'changed' is empty.
            # The key is that get_diff_and_promote() would have returned all pixels if forced.
            # However, the current draw_diff takes 'changed_subunits' which are already diffed.
            # The PixelFrameBuffer.get_diff_and_promote() now returns all pixels if forced.
            # So, `diff_pixels` will contain all pixels, and thus `changed` will too.
            # We need to pass the correct char_cell_pixel_height/width to draw_diff
            # These should correspond to the pixel dimensions of a single character cell
            # as used by the rendering functions (print_analog_clock, print_digital_clock).
            # This is complex because different elements might use different effective pixel sizes per char.
            # For simplicity, let's assume a base 1:1 mapping for now, or derive from scale.
            # A better approach might be to have render_fn return the rendered pixel array AND the effective
            # pixel dimensions per character cell for the main clock area.
            # For now, let's assume 1:1 pixel to char cell mapping for the main buffer drawing.
            draw_diff(changed, 
                      char_cell_pixel_height=1, # Assuming 1 pixel = 1 char cell for buffer drawing
                      char_cell_pixel_width=1,  # Assuming 1 pixel = 1 char cell for buffer drawing
                      active_ascii_ramp=theme_manager.get_current_ascii_ramp())

            # --- Text Overlay Drawing Phase ---
            # Draw text elements directly as overlays below the pixel buffer
            current_text_overlay_row = text_overlay_start_row
            available_cols = get_scaled_fb_dims()[1] # Use scaled width for text overlays

            # Clear the text overlay area before drawing new text
            # This prevents old text from lingering if new text is shorter
            text_overlay_height = BASE_FB_ROWS - BASE_FB_ROWS # Calculate total height needed for overlays
            # This calculation is tricky as text height depends on content and scale.
            # A simpler approach is to clear a fixed number of lines at the bottom.
            # Let's clear enough lines for Stopwatch (1), Offset (1), Legend (approx 10 lines).
            # Total ~12 lines. BASE_FB_ROWS is 35, so 45 total. Need ~10 lines below buffer.
            # Let's clear from text_overlay_start_row to the end of the terminal.
            # This requires knowing terminal height, which is hard.
            # Alternative: Clear a fixed number of lines based on max possible overlays.
            # Stopwatch (1) + Offset (1) + Legend (approx 10) = ~12 lines.
            # Let's clear 15 lines from text_overlay_start_row.
            clear_lines_count = 15
            for r in range(text_overlay_start_row, text_overlay_start_row + clear_lines_count):
                 draw_text_overlay(r, 1, " " * available_cols, Style.RESET_ALL) # Clear line with spaces

            # Reset row counter for drawing overlays
            current_text_overlay_row = text_overlay_start_row

            # Get current time objects for text overlays
            elapsed = time.perf_counter() - start
            offset = get_offset()
            system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
            internet = system + _dt.timedelta(seconds=offset)
            h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
            m, rem = divmod(rem, 60 * 1000)
            s, ms = divmod(rem, 1000)
            stopwatch_td = _dt.timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

            if display_state["stopwatch"]:
                stopwatch_text = f"Stopwatch: {stopwatch_td.total_seconds():09.3f}"
                draw_text_overlay(current_text_overlay_row, 1, stopwatch_text, Style.BRIGHT + Fore.YELLOW) # Example color
                current_text_overlay_row += 1 # Move to next line

            if display_state["offset_info"]:
                offset_text = f"Offset: {_dt.timedelta(seconds=offset)}"
                draw_text_overlay(current_text_overlay_row, 1, offset_text, Style.BRIGHT + Fore.CYAN) # Example color
                current_text_overlay_row += 1 # Move to next line

            if display_state["legend"]:
                legend_start_col = 1
                legend_col_width = available_cols // 2 # Split legend into two columns
                
                legend_parts = []
                for action, mapping in key_mappings.items():
                    keys_str = "/".join(mapping["keys"])
                    legend_parts.append(f"{keys_str}: {mapping['description']}")
                
                num_legend_items = len(legend_parts)
                col1_items = legend_parts[:(num_legend_items + 1) // 2]
                col2_items = legend_parts[(num_legend_items + 1) // 2:]

                for i in range(max(len(col1_items), len(col2_items))):
                    if current_text_overlay_row > get_scaled_fb_dims()[0] + clear_lines_count: # Prevent drawing outside cleared area
                         break
                    line_text_parts = []
                    # Use fixed ljust for text overlays, not scaled
                    ljust_val = 30 
                    if i < len(col1_items): line_text_parts.append(col1_items[i].ljust(ljust_val)) 
                    if i < len(col2_items): line_text_parts.append(col2_items[i])
                    legend_line_text = "  ".join(line_text_parts)
                    draw_text_overlay(current_text_overlay_row, legend_start_col, legend_line_text, Style.RESET_ALL) # Default color
                    current_text_overlay_row += 1 # Move to next line

            # Drain the input queue
            while not input_queue.empty():
                key = input_queue.get()
                input_buffer += key

            # Process characters from the raw input_buffer
            # We'll build a new buffer with unprocessed characters
            remaining_buffer = ""
            
            action_processed_this_loop = False
            if not stop_event.is_set() and input_buffer: # Check raw input_buffer
                for char_key_raw in input_buffer: # Iterate through raw characters from buffer
                    action_found = False
                    for action, mapping in key_mappings.items():
                        if char_key_raw in mapping["keys"]: # Compare raw char_key_raw
                            # --- Execute action ---
                            if action == "quit": stop_event.set()
                            elif action == "toggle_legend": display_state["legend"] = not display_state["legend"]
                            elif action == "cycle_ascii_style_forward": theme_manager.cycle_ascii_style() 
                            elif action == "cycle_ascii_style_backward": theme_manager.cycle_ascii_style(-1)
                            elif action == "cycle_palette_forward": theme_manager.cycle_palette(1)
                            elif action == "cycle_palette_backward": theme_manager.cycle_palette(-1)
                            elif action == "cycle_effects_forward": theme_manager.cycle_effects_preset(1)
                            elif action == "cycle_effects_backward": theme_manager.cycle_effects_preset(-1)
                            elif action == "cycle_post_processing_forward": theme_manager.cycle_post_processing_preset(1)
                            elif action == "cycle_post_processing_backward": theme_manager.cycle_post_processing_preset(-1)
                            elif action == "toggle_clock_inversion": theme_manager.toggle_clock_inversion() 
                            elif action == "toggle_backdrop_inversion": theme_manager.toggle_backdrop_inversion() 
                            
                            elif action == "cycle_slow_analog_units_forward": current_slow_analog_units_key = theme_manager.cycle_time_unit_set(current_slow_analog_units_key, 1)
                            elif action == "cycle_slow_analog_units_backward": current_slow_analog_units_key = theme_manager.cycle_time_unit_set(current_slow_analog_units_key, -1)
                            elif action == "cycle_fast_analog_units_forward": current_fast_analog_units_key = theme_manager.cycle_time_unit_set(current_fast_analog_units_key, 1)
                            elif action == "cycle_fast_analog_units_backward": current_fast_analog_units_key = theme_manager.cycle_time_unit_set(current_fast_analog_units_key, -1)
                            
                            elif action == "cycle_slow_digital_units_forward": current_slow_digital_units_key = theme_manager.cycle_time_unit_set(current_slow_digital_units_key, 1) 
                            elif action == "cycle_slow_digital_units_backward": current_slow_digital_units_key = theme_manager.cycle_time_unit_set(current_slow_digital_units_key, -1) 
                            elif action == "cycle_fast_digital_units_forward": current_fast_digital_units_key = theme_manager.cycle_time_unit_set(current_fast_digital_units_key, 1) 
                            elif action == "cycle_fast_digital_units_backward": current_fast_digital_units_key = theme_manager.cycle_time_unit_set(current_fast_digital_units_key, -1) 

                            elif action == "cycle_slow_digital_format_forward": current_slow_digital_format_key = theme_manager.cycle_digital_format_key(current_slow_digital_format_key, 1)
                            elif action == "cycle_slow_digital_format_backward": current_slow_digital_format_key = theme_manager.cycle_digital_format_key(current_slow_digital_format_key, -1)
                            elif action == "cycle_fast_digital_format_forward": current_fast_digital_format_key = theme_manager.cycle_digital_format_key(current_fast_digital_format_key, 1)
                            elif action == "cycle_fast_digital_format_backward": current_fast_digital_format_key = theme_manager.cycle_digital_format_key(current_fast_digital_format_key, -1)

                            elif action == "toggle_analog_clocks": display_state["analog_clocks"] = not display_state["analog_clocks"] 
                            elif action == "toggle_slow_digital_clock": display_state["slow_digital_clock"] = not display_state["slow_digital_clock"] 
                            elif action == "toggle_fast_digital_clock": display_state["fast_digital_clock"] = not display_state["fast_digital_clock"] 
                            elif action == "toggle_stopwatch": display_state["stopwatch"] = not display_state["stopwatch"] 
                            elif action == "toggle_offset_info": display_state["offset_info"] = not display_state["offset_info"] 

                            elif action == "increase_buffer_scale": PIXEL_BUFFER_SCALE = min(3.0, PIXEL_BUFFER_SCALE + 0.1)
                            elif action == "decrease_buffer_scale": PIXEL_BUFFER_SCALE = max(0.2, PIXEL_BUFFER_SCALE - 0.1)
                            elif action == "increase_text_field_scale": TEXT_FIELD_SCALE = min(3.0, TEXT_FIELD_SCALE + 0.1)
                            elif action == "decrease_text_field_scale": TEXT_FIELD_SCALE = max(0.2, TEXT_FIELD_SCALE - 0.1)
                            elif action == "cycle_backdrop_forward": theme_manager.cycle_backdrop(args.backdrops or [], 1) 
                            elif action == "cycle_backdrop_backward": theme_manager.cycle_backdrop(args.backdrops or [], -1) 
                            elif action == "enter_config_mode":
                                # Determine which clock to configure (e.g., based on active ones or a sub-menu)
                                # For simplicity, let's say it configures "slow_analog" if active, else prompts
                                active_clocks_for_config = []
                                if display_state["analog_clocks"]:
                                    active_clocks_for_config.append("slow_analog")
                                    active_clocks_for_config.append("fast_analog")
                                if display_state["slow_digital_clock"]: active_clocks_for_config.append("slow_digital")
                                if display_state["fast_digital_clock"]: active_clocks_for_config.append("fast_digital")
                                
                                if active_clocks_for_config:
                                    # Simple: configure the first one in the list, or implement a selection mechanism
                                    clock_to_config = active_clocks_for_config[0]
                                    # Pause render thread during config mode might be good
                                    stop_event.set() # Signal render thread to pause/stop
                                    render_thread.join(timeout=0.5)
                                    full_clear_and_reset_cursor()
                                    interactive_configure_mode(clock_to_config, clock_configs[clock_to_config], theme_manager)
                                    # Resume render thread
                                    stop_event.clear()
                                    render_thread = threading.Thread(
                                        target=render_loop,
                                        args=(framebuffer, lambda: render_fn(framebuffer), 10, stop_event),
                                        daemon=True,
                                    )
                                    render_thread.start()
                                    full_clear_and_reset_cursor() # Clear config screen
                                else:
                                    print("No active clock to configure.") # Or a small message on screen

                            action_found = True
                            action_processed_this_loop = True
                            break # Found action for this char_key
                    
                    if action_found:
                        if stop_event.is_set(): break # If quit was pressed, exit cmd processing
                        # Character was processed, do not add to remaining_buffer
                    else:
                        # If the character didn't match any action, keep it
                        remaining_buffer += char_key_raw
                
                input_buffer = remaining_buffer # Update buffer with unprocessed characters

            # Trigger keyframe if an action was processed
            if action_processed_this_loop:
                framebuffer.force_full_redraw_next_frame()
                # Also force redraw of text overlays? Not needed as they are redrawn every frame.
            # Periodic keyframe trigger
            current_loop_time = time.perf_counter()
            if current_loop_time - last_keyframe_time >= KEYFRAME_INTERVAL_SECONDS:
                framebuffer.force_full_redraw_next_frame()
                last_keyframe_time = current_loop_time

            if stop_event.is_set(): # Check if we should break out of the main loop
                break
            time.sleep(args.refresh_rate)

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        input_thread.join(timeout=2.0)
        render_thread.join(timeout=2.0)
        full_clear_and_reset_cursor()
        print("Demo stopped.")
        final_offset = get_offset()
        final_system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
        final_internet = final_system + _dt.timedelta(seconds=final_offset)
        print(f"Final system time: {final_system.strftime('%H:%M:%S')}")
        print(f"Final internet time: {final_internet.strftime('%H:%M:%S')}")
        print(f"Offset: {_dt.timedelta(seconds=final_offset)}")


if __name__ == "__main__":
    main()
