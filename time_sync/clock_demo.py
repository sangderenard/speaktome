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
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from time_sync import (
    get_offset, sync_offset,
    print_analog_clock, print_digital_clock,
    init_colorama_for_windows, reset_cursor_to_top, full_clear_and_reset_cursor
)
from time_sync.theme_manager import ThemeManager, ClockTheme # Ensure this import works
from time_sync.frame_buffer import PixelFrameBuffer
from time_sync.render_thread import render_loop
from time_sync.draw import draw_diff
from time_sync.ascii_digits import (
    compose_ascii_digits,
    ASCII_RAMP_BLOCK,
)
from PIL import Image

# Platform-specific input handling (adapted from AGENTS/tools/dev_group_menu.py)
if os.name == 'nt':  # Windows
    import msvcrt
    import threading
    def getch_timeout(timeout_seconds: float) -> str | None:
        """Get a single character with timeout on Windows."""
        result_char = [None] # Use a list to pass by reference
        def _input_thread():
            try:
                result_char[0] = msvcrt.getch().decode(errors='ignore')
            except Exception:
                result_char[0] = None # Or some other indicator
        
        thread = threading.Thread(target=_input_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        return result_char[0]

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


def load_config_from_json(config_path: str) -> dict:
    """Loads configuration from a JSON file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {config_path}. Using defaults.")
        except Exception as e:
            print(f"Warning: Error loading {config_path}: {e}. Using defaults.")
    return {}

def save_config_to_json(config_path: str, config_data: dict) -> None:
    """Saves configuration to a JSON file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")


def interactive_configure_mode(clock_type: str, initial_config: dict, backdrop_path: str | None):
    """Interactive mode for configuring clock appearance."""
    config = initial_config.copy()
    config_path = f"{clock_type}_config.json" # analog_config.json or digital_config.json

    print(f"Entering {clock_type} clock configuration mode...")
    print(f"Using backdrop: {backdrop_path if backdrop_path else 'None'}")
    print("Press 's' to save, 'q' to quit without saving.")
    time.sleep(1)

    adjustment_step = 2
    
    while True:
        full_clear_and_reset_cursor()
        current_time = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc) # Or use time_sync.now()

        print(f"--- {clock_type.upper()} CLOCK CONFIGURATION ---")
        if clock_type == "analog":
            # Ensure rect is a tuple if it exists
            rect = config.get("clock_drawing_rect_on_canvas")
            if isinstance(rect, list) and len(rect) == 4:
                config["clock_drawing_rect_on_canvas"] = tuple(rect)
            
            print_analog_clock(current_time, backdrop_image_path=backdrop_path, **config)
            print("\n--- Current Analog Config ---")
            rect = config.get("clock_drawing_rect_on_canvas", "Not set")
            print(f"Drawing Rect (x,y,w,h): {rect} (Use Arrow keys to move)")
            print(f"Resize Rect: Width (J/L), Height (I/K)")
            print(f"ASCII Diameter: {config.get('target_ascii_diameter', 'Default')} (+/-)")
            print(f"Canvas Size PX: {config.get('canvas_size_px', 'Default')} (c/C)")
            help_text = "[Arrows]:Move [J/L]:Width [I/K]:Height [+/-]:ASCII Dia. [c/C]:Canvas Px [s]:Save [q]:Quit"

        elif clock_type == "digital":
            print_digital_clock(current_time, backdrop_image_path=backdrop_path, **config)
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
            save_config_to_json(config_path, config)
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
    print(f"Exited {clock_type} clock configuration mode.")


def render_simple_text_to_pixel_array(
    text: str,
    target_pixel_width: int,
    target_pixel_height: int,
    font_size: int = 12, # Smaller default for simple text lines
    text_color: tuple[int,int,int,int] = (200,200,200,255), # Light grey
    bg_color: tuple[int,int,int,int] = (0,0,0,0), # Transparent background for PIL image
    theme_manager: ThemeManager | None = None,
    ) -> np.ndarray:
    """Renders a single line of text to a pixel array."""
    # Use compose_ascii_digits to leverage its text-to-image capabilities
    pixel_array = compose_ascii_digits(
        text,
        font_size=font_size,
        text_color_on_image=text_color,
        # No shadow, no outline for simple text by default
        shadow_color_on_image=(0,0,0,0), outline_thickness=0,
        as_pixel_array=True,
        target_pixel_width=target_pixel_width,
        target_pixel_height=target_pixel_height,
        ascii_ramp=theme_manager.get_current_ascii_ramp() if theme_manager else ASCII_RAMP_BLOCK,
        theme_manager=theme_manager,
    )
    return pixel_array
def main() -> None:
    """Run the clock demo until interrupted."""
    parser = argparse.ArgumentParser(description="Live clock demo with various displays.")
    parser.add_argument(
        "--configure", choices=["analog", "digital"], help="Enter interactive configuration mode for the specified clock type."
    )
    parser.add_argument(
        "--backdrop", type=str, help="Default backdrop image for all clocks if specific ones are not set."
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
        "--effects", choices=["neon", "crisp", "dramatic"],
        default="neon", help="Visual effects preset"
    )
    parser.add_argument(
        "--post-processing", choices=["high_contrast", "soft", "monochrome"],
        default="high_contrast", help="Post-processing preset"
    )
    parser.add_argument(
        "--ascii-style", choices=["block", "detailed", "minimal", "dots", "shapes"],
        default="detailed", help="Initial ASCII character style" # Changed default to detailed
    )
    parser.set_defaults(
        show_analog=True,
        show_digital_system=True,
        show_digital_internet=True,
        show_stopwatch=True,
        show_offset=True,
    )
    args = parser.parse_args()

    init_colorama_for_windows()

    # Load configurations from JSON files
    default_analog_config = load_config_from_json("analog_config.json")
    default_digital_config = load_config_from_json("digital_config.json")

    if args.configure:
        if args.configure == "analog":
            # Determine backdrop for analog configuration
            analog_config_backdrop = args.analog_backdrop if args.analog_backdrop is not None else args.backdrop
            interactive_configure_mode("analog", default_analog_config, analog_config_backdrop)
        elif args.configure == "digital":
            # Determine backdrop for digital configuration
            digital_config_backdrop = args.digital_backdrop if args.digital_backdrop is not None else args.backdrop
            interactive_configure_mode("digital", default_digital_config, digital_config_backdrop)
        return # Exit after configuration mode


    init_colorama_for_windows()
    sync_offset()
    start = time.perf_counter()

    # Initialize ThemeManager and set current theme from args
    # Assuming presets/default_themes.json is relative to where clock_demo.py is run,
    # or adjust path in ThemeManager.
    # For a library, ThemeManager should ideally find its presets relative to its own location.
    # For now, let's assume it's findable from the CWD or an adjusted path.
    # Path to presets, assuming clock_demo.py is in time_sync/ and presets is in time_sync/time_sync/presets
    presets_file_path = os.path.join(os.path.dirname(__file__), "time_sync", "presets", "default_themes.json")
    
    # Try to find presets relative to the script or common project structures
    # This helps if the script is run from different working directories.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path 1: <script_dir>/time_sync/presets/default_themes.json (if clock_demo.py is in time_sync root)
    path1 = os.path.join(script_dir, "time_sync", "presets", "default_themes.json")
    # Path 2: <script_dir>/../presets/default_themes.json (if clock_demo.py is in time_sync/time_sync/ and presets is in time_sync/presets)
    path2 = os.path.join(script_dir, "..", "presets", "default_themes.json")
    # Path 3: presets/default_themes.json (if run from project root and presets is in project_root/presets)
    path3 = "presets/default_themes.json"
    # Path 4: time_sync/presets/default_themes.json (if run from project root and presets is in project_root/time_sync/presets)
    path4 = "time_sync/presets/default_themes.json"

    if not os.path.exists(presets_file_path) and os.path.exists("presets/default_themes.json"): # If run from repo root
        presets_file_path = "presets/default_themes.json" 
    elif not os.path.exists(presets_file_path) and os.path.exists("time_sync/presets/default_themes.json"): # If run from one level up
        presets_file_path = "time_sync/presets/default_themes.json"

    theme_manager = ThemeManager(presets_path=presets_file_path)
    theme_manager.set_palette(args.theme)
    theme_manager.set_effects(args.effects)
    theme_manager.set_post_processing(args.post_processing)
    
    # Initialize clock parameters by merging defaults with command-line args
    analog_params = default_analog_config.copy()
    if args.analog_clock_rect:
        analog_params["clock_drawing_rect_on_canvas"] = args.analog_clock_rect
    if args.analog_backdrop or args.backdrop:
        analog_params["backdrop_image_path"] = args.analog_backdrop if args.analog_backdrop is not None else args.backdrop

    digital_params = default_digital_config.copy()
    if args.digital_backdrop or args.backdrop:
        digital_params["backdrop_image_path"] = args.digital_backdrop if args.digital_backdrop is not None else args.backdrop

    # --- FRAMEBUFFER COMPOSITION ---
    fb_rows = 40
    fb_cols = 120 # Pixel columns, can be wider than typical char cells
    # For pixel buffer, fb_rows and fb_cols define the pixel resolution.
    # Each "pixel" will be mapped to a terminal character cell by draw_diff.
    framebuffer = PixelFrameBuffer((fb_rows, fb_cols))
    stop_event = threading.Event()

    def compose_full_frame(system, internet, stopwatch, offset):
        # Initialize buffer with black pixels
        buf = np.full((fb_rows, fb_cols, 3), [0,0,0], dtype=np.uint8)
        row = 0
        available_width = fb_cols

        if args.show_analog:
            analog_arr = print_analog_clock(
                internet,
                theme_manager=theme_manager,
                as_pixel=True,
                **analog_params
            )
            h, w, _ = analog_arr.shape
            place_w = min(w, available_width)
            buf[row:row+h, :place_w] = analog_arr[:, :place_w]
            row += h
            # Add a small black spacer row if there's space
            if row < fb_rows: buf[row, :available_width] = [10,10,10]; row +=1 # Dark grey spacer

        if args.show_digital_system:
            sys_arr = print_digital_clock(
                system,
                theme_manager=theme_manager,
                as_pixel=True,
                **digital_params
            )
            h, w, _ = sys_arr.shape
            place_w = min(w, available_width)
            buf[row:row+h, :place_w] = sys_arr[:, :place_w]
            row += h
            if row < fb_rows: buf[row, :available_width] = [10,10,10]; row +=1

        if args.show_digital_internet:
            # Use compose_ascii_digits directly for pixel array
            # Ensure default_digital_config has appropriate target_ascii_width/height
            # which will be used as target_pixel_width/height
            net_params = default_digital_config.copy()
            # Example: define pixel dimensions for this specific text element
            target_h = net_params.get("target_ascii_height", 7) 
            target_w = net_params.get("target_ascii_width", 60)

            net_arr = compose_ascii_digits(
                internet.strftime("%H:%M:%S"),
                as_pixel_array=True,
                target_pixel_width=target_w,
                target_pixel_height=target_h,
                ascii_ramp=theme_manager.get_current_ascii_ramp(),
                theme_manager=theme_manager,
                **net_params
            )
            h, w, _ = net_arr.shape
            place_w = min(w, available_width)
            if row + h <= fb_rows:
                buf[row:row+h, :place_w] = net_arr[:, :place_w]
                row += h
                if row < fb_rows: buf[row, :available_width] = [10,10,10]; row +=1

        if args.show_stopwatch:
            text_h = 3 # Small height for single line text
            if row + text_h <= fb_rows:
                sw_arr = render_simple_text_to_pixel_array(
                    f"Stopwatch: {stopwatch}",
                    available_width,
                    text_h,
                    font_size=10,
                    theme_manager=theme_manager,
                )
                buf[row:row+text_h, :available_width] = sw_arr
                row += text_h
                if row < fb_rows: buf[row, :available_width] = [10,10,10]; row +=1

        if args.show_offset:
            text_h = 3
            if row + text_h <= fb_rows:
                off_arr = render_simple_text_to_pixel_array(
                    f"Offset: {_dt.timedelta(seconds=offset)}",
                    available_width,
                    text_h,
                    font_size=10,
                    theme_manager=theme_manager,
                )
                buf[row:row+text_h, :available_width] = off_arr
                row += text_h

        return buf

    def render_fn():
        elapsed = time.perf_counter() - start
        offset = get_offset()
        system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
        internet = system + _dt.timedelta(seconds=offset)

        h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
        m, rem = divmod(rem, 60 * 1000)
        s, ms = divmod(rem, 1000)
        stopwatch = f"{h:02}:{m:02}:{s:02}.{ms:03}"

        frame = compose_full_frame(system, internet, stopwatch, offset)
        img = Image.fromarray(frame, mode="RGB")
        img = theme_manager.apply_effects(img)
        img = theme_manager.apply_theme(img)
        return np.array(img)

    render_thread = threading.Thread(
        target=render_loop,
        args=(framebuffer, render_fn, 10, stop_event),
        daemon=True,
    )
    render_thread.start()

    full_clear_and_reset_cursor()
    input_buffer = ""
    try:
        while True:
            diff_pixels = framebuffer.get_diff_and_promote()
            changed = [
                (y, x, np.array([[color]], dtype=np.uint8))
                for y, x, color in diff_pixels
            ]
            draw_diff(changed)
            while True:
                key = getch_timeout(0)
                if not key:
                    break
                input_buffer += key

            cmd = input_buffer.strip().lower()
            running = True
            if cmd in {"q", "quit", "exit"}:
                stop_event.set()
                running = False
            elif cmd:
                for char in cmd:
                    if char == 'a':
                        theme_manager.cycle_ascii_style()
                    elif char == 'i':
                        theme_manager.toggle_clock_inversion()
                    elif char == 'b':
                        theme_manager.toggle_backdrop_inversion()
                    elif char == 't':
                        theme_manager.cycle_palette(1)
                    elif char == 'T':
                        theme_manager.cycle_palette(-1)
            input_buffer = ""
            if not running:
                break
            time.sleep(args.refresh_rate)

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
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
