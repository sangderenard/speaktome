"""Demonstrate time synchronization with analog and digital clocks."""

from __future__ import annotations

import datetime as _dt
import argparse
import time
import json
import os

from time_sync import (
    get_offset, sync_offset,
    print_analog_clock, print_digital_clock,
    init_colorama_for_windows, reset_cursor_to_top, full_clear_and_reset_cursor
)
from time_sync.theme_manager import ThemeManager, ClockTheme # Ensure this import works

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
    if not os.path.exists(presets_file_path) and os.path.exists("presets/default_themes.json"): # If run from repo root
        presets_file_path = "presets/default_themes.json" 
    elif not os.path.exists(presets_file_path) and os.path.exists("time_sync/presets/default_themes.json"): # If run from one level up
        presets_file_path = "time_sync/presets/default_themes.json"

    theme_manager = ThemeManager(presets_path=presets_file_path)
    
    try:
        while True:
            elapsed = time.perf_counter() - start
            offset = get_offset()
            system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
            internet = system + _dt.timedelta(seconds=offset)

            h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
            m, rem = divmod(rem, 60 * 1000)
            s, ms = divmod(rem, 1000)
            stopwatch = f"{h:02}:{m:02}:{s:02}.{ms:03}"

            # Move cursor to top to overwrite, reducing flicker.
            # Note: If new content is shorter than old, remnants may remain.
            reset_cursor_to_top()

            # Set current theme based on args (or allow interactive changes to override)
            # This should ideally only be done once or when args change, but for simplicity here:
            if not hasattr(theme_manager, "_initial_theme_set"): # Apply CLI theme args once
                theme_manager.current_theme.palette = theme_manager.presets["color_palettes"].get(args.theme, {})
                theme_manager.current_theme.effects = theme_manager.presets["effects_presets"].get(args.effects, {})
                theme_manager.current_theme.ascii_style = args.ascii_style # Use the direct style name
                theme_manager.current_theme.post_processing = theme_manager.presets["post_processing"].get(args.post_processing, {})
                theme_manager._initial_theme_set = True


            # Prepare analog clock config
            analog_params = default_analog_config.copy()
            analog_params["backdrop_image_path"] = args.analog_backdrop if args.analog_backdrop is not None else args.backdrop
            if args.analog_clock_rect: # CLI overrides JSON
                analog_params["clock_drawing_rect_on_canvas"] = args.analog_clock_rect
            
            # Prepare digital clock config
            digital_params = default_digital_config.copy()
            digital_params["backdrop_image_path"] = args.digital_backdrop if args.digital_backdrop is not None else args.backdrop
            # Add CLI overrides for digital params if any were added

            if args.show_analog:
                print_analog_clock(
                    internet,
                    theme_manager=theme_manager,
                    **analog_params
                )
                print()
            if args.show_digital_system:
                print_digital_clock(
                    system, 
                    theme_manager=theme_manager,
                    **digital_params
                )
                print()
            if args.show_digital_internet:
                print_digital_clock(
                    internet, 
                    theme_manager=theme_manager,
                    **digital_params)
                print()
            if args.show_stopwatch:
                print(f"Stopwatch: {stopwatch}")
            if args.show_offset:
                print(f"Offset: {_dt.timedelta(seconds=offset)}")
            
            key = getch_timeout(args.refresh_rate)
            if key:
                if key.lower() == 'q':
                    break
                elif key.lower() == 'a':
                    theme_manager.cycle_ascii_style()
                elif key.lower() == 'i':
                    theme_manager.toggle_clock_inversion()
                elif key.lower() == 'b':
                    theme_manager.toggle_backdrop_inversion()
                elif key.lower() == 't':
                    # Cycle color palettes
                    palettes = list(theme_manager.presets["color_palettes"].keys())
                    idx = palettes.index(theme_manager.current_theme.palette.get("name", args.theme)) if "name" in theme_manager.current_theme.palette else palettes.index(args.theme)
                    next_palette = palettes[(idx + 1) % len(palettes)]
                    theme_manager.current_theme.palette = theme_manager.presets["color_palettes"][next_palette]
                    theme_manager.current_theme.palette["name"] = next_palette
                elif key.lower() == 'e':
                    # Cycle effects
                    effects = list(theme_manager.presets["effects_presets"].keys())
                    idx = effects.index(theme_manager.current_theme.effects.get("name", args.effects)) if "name" in theme_manager.current_theme.effects else effects.index(args.effects)
                    next_effect = effects[(idx + 1) % len(effects)]
                    theme_manager.current_theme.effects = theme_manager.presets["effects_presets"][next_effect]
                    theme_manager.current_theme.effects["name"] = next_effect
                elif key.lower() == 'p':
                    # Cycle post-processing
                    posts = list(theme_manager.presets["post_processing"].keys())
                    idx = posts.index(theme_manager.current_theme.post_processing.get("name", args.post_processing)) if "name" in theme_manager.current_theme.post_processing else posts.index(args.post_processing)
                    next_post = posts[(idx + 1) % len(posts)]
                    theme_manager.current_theme.post_processing = theme_manager.presets["post_processing"][next_post]
                    theme_manager.current_theme.post_processing["name"] = next_post
            else: # No key pressed, just sleep for the remaining refresh cycle
                time.sleep(args.refresh_rate)

    except KeyboardInterrupt:
        # On exit, do a full clear for a clean terminal.
        full_clear_and_reset_cursor()
        print("Demo stopped.")
        print(f"Final system time: {system.strftime('%H:%M:%S')}")
        print(f"Final internet time: {internet.strftime('%H:%M:%S')}")
        print(f"Offset: {_dt.timedelta(seconds=offset)}")


if __name__ == "__main__":
    main()
