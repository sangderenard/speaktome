from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter

@dataclass
class ClockTheme:
    palette: dict
    effects: dict
    ascii_style: str
    post_processing: dict
    invert_clock: bool = False
    invert_backdrop: bool = False

class ThemeManager:
    def __init__(self, presets_path: str = "presets/default_themes.json"):
        self.presets_path = presets_path
        self.current_theme = ClockTheme({}, {}, "block", {})
        self._load_presets()

    def _load_presets(self) -> None:
        try:
            with open(self.presets_path, 'r', encoding='utf-8') as f:
                self.presets = json.load(f)
        except Exception as e:
            print(f"Error loading presets: {e}")
            self.presets = {"color_palettes": {}, "effects_presets": {}, 
                          "ascii_styles": {}, "post_processing": {}}

    def apply_theme(self, image: Image.Image) -> Image.Image:
        """Apply current theme's post-processing to image"""
        if self.current_theme.invert_clock:
            image = Image.eval(image, lambda x: 255 - x)
        
        pp = self.current_theme.post_processing
        if pp:
            brightness = pp.get("brightness", 1.0)
            contrast = pp.get("contrast", 1.0)
            saturation = pp.get("saturation", 1.0)
            
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)

        return image

    def apply_effects(self, image: Image.Image) -> Image.Image:
        """Apply current theme's effects to image"""
        effects = self.current_theme.effects
        if effects:
            if effects.get("glow_radius", 0) > 0:
                glow = image.filter(ImageFilter.GaussianBlur(effects["glow_radius"]))
                image = Image.blend(image, glow, effects["glow_intensity"])
            
            if effects.get("shadow_blur", 0) > 0:
                # Create and blend shadow...
                pass  # (Implementation details for shadow effect)

        return image

    def get_current_ascii_ramp(self) -> str:
        """Get current ASCII style ramp.
        Falls back to "block" style if current is not found,
        then to a hardcoded default if "block" is also not found (e.g. due to loading error).
        """
        # Hardcoded default ramp, matching the "block" style in default_themes.json
        DEFAULT_FALLBACK_RAMP = " .:░▒▓█"

        ascii_styles_preset = self.presets.get("ascii_styles", {})

        # 1. Try to get the ramp for the current theme's ascii_style
        current_style_ramp = ascii_styles_preset.get(self.current_theme.ascii_style)
        if current_style_ramp is not None: # Check for None in case a style has an empty string ramp
            return current_style_ramp

        # 2. If current theme's style not found, try to get the "block" style ramp
        block_style_ramp = ascii_styles_preset.get("block")
        if block_style_ramp is not None:
            return block_style_ramp
        
        # 3. If "block" style is also not found, return the hardcoded default fallback ramp.
        return DEFAULT_FALLBACK_RAMP

    def cycle_ascii_style(self) -> str:
        """Cycle to next ASCII style"""
        styles = list(self.presets.get("ascii_styles", {}).keys())
        if not styles: # Fallback if no styles are loaded
            self.current_theme.ascii_style = "block" # Default to "block" style name
            return self.current_theme.ascii_style
        try:
            current_idx = styles.index(self.current_theme.ascii_style)
            next_idx = (current_idx + 1) % len(styles)
            self.current_theme.ascii_style = styles[next_idx]
        except ValueError:
            self.current_theme.ascii_style = styles[0]
        return self.current_theme.ascii_style

    def toggle_clock_inversion(self) -> bool:
        self.current_theme.invert_clock = not self.current_theme.invert_clock
        return self.current_theme.invert_clock

    def toggle_backdrop_inversion(self) -> bool:
        self.current_theme.invert_backdrop = not self.current_theme.invert_backdrop
        return self.current_theme.invert_backdrop

    def set_palette(self, palette_name: str) -> None:
        """Set the color palette by name"""
        if palette_name in self.presets["color_palettes"]:
            palette = self.presets["color_palettes"][palette_name].copy()
            palette["name"] = palette_name
            self.current_theme.palette = palette
        else:
            print(f"Palette '{palette_name}' not found.")

    def set_effects(self, effects_name: str) -> None:
        """Set the effects preset by name"""
        if effects_name in self.presets["effects_presets"]:
            effects = self.presets["effects_presets"][effects_name].copy()
            effects["name"] = effects_name
            self.current_theme.effects = effects
        else:
            print(f"Effects preset '{effects_name}' not found.")

    def set_post_processing(self, pp_name: str) -> None:
        """Set the post-processing preset by name"""
        if pp_name in self.presets["post_processing"]:
            post_processing = self.presets["post_processing"][pp_name].copy()
            post_processing["name"] = pp_name
            self.current_theme.post_processing = post_processing
        else:
            print(f"Post-processing preset '{pp_name}' not found.")