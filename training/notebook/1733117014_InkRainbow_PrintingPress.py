import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
import numpy as np
class InkRainbow:
    """
    **InkRainbow**
    A tool for applying FFT-driven rainbow ink to glyph tensors.

    This class defines the spectral transition behavior of ink, ensuring
    dynamic and seamless blending across text while preserving artistic
    and symbolic integrity.
    """
    def __init__(self, wavelength_range: Tuple[float, float] = (400, 700), bleed_factor: float = 0.1):
        """
        Initialize the rainbow ink design parameters.

        Args:
            wavelength_range (Tuple[float, float]): The range of wavelengths (in nm) to simulate.
            bleed_factor (float): The degree of ink bleeding across glyphs.
        """
        self.wavelength_range = wavelength_range
        self.bleed_factor = bleed_factor

    def apply_fft_rainbow(self, tensor_page: torch.Tensor, fft_data: torch.Tensor) -> torch.Tensor:
        """
        Apply rainbow ink using FFT-modulated spectra.

        Args:
            tensor_page (torch.Tensor): The tensor representing the page.
            fft_data (torch.Tensor): The FFT data dictating spectral transitions.

        Returns:
            torch.Tensor: The tensor with rainbow ink applied.
        """
        height, width = tensor_page.shape
        spectrum = torch.linspace(self.wavelength_range[0], self.wavelength_range[1], width)
        color_map = self._generate_color_map(spectrum, fft_data)

        # Create RGB page tensor
        rgb_page = torch.zeros((3, height, width), dtype=torch.float32)

        # Apply spectral colors to the page
        for y in range(height):
            for x in range(width):
                if tensor_page[y, x] > 0:
                    rgb_page[:, y, x] = color_map[:, x] * tensor_page[y, x] / 255.0

        return rgb_page

    def _generate_color_map(self, spectrum: torch.Tensor, fft_data: torch.Tensor) -> torch.Tensor:
        """
        Generate a color map based on the spectrum and FFT data.

        Args:
            spectrum (torch.Tensor): The wavelengths across the page.
            fft_data (torch.Tensor): The FFT data dictating spectral transitions.

        Returns:
            torch.Tensor: A tensor representing the RGB color map.
        """
        colors = []
        for wavelength in spectrum:
            intensity = fft_data[int(wavelength) % len(fft_data)]  # Modulate by FFT
            color = self._wavelength_to_rgb(wavelength, intensity)
            colors.append(color)
        return torch.stack(colors, dim=1)  # Shape: (3, width)

    def _wavelength_to_rgb(self, wavelength: float, intensity: float) -> torch.Tensor:
        """
        Convert a wavelength to an RGB color.

        Args:
            wavelength (float): The wavelength in nm.
            intensity (float): The intensity of the color.

        Returns:
            torch.Tensor: The RGB color tensor.
        """
        if 380 <= wavelength < 440:
            r = -(wavelength - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= wavelength < 490:
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif 490 <= wavelength < 510:
            r = 0.0
            g = 1.0
            b = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= wavelength < 645:
            r = 1.0
            g = -(wavelength - 645) / (645 - 580)
            b = 0.0
        elif 645 <= wavelength <= 780:
            r = 1.0
            g = 0.0
            b = 0.0
        else:
            r = g = b = 0.0

        # Intensity adjustment
        factor = intensity * 255.0
        return torch.tensor([r, g, b]) * factor

class PrintingPress:
    """
    The Grand Artisanal Printing Press
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    "In an age where whispers of electrons carry the weight of words,
    we return to the elegance of the mechanical, the tangible, the profound.
    This press does not merely print—it orchestrates a symphony of pixels,
    each playing its part in perfect harmony.

    Here, in this code, lies the soul of a craftsman,
    weaving threads of logic into tapestries of text.
    No loop disrupts the flow, no haste mars the perfection.
    With one monumental turn of the handle, the grand sheet is pressed,
    a testament to the beauty of collective effort.

    Let this printing press stand as a tribute to those who believe
    that efficiency need not forsake artistry, and that code,
    like the finest of machines, can be both purposeful and poetic."

    ~ Crafted with pride and precision
    """

    def __init__(self, page_width=800, page_height=1200, margin=50, 
                 wavelength_range=(400, 700), bleed_factor=0.1):

        self.glyph_libraries = {}
        self.char_to_idx = {}
        self.font_sizes = {}
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin
        self.line_spacing = 10
        self.letter_spacing = 2
        self.ink_rainbow = InkRainbow(wavelength_range, bleed_factor) 

    def load_font(self, font_path, font_size):
        """
        Load a font and generate glyph tensors in parallel.
        """
        font = ImageFont.truetype(font_path, font_size)
        characters = [chr(i) for i in range(32, 127)]
        
        # Generate all glyph images in parallel
        glyph_images = [
            self._render_char_to_image(font, char) for char in characters
        ]

        # Calculate bounding boxes in parallel
        bboxes = [
            font.getbbox(char) for char in characters
        ]
        widths = torch.tensor([bbox[2] - bbox[0] for bbox in bboxes])
        heights = torch.tensor([bbox[3] - bbox[1] for bbox in bboxes])
        max_width = widths.max()
        max_height = heights.max()

        # Pad glyphs in parallel
        padded_glyphs = torch.stack([
            F.pad(glyph, (0, max_width - w, 0, max_height - h), value=0)
            for glyph, w, h in zip(glyph_images, widths, heights)
        ])

        key = (font_path, font_size)
        self.glyph_libraries[key] = padded_glyphs
        self.char_to_idx[key] = {char: idx for idx, char in enumerate(characters)}
        self.font_sizes[key] = (max_height, max_width)

    def _render_char_to_image(self, font, char):
        """
        Render a single character to a tensor image.
        """
        image = Image.new('L', font.getsize(char), color=0)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), char, fill=255, font=font)
        return torch.from_numpy(np.array(image, dtype=np.uint8))

    def print_text(self, text, font_path, font_size, 
                  fft_data=None,  # Add FFT data for color modulation
                  fancy_font_path=None, fancy_font_size=None):
        """
        Print text with rainbow ink and optional drop caps.
        """
        key = (font_path, font_size)
        if key not in self.glyph_libraries:
            self.load_font(font_path, font_size)
        if fancy_font_path and fancy_font_size:
            fancy_key = (fancy_font_path, fancy_font_size)
            if fancy_key not in self.glyph_libraries:
                self.load_font(fancy_font_path, fancy_font_size)
        else:
            fancy_key = key 

        lines = text.split('\n')
        num_lines = len(lines)
        reg_height, reg_width = self.font_sizes[key]
        fan_height, fan_width = self.font_sizes[fancy_key]

        # Calculate max line length considering drop cap
        max_line_length = max(len(line) for line in lines) 
        if fancy_font_path:
            max_line_length = max(max_line_length - 1, 0)  # Account for drop cap

        # Calculate output dimensions
        output_height = num_lines * (reg_height + self.line_spacing) - self.line_spacing
        output_width = (max_line_length * (reg_width + self.letter_spacing) 
                        - self.letter_spacing + fan_width + 2 * self.margin)

        output_tensor = torch.zeros((output_height, output_width), dtype=torch.uint8)

        # --- Parallel Glyph Placement ---

        # Generate indices for all glyphs in all lines
        all_indices = []
        for i, line in enumerate(lines):
            line_chars = list(line)
            if i == 0 and fancy_font_path:
                # Drop cap handling
                first_char = line_chars[0]
                rest_chars = line_chars[1:]
                fancy_idx = self.char_to_idx[fancy_key].get(first_char, 0)
                all_indices.append([fancy_idx] + [self.char_to_idx[key].get(char, 0) for char in rest_chars])
            else:
                all_indices.append([self.char_to_idx[key].get(char, 0) for char in line_chars])

        # Pad indices to ensure consistent tensor shape
        max_len = max(len(indices) for indices in all_indices)
        all_indices = [indices + [0] * (max_len - len(indices)) for indices in all_indices]
        indices_tensor = torch.tensor(all_indices, dtype=torch.long)

        # Calculate x and y offsets for each glyph
        x_offsets_chars = torch.arange(max_len) * (reg_width + self.letter_spacing)
        y_offsets_lines = torch.arange(num_lines) * (reg_height + self.line_spacing)
        x_offsets = x_offsets_chars.unsqueeze(0) + self.margin + fan_width  # Add drop cap offset
        y_offsets = y_offsets_lines.unsqueeze(1) + self.margin

        # Get all glyphs as a single tensor
        glyphs_tensor = torch.stack([
            self.glyph_libraries[key][line_indices] for line_indices in all_indices
        ])

        # Flatten tensors for parallel placement
        num_glyphs = num_lines * max_len
        glyphs_flat = glyphs_tensor.reshape(num_glyphs, reg_height, reg_width)
        x_positions_flat = x_offsets.reshape(num_glyphs)
        y_positions_flat = y_offsets.reshape(num_glyphs)

        # Prepare pixel coordinates
        gx = torch.arange(reg_width).unsqueeze(0).expand(reg_height, reg_width)
        gy = torch.arange(reg_height).unsqueeze(1).expand(reg_height, reg_width)
        gx = gx.unsqueeze(0).expand(num_glyphs, reg_height, reg_width)
        gy = gy.unsqueeze(0).expand(num_glyphs, reg_height, reg_width)
        x_coords = x_positions_flat.unsqueeze(1).unsqueeze(2) + gx
        y_coords = y_positions_flat.unsqueeze(1).unsqueeze(2) + gy
        x_coords_flat = x_coords.reshape(-1).long()
        y_coords_flat = y_coords.reshape(-1).long()
        glyphs_flat_values = glyphs_flat.reshape(-1)

        # Filter valid positions
        valid_mask = (x_coords_flat >= 0) & (x_coords_flat < output_width) & (y_coords_flat >= 0) & (y_coords_flat < output_height)
        x_coords_flat = x_coords_flat[valid_mask]
        y_coords_flat = y_coords_flat[valid_mask]
        glyphs_flat_values = glyphs_flat_values[valid_mask]

        # Place glyphs onto the output tensor
        output_tensor.index_put_((y_coords_flat, x_coords_flat), glyphs_flat_values, accumulate=True)

        # --- Apply Rainbow Ink ---
        if fft_data is not None:
            output_tensor = self.ink_rainbow.apply_fft_rainbow(output_tensor, fft_data)

        return output_tensor


import markdown
import yaml
from bs4 import BeautifulSoup
from collections import defaultdict

def parse_markup_to_typesetting_data(markup: str) -> dict:
    """
    Parses a markup document into a structured dictionary object for typesetting instructions,
    specifically designed for the Hot-Rod Printing Press tool.

    Args:
        markup (str): The input markup document in a standard markup language (Markdown).

    Returns:
        dict: A structured dictionary with parameterized typesetting instructions.
    """
    # Convert Markdown to HTML using markdown package
    html_content = markdown.markdown(markup)

    # Parse HTML content with BeautifulSoup to extract elements
    soup = BeautifulSoup(html_content, 'html.parser')

    typesetting_dict = defaultdict(lambda: defaultdict(dict))

    # Extract and categorize content based on HTML tags
    for element in soup.descendants:
        if element.name == 'h1':
            typesetting_dict['header']['text'] = element.get_text()
            typesetting_dict['header']['font'] = 'Arial'
            typesetting_dict['header']['size'] = 24
            typesetting_dict['header']['alignment'] = 'center'
        elif element.name == 'p':
            if 'body' not in typesetting_dict:
                typesetting_dict['body']['text'] = ''
            typesetting_dict['body']['text'] += element.get_text() + '\n'
            typesetting_dict['body']['font'] = 'TimesNewRoman'
            typesetting_dict['body']['size'] = 12
            typesetting_dict['body']['alignment'] = 'justify'
        elif element.name == 'footer':
            typesetting_dict['footer']['text'] = element.get_text()
            typesetting_dict['footer']['font'] = 'Courier'
            typesetting_dict['footer']['size'] = 10
            typesetting_dict['footer']['alignment'] = 'right'

    return dict(typesetting_dict)

# Example usage of the tool
markup_text = """
# The Grand Opening

Welcome to the Hot-Rod Printing Press, where technology meets art.

<footer>Page 1 of 10</footer>
"""

# Parse the markup to a structured dictionary
typesetting_data = parse_markup_to_typesetting_data(markup_text)
print(yaml.dump(typesetting_data, default_flow_style=False))

# Output will be a structured dictionary suitable for the Hot-Rod Printing Press,
# including parameterized typesetting instructions for fonts, alignments, and content.
