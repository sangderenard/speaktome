import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from typing import Dict, Tuple

class PrintingPress:
    def __init__(self):
        """
        Initialize the PrintingPress with empty glyph libraries.
        """
        self.glyph_libraries: Dict[Tuple[str, int], torch.Tensor] = {}
        self.char_to_idx: Dict[Tuple[str, int], Dict[str, int]] = {}
        self.font_sizes: Dict[Tuple[str, int], Tuple[int, int]] = {}

    def load_font(self, font_path: str, font_size: int):
        """
        Load a font and generate glyph tensors.

        Args:
            font_path (str): Path to the .ttf font file.
            font_size (int): Size of the font.
        """
        # Create font object
        font = ImageFont.truetype(font_path, font_size)
        # Get all printable ASCII characters
        characters = [chr(i) for i in range(32, 127)]
        # Render each character and store in a list
        glyphs = []
        widths = []
        heights = []
        for char in characters:
            # Render character to image
            bbox = font.getbbox(char)  # Get the bounding box of the character
            width = bbox[2] - bbox[0]  # Calculate width
            height = bbox[3] - bbox[1]  # Calculate height
            image = Image.new('L', (width, height), color=0)

            draw = ImageDraw.Draw(image)
            draw.text((0, 0), char, fill=255, font=font)
            # Convert to tensor
            tensor = torch.ByteTensor(bytearray(image.tobytes())).view(image.size[1], image.size[0])
            glyphs.append(tensor)
            widths.append(tensor.shape[1])
            heights.append(tensor.shape[0])
        # Determine max width and height
        max_width = max(widths)
        max_height = max(heights)
        # Pad glyphs to have the same dimensions
        padded_glyphs = []
        for glyph in glyphs:
            pad_width = max_width - glyph.shape[1]
            pad_height = max_height - glyph.shape[0]
            padded_glyph = F.pad(glyph, (0, pad_width, pad_height, 0), value=0)
            padded_glyphs.append(padded_glyph)
        # Stack glyphs into a tensor
        glyph_tensor = torch.stack(padded_glyphs)  # Shape: (num_chars, max_height, max_width)
        # Store glyph library
        key = (font_path, font_size)
        self.glyph_libraries[key] = glyph_tensor
        # Create character to index mapping
        self.char_to_idx[key] = {char: idx for idx, char in enumerate(characters)}
        # Store font size information
        self.font_sizes[key] = (max_height, max_width)

    def print_text(self, text: str, font_path: str, font_size: int, fancy_font_path: str = None, fancy_font_size: int = None) -> torch.Tensor:
        """
        Assemble the text into a tensor image with optional drop caps and bottom justification.

        Args:
            text (str): The text to print.
            font_path (str): Path to the regular font file.
            font_size (int): Size of the regular font.
            fancy_font_path (str, optional): Path to the fancy font file for drop caps.
            fancy_font_size (int, optional): Size of the fancy font for drop caps.

        Returns:
            torch.Tensor: The assembled text image tensor.
        """
        # Ensure fonts are loaded
        regular_key = (font_path, font_size)
        if regular_key not in self.glyph_libraries:
            self.load_font(font_path, font_size)
        if fancy_font_path and fancy_font_size:
            fancy_key = (fancy_font_path, fancy_font_size)
            if fancy_key not in self.glyph_libraries:
                self.load_font(fancy_font_path, fancy_font_size)
        else:
            fancy_key = regular_key  # Use regular font if fancy font not provided

        # Split text into lines
        lines = text.split('\n')
        num_lines = len(lines)

        # Get font dimensions
        reg_height, reg_width = self.font_sizes[regular_key]
        fan_height, fan_width = self.font_sizes[fancy_key]

        # Determine output tensor dimensions
        max_line_length = max(len(line) for line in lines)
        output_height = num_lines * reg_height
        output_width = max_line_length * reg_width + fan_width  # Add space for drop cap

        # Initialize output tensor
        output_tensor = torch.zeros((output_height, output_width), dtype=torch.uint8)

        # Prepare line indices (bottom-justified)
        line_indices = torch.arange(num_lines - 1, -1, -1) * reg_height

        # Create grid for positions
        y_offsets = line_indices.repeat_interleave(max_line_length).view(num_lines, max_line_length)
        x_offsets = torch.arange(0, max_line_length * reg_width, reg_width).repeat(num_lines, 1)
        # Adjust x_offsets to account for drop cap space
        x_offsets += fan_width

        # Process each line
        assembled_glyphs = []
        for i, line in enumerate(lines):
            # Handle empty lines
            if not line:
                continue
            # Convert line to character indices
            line_chars = list(line)
            # Handle drop cap for the first character of the first line
            if i == 0 and fancy_key:
                first_char = line_chars[0]
                rest_chars = line_chars[1:]
                # Get glyph indices
                fancy_idx = self.char_to_idx[fancy_key].get(first_char, None)
                if fancy_idx is not None:
                    fancy_glyph = self.glyph_libraries[fancy_key][fancy_idx]
                    # Place fancy glyph at the beginning of the line
                    y = output_height - fan_height
                    x = 0
                    h, w = fancy_glyph.shape
                    output_tensor[y:y+h, x:x+w] = fancy_glyph
                # Process rest of the line with regular font
                line_chars = rest_chars
                x_offset = fan_width
            else:
                x_offset = fan_width
            # Get glyph indices for regular font
            indices = [self.char_to_idx[regular_key].get(char, None) for char in line_chars]
            valid_indices = [idx for idx in indices if idx is not None]
            if not valid_indices:
                continue
            # Stack glyphs
            glyphs = self.glyph_libraries[regular_key][valid_indices]  # Shape: (num_chars, height, width)
            # Assemble line tensor
            line_tensor = glyphs.permute(1, 0, 2).reshape(reg_height, -1)
            # Determine position
            y = output_height - (i + 1) * reg_height
            x = x_offset
            h, w = line_tensor.shape
            # Place line tensor into output tensor
            output_tensor[y:y+h, x:x+w] = line_tensor

        return output_tensor
# Initialize PrintingPress
pp = PrintingPress()

regular_font_path = "arial.ttf"        # Regular font
fancy_font_path = "consola.ttf"        # Fancy font for drop caps

try:
    regular_font = ImageFont.truetype(regular_font_path, 24)
    fancy_font = ImageFont.truetype(fancy_font_path, 36)
    print("Fonts loaded successfully!")
except OSError as e:
    print(f"Error loading fonts: {e}")
# Define font sizes
regular_font_size = 24
fancy_font_size = 48

# Sample text
text = """This is a sample
multiline text
to demonstrate the
PrintingPress class."""

# Print text using PrintingPress
output_image_tensor = pp.print_text(
    text=text,
    font_path=regular_font_path,
    font_size=regular_font_size,
    fancy_font_path=fancy_font_path,
    fancy_font_size=fancy_font_size
)

# Convert tensor to PIL image and save or display
output_image = Image.fromarray(output_image_tensor.numpy(), mode='L')
output_image.save('output.png')  # Save to file
output_image.show()              # Display the image
