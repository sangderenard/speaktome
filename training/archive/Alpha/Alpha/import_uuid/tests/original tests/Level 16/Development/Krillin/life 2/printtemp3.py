import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from typing import Dict, Tuple, List
import time

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

    def __init__(self, page_width: int = 800, page_height: int = 1200, margin: int = 50):
        """
        Initialize the Printing Press with empty type cases and set up the platen.

        Args:
            page_width (int): The width of the page in points.
            page_height (int): The height of the page in points.
            margin (int): The margin to apply on all sides of the page.
        """
        self.type_cases: Dict[Tuple[str, int], torch.Tensor] = {}  # Storage for glyph matrices
        self.type_indices: Dict[Tuple[str, int], Dict[str, int]] = {}  # Mapping of letters to type indices
        self.type_sizes: Dict[Tuple[str, int], Tuple[int, int]] = {}  # Dimensions of each glyph
        self.compositor_stone: List[str] = []  # The cache where lines await assembly
        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin  # Margins on the page

        # Constants noted by the seasoned craftsmen
        self.line_spacing = 10  # Extra space between lines (points)
        self.letter_spacing = 2  # Extra space between letters (points)

        # Mechanical nuances, cherished imperfections
        self.jitter_max = 2  # Maximum misalignment due to mechanical quirks (points)
        self.jitter_seed = 42  # Seed for consistent, artful randomness

    def load_type_case(self, font_path: str, font_size: int):
        """
        Prepare the type case by casting the glyphs in metal.

        Args:
            font_path (str): Path to the font file.
            font_size (int): Size of the font.
        """
        key = (font_path, font_size)
        if key in self.type_cases:
            return  # Type case already prepared

        # Cast each glyph into its metal form
        font = ImageFont.truetype(font_path, font_size)
        characters = [chr(i) for i in range(32, 127)]  # Standard ASCII
        glyphs = []
        max_width = max_height = 0

        for idx, char in enumerate(characters):
            # Render the character onto a fresh piece of parchment
            image = Image.new('L', font.getsize(char), color=0)
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), char, fill=255, font=font)
            tensor = torch.from_numpy(np.array(image, dtype=np.uint8))
            glyphs.append(tensor)
            width, height = image.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)

        # Adjust each glyph to match the largest dimensions, for uniformity
        for i, glyph in enumerate(glyphs):
            h, w = glyph.shape
            pad_width = max_width - w
            pad_height = max_height - h
            glyphs[i] = F.pad(glyph, (0, pad_width, 0, pad_height), value=0)

        # Store the type case and mappings
        self.type_cases[key] = torch.stack(glyphs)  # Shape: (num_chars, max_height, max_width)
        self.type_indices[key] = {char: idx for idx, char in enumerate(characters)}
        self.type_sizes[key] = (max_height, max_width)  # Height and width of the glyphs

        # A small note in the margin: "Mind the serifs; they hold secrets."

    def set_compositor_stone(self, lines: List[str]):
        """
        Lay out the lines onto the compositor's stone, ready for assembly.

        Args:
            lines (List[str]): The lines of text to prepare.
        """
        self.compositor_stone.extend(lines)

    def assemble_page(self, font_path: str, font_size: int) -> List[torch.Tensor]:
        """
        Assemble the page by arranging the lines and slicing into pages.

        Args:
            font_path (str): Path to the font file.
            font_size (int): Size of the font.

        Returns:
            List[torch.Tensor]: A list of tensors, each representing a page.
        """
        key = (font_path, font_size)
        self.load_type_case(font_path, font_size)
        glyphs = self.type_cases[key]  # Shape: (num_chars, glyph_height, glyph_width)
        char_to_idx = self.type_indices[key]
        glyph_height, glyph_width = self.type_sizes[key]

        # Prepare text as indices tensor
        lines = self.compositor_stone.copy()
        total_lines = len(lines)
        max_line_length = max(len(line) for line in lines)
        max_chars_per_line = (self.page_width - 2 * self.margin) // (glyph_width + self.letter_spacing)
        max_line_length = min(max_line_length, max_chars_per_line)

        indices_list = []
        for line in lines:
            line_chars = list(line)[:max_line_length]
            line_indices = [char_to_idx.get(char, char_to_idx.get(' ', 0)) for char in line_chars]
            padding_length = max_line_length - len(line_indices)
            if padding_length > 0:
                line_indices.extend([char_to_idx.get(' ', 0)] * padding_length)
            indices_list.append(line_indices)

        indices_tensor = torch.tensor(indices_list, dtype=torch.long)  # Shape: (total_lines, max_line_length)

        # Calculate positions
        glyph_width_with_spacing = glyph_width + self.letter_spacing
        glyph_height_with_spacing = glyph_height + self.line_spacing

        x_offsets_chars = torch.arange(max_line_length) * glyph_width_with_spacing  # (max_line_length,)
        y_offsets_lines = torch.arange(total_lines) * glyph_height_with_spacing  # (total_lines,)

        torch.manual_seed(self.jitter_seed)
        x_jitter = torch.randint(-self.jitter_max, self.jitter_max + 1, (total_lines, max_line_length))
        y_jitter = torch.randint(-self.jitter_max, self.jitter_max + 1, (total_lines, 1))

        x_offsets = x_offsets_chars.unsqueeze(0) + x_jitter  # (total_lines, max_line_length)
        y_offsets = y_offsets_lines.unsqueeze(1) + y_jitter  # (total_lines, 1)

        x_positions = x_offsets + self.margin
        y_positions = y_offsets + self.margin

        # Grand sheet dimensions
        total_width = int(x_positions.max().item() + glyph_width + self.margin)
        total_height = int(y_positions.max().item() + glyph_height + self.margin)

        # Initialize grand sheet
        grand_sheet = torch.zeros((total_height, total_width), dtype=torch.uint8)

        # Fetch glyphs
        glyphs_tensor = glyphs[indices_tensor]  # (total_lines, max_line_length, glyph_height, glyph_width)

        # Flatten tensors
        num_glyphs = total_lines * max_line_length
        glyphs_flat = glyphs_tensor.reshape(num_glyphs, glyph_height, glyph_width)
        x_positions_flat = x_positions.reshape(num_glyphs)
        y_positions_flat = y_positions.reshape(num_glyphs)

        # Prepare pixel coordinates
        gx = torch.arange(glyph_width).unsqueeze(0).expand(glyph_height, glyph_width)
        gy = torch.arange(glyph_height).unsqueeze(1).expand(glyph_height, glyph_width)

        gx = gx.unsqueeze(0).expand(num_glyphs, glyph_height, glyph_width)
        gy = gy.unsqueeze(0).expand(num_glyphs, glyph_height, glyph_width)

        x_coords = x_positions_flat.unsqueeze(1).unsqueeze(2) + gx  # (num_glyphs, glyph_height, glyph_width)
        y_coords = y_positions_flat.unsqueeze(1).unsqueeze(2) + gy  # (num_glyphs, glyph_height, glyph_width)

        x_coords_flat = x_coords.reshape(-1).long()
        y_coords_flat = y_coords.reshape(-1).long()
        glyphs_flat_values = glyphs_flat.reshape(-1)

        # Filter valid positions
        valid_mask = (x_coords_flat >= 0) & (x_coords_flat < total_width) & (y_coords_flat >= 0) & (y_coords_flat < total_height)
        x_coords_flat = x_coords_flat[valid_mask]
        y_coords_flat = y_coords_flat[valid_mask]
        glyphs_flat_values = glyphs_flat_values[valid_mask]

        # Place glyphs onto grand sheet
        grand_sheet.index_put_((y_coords_flat, x_coords_flat), glyphs_flat_values, accumulate=True)

        # Slice grand sheet into pages
        pages = []
        page_height = self.page_height
        for page_start in range(0, total_height, page_height):
            page_end = min(page_start + page_height, total_height)
            page_tensor = grand_sheet[page_start:page_end, :]
            pages.append(page_tensor)

        return pages

    def print_pages(self, pages: List[torch.Tensor]):
        """
        Output the pages as images.

        Args:
            pages (List[torch.Tensor]): The list of page tensors to print.
        """
        for idx, page in enumerate(pages):
            image = Image.fromarray(page.numpy(), mode='L')
            image.save(f'page_{idx + 1}.png')
            image.show()

# Initialize the Printing Press
press = PrintingPress(page_width=800, page_height=1200, margin=50)

# Prepare the text
text_lines = [
    "Once upon a time, in a land where code and art entwined,",
    "there stood a Printing Press unlike any other.",
    "Crafted not of steel and wood, but of logic and precision,",
    "it pressed ideas onto the canvas of the digital realm.",
    "With a single, monumental operation, it whispered:",
    "\"Behold the power of collective thought.\"",
]

# Set the compositor's stone with the lines
press.set_compositor_stone(text_lines)

# Assemble the pages using the desired font and size
font_path = "arial.ttf"  # Replace with actual font path
font_size = 24
pages = press.assemble_page(font_path=font_path, font_size=font_size)

# Print the pages
press.print_pages(pages)
