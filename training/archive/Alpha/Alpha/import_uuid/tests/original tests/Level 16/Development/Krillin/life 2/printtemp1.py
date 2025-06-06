import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from typing import Dict, Tuple, List
import math
import time

class PrintingPress:
    """
    The Grand Mechanical Printing Press
    ~ Established in the Year of Our Lord ~

    Crafted for the precise and elegant transfer of words to page.
    Handle with care and respect its mechanisms.
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
        self.time_delay = 0.5  # Delay to simulate time between line compositions (in seconds)
        self.last_composition_time = time.time()

        # Constants noted by old hands
        self.line_spacing = 10  # Extra space between lines (points)
        self.letter_spacing = 2  # Extra space between letters (points)

        # Creaky adjustments, mind the temperamental nature
        self.jitter_max = 2  # Maximum misalignment due to mechanical imperfections (points)

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
        glyphs = self.type_cases[key]
        char_to_idx = self.type_indices[key]
        glyph_height, glyph_width = self.type_sizes[key]

        # Determine the number of glyphs that fit in a line and the number of lines per page
        usable_width = self.page_width - 2 * self.margin
        usable_height = self.page_height - 2 * self.margin
        max_chars_per_line = (usable_width + self.letter_spacing) // (glyph_width + self.letter_spacing)
        max_lines_per_page = (usable_height + self.line_spacing) // (glyph_height + self.line_spacing)

        # Assemble all lines into a single matrix
        lines = self.compositor_stone.copy()
        total_lines = len(lines)

        # Prepare the grand sheet to hold all lines
        total_height = total_lines * (glyph_height + self.line_spacing)
        total_width = self.page_width
        grand_sheet = torch.zeros((total_height, total_width), dtype=torch.uint8)

        # Place each line onto the grand sheet
        for i, line in enumerate(lines):
            y_offset = i * (glyph_height + self.line_spacing)
            x_offset = self.margin  # Start at left margin

            # Convert line to glyph indices
            line_chars = list(line)[:max_chars_per_line]  # Truncate if too long
            indices = [char_to_idx.get(char, 0) for char in line_chars]
            line_glyphs = glyphs[indices]  # Shape: (num_chars, glyph_height, glyph_width)

            # Assemble line into a tensor
            line_tensor = line_glyphs.permute(1, 0, 2).reshape(glyph_height, -1)

            # Apply letter spacing
            if self.letter_spacing > 0:
                spaced_line = torch.zeros((glyph_height, line_tensor.shape[1] + self.letter_spacing * (len(line_chars) - 1)), dtype=torch.uint8)
                pos = 0
                for idx, char_glyph in enumerate(line_glyphs):
                    w = char_glyph.shape[1]
                    spaced_line[:, pos:pos + w] = char_glyph
                    pos += w + self.letter_spacing
                line_tensor = spaced_line

            # Apply jitter for mechanical imperfections
            x_jitter = torch.randint(-self.jitter_max, self.jitter_max + 1, (1,)).item()
            y_jitter = torch.randint(-self.jitter_max, self.jitter_max + 1, (1,)).item()

            # Place the line onto the grand sheet
            y_start = y_offset + y_jitter
            x_start = x_offset + x_jitter
            h, w = line_tensor.shape
            grand_sheet[y_start:y_start + h, x_start:x_start + w] = line_tensor

            # Simulate time delay between line compositions
            time_since_last = time.time() - self.last_composition_time
            if time_since_last < self.time_delay:
                time.sleep(self.time_delay - time_since_last)
            self.last_composition_time = time.time()

        # Slice the grand sheet into pages
        pages = []
        for page_start in range(0, total_height, usable_height + self.line_spacing):
            page_end = page_start + usable_height
            page_tensor = grand_sheet[page_start:page_end, :]
            # Apply margins
            page_with_margins = torch.zeros((self.page_height, self.page_width), dtype=torch.uint8)
            page_with_margins[self.margin:self.margin + page_tensor.shape[0], self.margin:self.margin + page_tensor.shape[1]] = page_tensor
            pages.append(page_with_margins)

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
    "In the beginning God created the heaven and the earth.",
    "And the earth was without form, and void; and darkness was upon the face of the deep.",
    "And the Spirit of God moved upon the face of the waters.",
    "And God said, Let there be light: and there was light.",
    # ... (more lines)
]

# Set the compositor's stone with the lines
press.set_compositor_stone(text_lines)

# Assemble the pages using the desired font and size
font_path = "arial.ttf"  # Replace with actual font path
font_size = 24
pages = press.assemble_page(font_path=font_path, font_size=font_size)

# Print the pages
press.print_pages(pages)
