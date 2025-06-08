import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import markdown
import yaml
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Tuple

import pdfplumber

class GlyphBuilderEncyclopedia:
    """
    A master reference structure for constructing, organizing, and managing glyphs
    of diverse origins (TTF, FFT, etc.) with metadata concordance and operations.
    """

    def __init__(self):
        self.entries = {}  # All glyphs and metadata
        self.concordance = defaultdict(list)  # Metadata-based indices
        self.active_working_set = None  # Preallocated tensor for work
        self.working_set_indices = None  # Indices for working set

    def add_glyph_entry(self, name: str, tensor: torch.Tensor, metadata: dict):
        """
        Add a glyph entry to the encyclopedia.
        
        Args:
            name (str): Unique name for the glyph.
            tensor (torch.Tensor): The tensor representing the glyph.
            metadata (dict): Metadata describing the glyph's nature and use.
        """
        if name in self.entries:
            raise ValueError(f"Glyph '{name}' already exists.")
        
        # Store the glyph and metadata
        self.entries[name] = {"tensor": tensor, "metadata": metadata}

        # Update concordance for metadata-based indexing
        for key, value in metadata.items():
            self.concordance[key].append(name)

    def get_glyph(self, name: str):
        """
        Retrieve a glyph entry by name.
        
        Args:
            name (str): Name of the glyph.

        Returns:
            dict: The glyph tensor and its metadata.
        """
        return self.entries.get(name)

    def query_concordance(self, **filters):
        """
        Query the concordance for glyphs matching metadata filters.
        
        Args:
            **filters: Metadata filters as key-value pairs.

        Returns:
            list[str]: Names of glyphs matching the filters.
        """
        matches = set(self.entries.keys())
        for key, value in filters.items():
            filtered = {name for name in self.concordance.get(key, []) if self.entries[name]["metadata"].get(key) == value}
            matches &= filtered
        return list(matches)

    def set_working_set(self, names: list):
        """
        Define the working set of glyphs for efficient operations.
        
        Args:
            names (list[str]): Names of glyphs to include in the working set.
        """
        tensors = [self.entries[name]["tensor"] for name in names]
        self.active_working_set = torch.stack(tensors)
        self.working_set_indices = names

    def apply_operation_to_working_set(self, operation: callable, **kwargs):
        """
        Apply an operation to the working set.
        
        Args:
            operation (callable): Function to apply to the working set.
            **kwargs: Additional parameters for the operation.

        Returns:
            torch.Tensor: Result of the operation.
        """
        if self.active_working_set is None:
            raise ValueError("Working set is not defined.")
        return operation(self.active_working_set, **kwargs)

    def build_from_source(self, name: str, source: any, builder: callable, **params):
        """
        Build a glyph from a source and add it to the encyclopedia.
        
        Args:
            name (str): Unique name for the glyph.
            source (any): Source data (e.g., TTF file, FFT data).
            builder (callable): Function to transform the source into a tensor.
            **params: Additional parameters for the builder.
        """
        tensor, metadata = builder(source, **params)
        self.add_glyph_entry(name, tensor, metadata)




class GrandPrintingPress:
    """
    **GrandPrintingPress**
    A comprehensive tool for procedurally rendering large amounts of text into high-intensity
    visual representations, suitable for creating procedural art with comfortable and whimsical
    reading experiences. This class integrates font management, glyph processing, FFT-driven
    rainbow ink application, and markup parsing into a unified system designed for parallel
    processing and high-speed OpenGL rendering.
    """


    def __init__(self, page_width=800, page_height=1200, margin=50, 
                 wavelength_range=(400, 700), bleed_factor=0.1):
        """
        Initialize the GrandPrintingPress with page dimensions and ink parameters.

        Args:
            page_width (int): Width of the page in pixels.
            page_height (int): Height of the page in pixels.
            margin (int): Margin size in pixels.
            wavelength_range (Tuple[float, float]): Range of wavelengths for rainbow ink.
            bleed_factor (float): Degree of ink bleeding across glyphs.
        """
        self.glyph_libraries = {}
        self.kernel_libraries = {}
        self.char_to_idx = {}
        self.font_sizes = {}

        self.line_spacing = 10
        self.letter_spacing = 2
        #self.ink_rainbow = self.InkRainbow(wavelength_range, bleed_factor) 

        self.page_width = page_width
        self.page_height = page_height
        self.margin = margin

        # Data structures for kernels and glyphs
        self.kernel_tools = {}
        self.glyph_tools = {}
        self.kernel_execution_pipeline = []  # Stores the sequence of operations
        self.tensor_library = {}  # Stores glyphs, masks, gradients, paper tensors

    def register_kernel_tool(self, name, kernel_function, metadata=None):
        """
        Register a kernel tool with metadata.

        Args:
            name (str): Name of the kernel tool.
            kernel_function (callable): The kernel function.
            metadata (dict, optional): Metadata for the kernel tool.
        """
        self.kernel_tools[name] = {
            "function": kernel_function,
            "metadata": metadata or {},
        }

    def register_glyph_tool(self, name, glyph_tensor, metadata=None):
        """
        Register a glyph tool with metadata.

        Args:
            name (str): Name of the glyph tool.
            glyph_tensor (torch.Tensor): The glyph tensor.
            metadata (dict, optional): Metadata for the glyph tool.
        """
        self.glyph_tools[name] = {
            "tensor": glyph_tensor,
            "metadata": metadata or {},
        }

    def apply_kernel_stage(self, stage_config):
        """
        Apply a single kernel stage to the tensors in the library.

        Args:
            stage_config (dict): Configuration for the kernel stage.
        """
        # Extract stage parameters
        input_tensors = [self.tensor_library[name] for name in stage_config["inputs"]]
        kernel_name = stage_config["kernel"]
        output_name = stage_config["output"]

        # Retrieve the kernel function
        kernel_tool = self.kernel_tools[kernel_name]
        kernel_function = kernel_tool["function"]

        # Apply the kernel function
        combined_tensor = kernel_function(*input_tensors)

        # Store the result in the tensor library
        self.tensor_library[output_name] = combined_tensor

    def process_with_kernels(self, pipeline):
        """
        Process the press using a pipeline of kernel stages.

        Args:
            pipeline (list[dict]): A list of stage configurations.
        """
        for stage_config in pipeline:
            self.apply_kernel_stage(stage_config)

    # Example kernel function
    @staticmethod
    def blend_kernel(glyph_tensor, gradient_tensor, mask_tensor):
        """
        Example kernel function: blends glyphs, gradients, and masks.

        Args:
            glyph_tensor (torch.Tensor): The glyph tensor.
            gradient_tensor (torch.Tensor): The gradient tensor.
            mask_tensor (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The blended result.
        """
        return glyph_tensor * mask_tensor + gradient_tensor * (1 - mask_tensor)

    # Example of registering and running a kernel stage
    def example_execution(self):
        """
        Example execution of the press with registered kernels and pipeline.
        """
        # Example tensors
        glyph_tensor = torch.rand((1, self.page_height, self.page_width))
        gradient_tensor = torch.linspace(0, 1, steps=self.page_width).repeat(self.page_height, 1).unsqueeze(0)
        mask_tensor = torch.ones_like(glyph_tensor) * 0.5

        # Store tensors in the library
        self.tensor_library["glyph"] = glyph_tensor
        self.tensor_library["gradient"] = gradient_tensor
        self.tensor_library["mask"] = mask_tensor

        # Register a blend kernel
        self.register_kernel_tool("blend", self.blend_kernel, {"description": "Blend kernel for mixing layers"})

        # Define a pipeline
        pipeline = [
            {
                "inputs": ["glyph", "gradient", "mask"],
                "kernel": "blend",
                "output": "blended_result",
            }
        ]

        # Process the pipeline
        self.process_with_kernels(pipeline)

        # Return the final result
        return self.tensor_library["blended_result"]
    def load_font(self, font_path: str, font_size: int):
        """
        Load a font and generate glyph tensors in parallel.

        Args:
            font_path (str): Path to the .ttf or .otf font file.
            font_size (int): Size of the font.
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
    def create_glyph_tool(self, name, source_data, params):
        """
        Create a glyph tool instance from the library.

        Args:
            name (str): Name of the glyph tool.
            source_data: Input data to generate glyphs.
            params (dict): Parameters for the tool function.

        Returns:
            dict: The created glyph tool instance.
        """
        if name not in self.glyph_tool_library:
            raise ValueError(f"Glyph tool '{name}' not found in library.")
        tool_function = self.glyph_tool_library[name]["function"]
        glyph_tensor = tool_function(source_data, **params)
        self.glyph_tools[name] = {
            "tensor": glyph_tensor,
            "metadata": params
        }
        return self.glyph_tools[name]
    def _render_char_to_image(self, font: ImageFont.FreeTypeFont, char: str) -> torch.Tensor:
        """
        Render a single character to a tensor image using accurate bounding box data.

        Args:
            font (ImageFont.FreeTypeFont): The loaded font.
            char (str): The character to render.

        Returns:
            torch.Tensor: Tensor representation of the rendered character.
        """
        # Calculate the bounding box of the character
        bbox = font.getbbox(char)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Create an image with the exact size of the bounding box
        image = Image.new('L', (width, height), color=0)
        draw = ImageDraw.Draw(image)

        # Draw the character at the correct position
        draw.text((-bbox[0], -bbox[1]), char, fill=255, font=font)

        # Convert the image to a PyTorch tensor
        return torch.from_numpy(np.array(image, dtype=np.uint8))


    def parse_markup_to_typesetting_data(self, markup: str) -> dict:
        """
        Parses a markup document into a structured dictionary object for typesetting instructions,
        specifically designed for the GrandPrintingPress tool.

        Args:
            markup (str): The input markup document in a standard markup language (Markdown).

        Returns:
            dict: A structured dictionary with parameterized typesetting instructions.
        """
        # Convert Markdown to HTML using markdown package
        html_content = markdown.markdown(markup)

        # Parse HTML content with BeautifulSoup to extract elements
        soup = BeautifulSoup(html_content, 'html.parser')
        print(soup)
        typesetting_dict = defaultdict(lambda: defaultdict(dict))

        # Extract and categorize content based on HTML tags
        for element in soup.find_all(['h1', 'p', 'footer']):
            if element.name == 'h1':
                typesetting_dict['header'] = {
                    'text': element.get_text(),
                    'font': 'consolas.ttf',  # Specify a valid font file
                    'size': 24,
                    'alignment': 'center',
                }
            elif element.name == 'p':
                if 'body' not in typesetting_dict:
                    typesetting_dict['body'] = {
                        'text': '',
                        'font': 'arial.ttf',  # Specify a valid font file
                        'size': 12,
                        'alignment': 'justify',
                    }
                typesetting_dict['body']['text'] += element.get_text() + '\n'
            elif element.name == 'footer':
                typesetting_dict['footer'] = {
                    'text': element.get_text(),
                    'font': 'consolas.ttf',  # Specify a valid font file
                    'size': 10,
                    'alignment': 'right',
                }

        return dict(typesetting_dict)


    def print_text(self, text: str, font_path: str, font_size: int, 
                  fft_data: torch.Tensor = None,  
                  fancy_font_path: str = None, fancy_font_size: int = None) -> torch.Tensor:
        """
        Render text with rainbow ink and optional drop caps.

        Args:
            text (str): The input text to render.
            font_path (str): Path to the primary font file.
            font_size (int): Size of the primary font.
            fft_data (torch.Tensor, optional): FFT data for color modulation.
            fancy_font_path (str, optional): Path to the fancy font file for drop caps.
            fancy_font_size (int, optional): Size of the fancy font.

        Returns:
            torch.Tensor: The rendered text as a tensor with applied rainbow ink.
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
        # Create a grid of offsets for all characters
        x_positions_flat = x_offsets.repeat(num_lines, 1)  # Repeat x_offsets for each line
        y_positions_flat = y_offsets.repeat(1, max_len)  # Expand y_offsets across all characters in a line

        # Flatten the resulting grids
        x_positions_flat = x_positions_flat.flatten()
        y_positions_flat = y_positions_flat.flatten()

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
        valid_mask = (x_coords_flat >= 0) & (x_coords_flat < output_width) & \
                     (y_coords_flat >= 0) & (y_coords_flat < output_height)
        x_coords_flat = x_coords_flat[valid_mask]
        y_coords_flat = y_coords_flat[valid_mask]
        glyphs_flat_values = glyphs_flat_values[valid_mask]

        # Place glyphs onto the output tensor
        output_tensor.index_put_((y_coords_flat, x_coords_flat), glyphs_flat_values, accumulate=True)

        # --- Apply Rainbow Ink ---
        #if fft_data is not None:
        #    output_tensor = self.ink_rainbow.apply_fft_rainbow(output_tensor, fft_data)

        return output_tensor

    def extract_pdf_text_positions(pdf_path):
        """
        Extract text positions and font details from a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            list[dict]: List of text details with positions, sizes, and font.
        """
        extracted_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                for char in page.chars:
                    extracted_data.append({
                        "page": page_number + 1,
                        "char": char.get("text"),
                        "x0": char.get("x0"),
                        "y0": char.get("top"),
                        "x1": char.get("x1"),
                        "y1": char.get("bottom"),
                        "size": char.get("size"),
                        "font": char.get("fontname"),
                    })
        return extracted_data

    # Example usage
    if __name__ == "__main__":
        pdf_path = "example.pdf"
        positions = extract_pdf_text_positions(pdf_path)
        for item in positions[:10]:  # Print first 10 entries
            print(item)

    def typeset_from_metrics(self, metrics, page_width=800, page_height=1200):
        """
        Generate a tensor representation from text metrics.

        Args:
            metrics (list[dict]): Layout metrics extracted from HTML.
            page_width (int): Width of the page.
            page_height (int): Height of the page.

        Returns:
            torch.Tensor: A tensor representing the typeset page.
        """
        output_tensor = torch.zeros((page_height, page_width), dtype=torch.uint8)
        for metric in metrics:
            x_start = int(metric["x"])
            y_start = int(metric["y"])
            width = int(metric["width"])
            height = int(metric["height"])

            # Simulate glyph placement with rectangles for simplicity
            output_tensor[y_start:y_start + height, x_start:x_start + width] = 255
        return output_tensor

if __name__ == "__main__":
    # Initialize the GrandPrintingPress
    press = GrandPrintingPress(page_width=800, page_height=1200, margin=50)

    # Example HTML input
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; font-size: 16px; margin: 50px; }
            h1 { font-size: 24px; text-align: center; }
            p { text-align: justify; }
        </style>
    </head>
    <body>
        <h1>The Grand Opening</h1>
        <p>Welcome to the Hot-Rod Printing Press, where technology meets art.</p>
    </body>
    </html>
    """

    # Extract text metrics
    metrics = press.extract_text_metrics(html_content)

    # Render the document using extracted metrics
    document_tensor = press.typeset_from_metrics(metrics)

    # Convert tensor to an image for visualization
    document_image = Image.fromarray(document_tensor.numpy().astype(np.uint8), mode="L")
    document_image.show()
