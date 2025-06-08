"""
TheGrandPrintShop: Comprehensive Print Shop Simulation
=====================================================

This script provides a full prototype development template for TheGrandPrintShop, integrating
all necessary classes and components to simulate a sophisticated digital print shop. The system
is designed with modularity, scalability, and future-proofing in mind, facilitating the integration
of advanced features like neural networks, shader pipelines, and dynamic task management.

Dependencies:
- torch
- PIL (Pillow)
- numpy
- PyOpenGL
- threading
- queue

Ensure that the `hotrodprintingpress.py` module is available and contains the `GrandPrintingPress`
class as defined in previous implementations.

Author: Your Name
Date: YYYY-MM-DD
"""

import threading
from queue import Queue
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from OpenGL.GL import *
import time
import itertools

# Import the GrandPrintingPress from the hotrodprintingpress module
try:
    from hotrodprintingpress import GrandPrintingPress
except ImportError:
    # Placeholder for GrandPrintingPress if hotrodprintingpress.py is not available
    class GrandPrintingPress:
        def __init__(self, page_width, page_height, margin):
            self.page_width = page_width
            self.page_height = page_height
            self.margin = margin

        def press(self, material_data, glyph_tensor, gradient_tensor, mask_tensor=None):
            """
            Simulate the pressing operation by combining tensors.
            """
            print("[GrandPrintingPress] Pressing material with glyphs and gradients.")
            combined = material_data + glyph_tensor + gradient_tensor
            if mask_tensor is not None:
                combined *= mask_tensor
            return combined


# -----------------------------------------------------------------------------
# Supporting Classes
# -----------------------------------------------------------------------------

class Pigment:
    """
    **Pigment**
    Represents the properties of a pigment used in ink.
    """
    def __init__(self, scattering: float, absorption: float, refractive_index: float, spectrum: torch.Tensor):
        """
        Initialize the Pigment with its properties.

        Args:
            scattering (float): Scattering coefficient.
            absorption (float): Absorption coefficient.
            refractive_index (float): Refractive index.
            spectrum (torch.Tensor): RGB spectrum tensor.
        """
        self.scattering = scattering
        self.absorption = absorption
        self.refractive_index = refractive_index
        self.spectrum = spectrum  # Expected shape: (3,)


class Ruler:
    """
    **Ruler**
    A versatile unit converter and coordinate translator that defines and manages relationships
    between physical units (e.g., mm, inches, DPI) and tensor indices (base unit: 'count').
    """

    def __init__(self, dpi: int = 300):
        """
        Initialize the Ruler with a default DPI (dots per inch).

        Args:
            dpi (int): Default resolution for conversions (dots per inch).
        """
        self.dpi = dpi
        self.base_unit = 'count'  # Base unit for tensors
        self.unit_registry = self._initialize_units()

    def _initialize_units(self):
        """
        Define the unit relationships using basic factors for clarity and extensibility.

        Returns:
            dict: A registry of unit conversions relative to the base unit.
        """
        mm = 'mm'
        inch = 'inch'
        count = self.base_unit

        dpi_conversion = 1 / self.dpi  # Inches per count
        unit_registry = {
            count: 1,
            inch: dpi_conversion,
            mm: dpi_conversion / 25.4,  # 25.4 mm in an inch
        }
        return unit_registry

    def convert(self, value, from_unit, to_unit):
        """
        Convert a value from one unit to another using the unit registry.

        Args:
            value (float): The value to convert.
            from_unit (str): The source unit.
            to_unit (str): The target unit.

        Returns:
            float: Converted value.
        """
        if from_unit not in self.unit_registry or to_unit not in self.unit_registry:
            raise ValueError(f"Unknown units: {from_unit} or {to_unit}")
        from_factor = self.unit_registry[from_unit]
        to_factor = self.unit_registry[to_unit]
        return value * from_factor / to_factor

    def tensor_to_coordinates(self, x, y, resolution=(800, 1200)):
        """
        Convert tensor indices to physical coordinates (e.g., mm or inches).

        Args:
            x (int): X index in the tensor.
            y (int): Y index in the tensor.
            resolution (tuple): Tensor resolution as (width, height).

        Returns:
            dict: Coordinates in various units (mm, inches, counts).
        """
        width, height = resolution
        coords = {
            "count": (x, y),
            "inch": (x / self.dpi, y / self.dpi),
            "mm": (x / self.dpi * 25.4, y / self.dpi * 25.4),
        }
        return coords

    def coordinates_to_tensor(self, x, y, unit="mm"):
        """
        Convert physical coordinates to tensor indices.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            unit (str): Unit of the input coordinates ('mm', 'inch', 'count').

        Returns:
            tuple: Tensor indices as (x_idx, y_idx).
        """
        if unit not in self.unit_registry:
            raise ValueError(f"Unknown unit: {unit}")
        conversion_factor = self.unit_registry[unit]
        return int(x / conversion_factor), int(y / conversion_factor)


class Granulator:
    """
    **Granulator**
    Handles resolution and coordinate translation.
    """

    def __init__(self, base_unit: float):
        """
        Initialize the granulator with a base unit.

        Args:
            base_unit (float): Base unit for grid resolution.
        """
        self.base_unit = base_unit

    def translate_coordinates(self, tensor: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Translate coordinates with a scaling factor.

        Args:
            tensor (torch.Tensor): Input tensor.
            scale_factor (float): Scaling factor.

        Returns:
            torch.Tensor: Translated tensor.
        """
        print(f"[Granulator] Translating coordinates with scale_factor={scale_factor}.")
        return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False).squeeze()


class Xuanzhi:
    """
    **Xuanzhi (宣纸)**
    Represents the properties of traditional paper.
    """

    def __init__(self, fiber_density: float, absorption: float, roughness: float):
        """
        Initialize Xuanzhi paper properties.

        Args:
            fiber_density (float): Density of fibers.
            absorption (float): Absorption coefficient.
            roughness (float): Surface roughness.
        """
        self.fiber_density = fiber_density
        self.absorption = absorption
        self.roughness = roughness

    def generate_great_sheet(self, dimensions: tuple):
        """
        Generate a tensor representing a great sheet.

        Args:
            dimensions (tuple): Sheet dimensions (C, H, W).

        Returns:
            torch.Tensor: Great sheet tensor.
        """
        print(f"[Xuanzhi] Generating great sheet with dimensions: {dimensions}.")
        return torch.ones(dimensions, dtype=torch.float32) * self.absorption


class PaperMaker:
    """
    **PaperMaker**
    Simulates the physical properties of paper, ink bleed, and ensures tensor allocation
    for Grand Sheets and pages.
    """

    def __init__(self, max_size_mb: int = 500):
        """
        Initialize the PaperMaker with default limits.

        Args:
            max_size_mb (int): Maximum allowed memory per Grand Sheet in megabytes.
        """
        self.max_size_mb = max_size_mb
        self.material_properties = self._default_material_properties()

    def _default_material_properties(self):
        """
        Define the default material properties for the paper.

        Returns:
            dict: Material properties with physical simulation parameters.
        """
        return {
            "roughness": 0.5,
            "specularity": 0.3,
            "absorption": 0.2,
            "scattering": 0.1,
            "thickness_mm": 0.1,
        }

    def simulate_ink_bleed(self, tensor: torch.Tensor, bleed_factor: float = 0.05):
        """
        Simulate ink bleed on a tensor representing a sheet of paper.

        Args:
            tensor (torch.Tensor): Input tensor to apply ink bleed.
            bleed_factor (float): Percentage of ink bleed effect.

        Returns:
            torch.Tensor: Tensor with simulated ink bleed applied.
        """
        kernel_size = max(int(bleed_factor * min(tensor.shape[-2:])), 1)
        kernel = torch.ones((kernel_size, kernel_size)) / kernel_size**2
        padded_tensor = F.pad(tensor, (kernel_size//2,) * 4, mode='reflect')
        bleed_tensor = F.conv2d(
            padded_tensor.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()
        return torch.clamp(bleed_tensor, 0, 1)

    def allocate_grand_sheet(self, width, height, channels=3):
        """
        Attempt to allocate a Grand Sheet tensor.

        Args:
            width (int): Width of the Grand Sheet.
            height (int): Height of the Grand Sheet.
            channels (int): Number of channels (e.g., RGB).

        Returns:
            torch.Tensor: Allocated tensor if successful.

        Raises:
            MemoryError: If allocation exceeds the maximum allowed size.
        """
        size_mb = width * height * channels * 4 / (1024**2)  # Size in MB (float32)
        if size_mb > self.max_size_mb:
            raise MemoryError(f"Grand Sheet exceeds max size ({size_mb:.2f} MB > {self.max_size_mb} MB)")
        return torch.zeros((channels, height, width), dtype=torch.float32)

    def calculate_page_limits(self, width, height, page_width, page_height):
        """
        Calculate the number of pages that can fit within a Grand Sheet.

        Args:
            width (int): Width of the Grand Sheet.
            height (int): Height of the Grand Sheet.
            page_width (int): Width of a single page.
            page_height (int): Height of a single page.

        Returns:
            int: Number of pages per Grand Sheet.
        """
        pages_across = width // page_width
        pages_down = height // page_height
        return pages_across * pages_down


class Inkmaker:
    """
    **Inkmaker**
    Handles ink simulation, pigment blending, and global application of gradients,
    supporting realistic bleed and material interaction at the pixel level.
    """

    def __init__(self, ruler: Ruler, granulator: Granulator, viscosity: float = 0.3):
        """
        Initialize the Inkmaker with its dependencies and properties.

        Args:
            ruler (Ruler): Ruler for unit translations.
            granulator (Granulator): Granulator for grid transformations.
            viscosity (float): Viscosity of the ink, affecting bleed behavior.
        """
        self.ruler = ruler
        self.granulator = granulator
        self.viscosity = viscosity
        self.pigments = {}  # Stores pigments by name

    def add_pigment(self, name: str, pigment: Pigment):
        """
        Add a pigment to the ink library.

        Args:
            name (str): Name of the pigment.
            pigment (Pigment): Pigment object to add.
        """
        self.pigments[name] = pigment
        print(f"[Inkmaker] Pigment '{name}' added.")

    def generate_gradient(self, shape: tuple, gradient_type: str = "linear"):
        """
        Generate a gradient tensor for ink application.

        Args:
            shape (tuple): Dimensions of the gradient tensor (C, H, W).
            gradient_type (str): Gradient mode ("linear", "radial").

        Returns:
            torch.Tensor: Gradient tensor.
        """
        print(f"[Inkmaker] Generating {gradient_type} gradient for shape {shape}.")
        if gradient_type == "linear":
            gradient = torch.linspace(0, 1, steps=shape[2]).unsqueeze(0).repeat(shape[0], shape[1], 1)
        elif gradient_type == "radial":
            center_x = shape[2] / 2
            center_y = shape[1] / 2
            y = torch.linspace(-1, 1, steps=shape[1]).unsqueeze(1).repeat(1, shape[2])
            x = torch.linspace(-1, 1, steps=shape[2]).unsqueeze(0).repeat(shape[1], 1)
            distance = torch.sqrt(x**2 + y**2)
            distance = torch.clamp(distance, 0, 1)
            gradient = distance.unsqueeze(0).repeat(shape[0], 1, 1)
        else:
            raise ValueError(f"Unsupported gradient type: {gradient_type}")
        return gradient

    def apply_ink_to_tensor(self, target_tensor: torch.Tensor, mask: torch.Tensor, pigment: str):
        """
        Apply ink to a tensor using a mask and specified pigment.

        Args:
            target_tensor (torch.Tensor): Tensor representing the target surface.
            mask (torch.Tensor): Mask tensor indicating where ink is applied.
            pigment (str): Name of the pigment to use.

        Returns:
            torch.Tensor: Updated target tensor.
        """
        if pigment not in self.pigments:
            raise ValueError(f"Pigment '{pigment}' not found in library.")
        
        # Get pigment spectrum
        spectrum = self.pigments[pigment].spectrum.unsqueeze(-1).unsqueeze(-1)
        print(f"[Inkmaker] Applying pigment '{pigment}' to tensor.")
        return target_tensor * (1 - mask) + spectrum * mask

    def simulate_bleed(self, ink_tensor: torch.Tensor, bleed_factor: float = 0.05):
        """
        Simulate ink bleed using kernel-based convolution.

        Args:
            ink_tensor (torch.Tensor): Tensor of ink application.
            bleed_factor (float): Proportional factor affecting bleed size.

        Returns:
            torch.Tensor: Tensor after simulating ink bleed.
        """
        print(f"[Inkmaker] Simulating bleed with bleed_factor={bleed_factor}.")
        kernel_size = max(int(bleed_factor * min(ink_tensor.shape[-2:])), 1)
        kernel = torch.ones((kernel_size, kernel_size)) / kernel_size**2
        padded_tensor = F.pad(ink_tensor, (kernel_size//2,) * 4, mode="reflect")
        bleed_tensor = F.conv2d(
            padded_tensor.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()
        return torch.clamp(bleed_tensor, 0, 1)

    def generate_noise(self, shape: tuple, noise_type: str = "white"):
        """
        Generate noise with specified spectral characteristics.

        Args:
            shape (tuple): Dimensions of the noise tensor (C, H, W).
            noise_type (str): Type of noise ('white', 'pink', 'brown').

        Returns:
            torch.Tensor: Noise tensor.
        """
        print(f"[Inkmaker] Generating {noise_type} noise for shape {shape}.")
        if noise_type == "white":
            noise = torch.randn(*shape)
        elif noise_type == "pink":
            # Apply 1/f spectral decay
            freqs = torch.fft.fftfreq(shape[2]).unsqueeze(0).repeat(shape[1], 1)
            scaling = torch.abs(freqs).clamp(min=1e-6) ** -0.5
            noise = torch.fft.ifft2(torch.fft.fft2(torch.randn(shape)) * scaling).real
        elif noise_type == "brown":
            # Apply 1/f^2 spectral decay
            freqs = torch.fft.fftfreq(shape[2]).unsqueeze(0).repeat(shape[1], 1)
            scaling = torch.abs(freqs).clamp(min=1e-6) ** -1
            noise = torch.fft.ifft2(torch.fft.fft2(torch.randn(shape)) * scaling).real
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return torch.clamp(noise, 0, 1)

    def full_simulation(self, paper: Xuanzhi, pigment: str, gradient_type: str = "radial"):
        """
        Simulate the complete application of ink to paper.

        Args:
            paper (Xuanzhi): Paper object for material properties.
            pigment (str): Name of the pigment to use.
            gradient_type (str): Type of gradient to apply.

        Returns:
            torch.Tensor: Simulated paper after ink application.
        """
        print(f"[Inkmaker] Starting full simulation with pigment '{pigment}' and gradient_type '{gradient_type}'.")
        sheet = paper.generate_great_sheet((3, 4096, 4096))
        gradient = self.generate_gradient(sheet.shape, gradient_type)
        mask = self.generate_noise(sheet.shape, noise_type="pink")
        inked_tensor = self.apply_ink_to_tensor(sheet, mask, pigment)
        return self.simulate_bleed(inked_tensor, bleed_factor=paper.absorption)


# -----------------------------------------------------------------------------
# Coordinator Classes
# -----------------------------------------------------------------------------

class ToolCoordinator:
    """
    Manages and provides access to tools (e.g., kernels, glyph renderers, gradients).
    """
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, tool):
        """
        Register a new tool.

        Args:
            name (str): Name of the tool.
            tool (Tool): Instance of a Tool subclass.
        """
        self.tools[name] = tool

    def get_tool(self, name: str):
        """
        Retrieve a registered tool by name.

        Args:
            name (str): Name of the tool.

        Returns:
            Tool: The requested tool instance.
        """
        return self.tools.get(name)

    def apply_tool(self, name: str, *args, **kwargs):
        """
        Apply a registered tool to the given arguments.

        Args:
            name (str): Name of the tool to apply.
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Result of the tool's operation.
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        return tool.execute(*args, **kwargs)


class MaterialCoordinator:
    """
    Manages materials, memory, and resource allocation.
    """
    def __init__(self, max_cache_size: int = 10):
        self.cache = []
        self.max_cache_size = max_cache_size

    def allocate_material(self, shape: tuple, dtype=torch.float32):
        """
        Allocate a new material tensor.

        Args:
            shape (tuple): Shape of the tensor (C, H, W).
            dtype (torch.dtype): Data type of the tensor.

        Returns:
            Material: Allocated material instance.
        """
        size_mb = torch.tensor(shape).prod().item() * torch.finfo(dtype).bits / 8 / (1024**2)
        if size_mb > self.max_cache_size:
            raise MemoryError(f"Material size {size_mb:.2f} MB exceeds max cache size {self.max_cache_size} MB.")
        data = torch.zeros(shape, dtype=dtype)
        material = Material(data=data)
        self.cache.append(material)
        if len(self.cache) > self.max_cache_size:
            self.cache.pop(0)  # Evict oldest material if cache exceeds size
        return material

    def transform_format(self, material: 'Material', target_format: str):
        """
        Transform the format of a material (e.g., tensor to OpenGL texture).

        Args:
            material (Material): The material instance.
            target_format (str): Target format (e.g., 'OpenGL').

        Returns:
            Converted material.
        """
        if target_format == "OpenGL":
            digitizer = DigitizerInstance()
            texture_id = digitizer.tensor_to_texture(material.data)
            return texture_id
        raise ValueError(f"Unsupported target format: {target_format}")


class MachineCoordinator:
    """
    Manages machines (workflows) that combine tools and materials.
    """
    def __init__(self, tool_coordinator: ToolCoordinator, material_coordinator: MaterialCoordinator):
        self.tool_coordinator = tool_coordinator
        self.material_coordinator = material_coordinator
        self.machines = {}

    def register_machine(self, name: str, machine):
        """
        Register a new machine.

        Args:
            name (str): Name of the machine.
            machine (Machine): Instance of a Machine subclass.
        """
        self.machines[name] = machine

    def get_machine(self, name: str):
        """
        Retrieve a registered machine by name.

        Args:
            name (str): Name of the machine.

        Returns:
            Machine: The requested machine instance.
        """
        return self.machines.get(name)

    def run_machine(self, name: str, material: 'Material', *args, **kwargs):
        """
        Run a registered machine on the given material.

        Args:
            name (str): Name of the machine to run.
            material (Material): The material to process.
            *args: Positional arguments for the machine.
            **kwargs: Keyword arguments for the machine.

        Returns:
            Result of the machine's operation.
        """
        machine = self.get_machine(name)
        if not machine:
            raise ValueError(f"Machine '{name}' not found.")
        return machine.operate(material, *args, **kwargs)


# -----------------------------------------------------------------------------
# Tool Classes
# -----------------------------------------------------------------------------

class Tool:
    """
    Base class for tools in the shop. Tools are used by machines or workers
    to perform specific tasks.
    """
    def __init__(self, name: str):
        self.name = name

    def execute(self, *args, **kwargs):
        """
        Execute the tool's operation. Must be implemented by subclasses.
        """
        raise NotImplementedError("Tool 'execute' method must be implemented by subclasses.")


class GlyphAssembler(Tool):
    def __init__(self, name: str):
        super().__init__(name)

    def execute(self, text: str):
        """
        Assembles glyphs into a tensor.

        Args:
            text (str): Text to render.

        Returns:
            torch.Tensor: Glyph tensor.
        """
        print(f"[GlyphAssembler] Assembling glyphs for text: '{text}'")
        # Placeholder glyph assembly logic
        # In a real implementation, this would generate a tensor representing the glyphs
        glyph_tensor = torch.rand(3, 100, 100)  # Example tensor
        return glyph_tensor


class GradientMaker(Tool):
    def __init__(self, name: str):
        super().__init__(name)

    def execute(self, dimensions: tuple):
        """
        Creates a gradient tensor.

        Args:
            dimensions (tuple): Dimensions of the gradient tensor (C, H, W).

        Returns:
            torch.Tensor: Gradient tensor.
        """
        print(f"[GradientMaker] Creating gradient for dimensions: {dimensions}")
        # Placeholder gradient creation logic
        gradient = torch.linspace(0, 1, steps=dimensions[2]).unsqueeze(0).repeat(dimensions[0], dimensions[1], 1)
        return gradient


# -----------------------------------------------------------------------------
# Machine Classes
# -----------------------------------------------------------------------------

class Machine:
    """
    Base class for machines in the shop.
    Machines handle tasks that are too complex or precise for workers alone.
    """
    def __init__(self, name: str, tools: list):
        self.name = name
        self.tools = tools

    def operate(self, material: Material, *args, **kwargs):
        """
        Perform the machine's operation on the given material.
        Must be implemented by subclasses.

        Args:
            material (Material): The material to process.

        Returns:
            Material: The processed material.
        """
        raise NotImplementedError("Machine 'operate' method must be implemented by subclasses.")


class GrandPress(Machine):
    def __init__(self, name: str, tools: list):
        super().__init__(name, tools)
        self.grand_press_instance = GrandPrintingPress(page_width=800, page_height=1200, margin=50)

    def operate(self, material: Material, glyph_tensor: torch.Tensor, gradient_tensor: torch.Tensor, mask_tensor: torch.Tensor = None):
        """
        Apply glyphs, gradients, and masks to the grand sheet.

        Args:
            material (Material): The grand sheet to process.
            glyph_tensor (torch.Tensor): Tensor representing the glyphs.
            gradient_tensor (torch.Tensor): Tensor representing the gradient.
            mask_tensor (torch.Tensor, optional): Tensor representing the mask.

        Returns:
            Material: The processed grand sheet.
        """
        print(f"[GrandPress] Operating on material with glyphs and gradients.")
        processed_tensor = self.grand_press_instance.press(material.data, glyph_tensor, gradient_tensor, mask_tensor)
        material.data = processed_tensor
        return material


class Digitizer(Machine):
    def __init__(self, name: str, tools: list):
        super().__init__(name, tools)
        self.digitizer_instance = DigitizerInstance()

    def operate(self, material: Material):
        """
        Converts tensors into OpenGL textures.

        Args:
            material (Material): The material tensor to digitize.

        Returns:
            int: OpenGL texture ID.
        """
        print(f"[Digitizer] Converting tensor to OpenGL texture.")
        texture_id = self.digitizer_instance.tensor_to_texture(material.data)
        return texture_id


# -----------------------------------------------------------------------------
# Digitizer Class
# -----------------------------------------------------------------------------

class DigitizerInstance:
    """
    Digitizer converts processed tensors into OpenGL textures, preparing them for shader pipeline integrations.
    """
    def __init__(self):
        """Initialize the Digitizer."""
        self.textures = []  # List to keep track of generated texture IDs

    def tensor_to_texture(self, tensor: torch.Tensor) -> int:
        """
        Convert a tensor to an OpenGL texture.

        Args:
            tensor (torch.Tensor): The tensor to convert (expected shape: CxHxW).

        Returns:
            int: OpenGL texture ID.
        """
        print("[Digitizer] Converting tensor to OpenGL texture.")
        # Ensure tensor is on CPU and in byte format
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).byte()

        # Add alpha channel if necessary
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(4, 1, 1)  # Convert to RGBA
            tensor[3, :, :] = 255  # Set alpha to opaque
        elif tensor.dim() == 3 and tensor.size(0) == 3:
            tensor = torch.cat([tensor, torch.full((1, tensor.size(1), tensor.size(2)), 255, dtype=torch.uint8)], dim=0)

        # Convert to numpy array in HWC format
        texture_data = tensor.permute(1, 2, 0).numpy()

        # Generate OpenGL texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            texture_data.shape[1], texture_data.shape[0],
            0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.textures.append(texture)
        print(f"[Digitizer] Texture ID {texture} generated.")
        return texture

    def cleanup(self):
        """Delete all generated textures."""
        if self.textures:
            glDeleteTextures(self.textures)
            print("[Digitizer] Cleaned up all textures.")
            self.textures = []


# -----------------------------------------------------------------------------
# Worker Classes
# -----------------------------------------------------------------------------

class Worker:
    """
    Base class for workers in the shop. Workers use machines and tools to complete tasks.
    """
    def __init__(self, name: str, machines: list, tools: list, coordinator_trio):
        self.name = name
        self.machines = machines
        self.tools = tools
        self.coordinator_trio = coordinator_trio  # Tuple of (ToolCoordinator, MaterialCoordinator, MachineCoordinator)

    def perform_task(self):
        """
        Perform a task. Must be implemented by subclasses.
        """
        raise NotImplementedError("Worker 'perform_task' method must be implemented by subclasses.")


class InkWorker(Worker):
    def __init__(self, name: str, machines: list, tools: list, coordinator_trio):
        super().__init__(name, machines, tools, coordinator_trio)

    def perform_task(self):
        print(f"[{self.name}] Preparing and applying ink.")
        tool_coordinator, material_coordinator, machine_coordinator = self.coordinator_trio
        # Apply gradient using GradientMaker tool
        gradient = tool_coordinator.apply_tool("Gradient Maker", dimensions=(3, 500, 500))
        # Allocate material for ink application
        ink_material = material_coordinator.allocate_material((3, 500, 500))
        ink_material.data = gradient
        # Operate Grand Press to apply ink
        grand_press = machine_coordinator.get_machine("Grand Press")
        grand_press.operate(
            material=ink_material,
            glyph_tensor=torch.rand(3, 100, 100),
            gradient_tensor=gradient
        )
        print(f"[{self.name}] Ink applied.")


class PaperWorker(Worker):
    def __init__(self, name: str, machines: list, tools: list, coordinator_trio):
        super().__init__(name, machines, tools, coordinator_trio)

    def perform_task(self):
        print(f"[{self.name}] Preparing paper.")
        tool_coordinator, material_coordinator, machine_coordinator = self.coordinator_trio
        # Assemble glyphs using GlyphAssembler tool
        glyphs = tool_coordinator.apply_tool("Glyph Assembler", text="Hello, World!")
        # Allocate material for paper
        paper_material = material_coordinator.allocate_material((3, 500, 500))
        paper_material.data = glyphs
        print(f"[{self.name}] Paper prepared.")


class PressOperator(Worker):
    def __init__(self, name: str, machines: list, tools: list, coordinator_trio):
        super().__init__(name, machines, tools, coordinator_trio)

    def perform_task(self):
        print(f"[{self.name}] Operating the Grand Press.")
        tool_coordinator, material_coordinator, machine_coordinator = self.coordinator_trio
        # Allocate material for pressing
        paper_material = material_coordinator.allocate_material((3, 500, 500))
        paper_material.data = torch.rand(3, 500, 500)  # Placeholder data
        # Apply pressing operation
        grand_press = machine_coordinator.get_machine("Grand Press")
        processed_material = grand_press.operate(
            material=paper_material,
            glyph_tensor=torch.rand(3, 100, 100),
            gradient_tensor=torch.rand(3, 500, 500)
        )
        print(f"[{self.name}] Grand Press operation completed.")


# -----------------------------------------------------------------------------
# Shop Class
# -----------------------------------------------------------------------------

class Shop:
    """
    The Shop class is the top-level manager for all aspects of the system. It
    coordinates workers, materials, tools, and machines while maintaining records
    and in-house research capabilities.
    """
    def __init__(self, name: str, max_cache_size: int = 10):
        self.name = name
        self.tools = {}
        self.materials = {}
        self.machines = {}
        self.workers = {}
        self.records = []
        self.research = {}

        # Initialize Coordinators
        self.tool_coordinator = ToolCoordinator()
        self.material_coordinator = MaterialCoordinator(max_cache_size=max_cache_size)
        self.machine_coordinator = MachineCoordinator(self.tool_coordinator, self.material_coordinator)

    # --- Core Functionality ---

    def add_tool(self, tool_name: str, tool: Tool):
        self.tool_coordinator.register_tool(tool_name, tool)
        self.tools[tool_name] = tool

    def add_material(self, material_name: str, material: Material):
        self.materials[material_name] = material

    def add_machine(self, machine_name: str, machine: Machine):
        self.machine_coordinator.register_machine(machine_name, machine)
        self.machines[machine_name] = machine

    def add_worker(self, worker_name: str, worker: Worker):
        self.workers[worker_name] = worker

    def log(self, record: str):
        self.records.append(record)
        print(f"[Shop] Log: {record}")

    def conduct_research(self, topic: str, method):
        """
        Conduct in-house research and store findings.
        """
        self.research[topic] = method()
        print(f"[Shop] Conducted research on: {topic}")

    def run(self):
        """
        Execute all workers and machines in the shop.
        """
        print(f"[Shop] Running shop operations.")
        threads = []
        for worker in self.workers.values():
            thread = threading.Thread(target=worker.perform_task, daemon=True)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    # --- Display Functions ---

    def display_inventory(self):
        print(f"\nShop: {self.name}")
        print("Tools:", list(self.tools.keys()))
        print("Materials:", list(self.materials.keys()))
        print("Machines:", list(self.machines.keys()))
        print("Workers:", list(self.workers.keys()))
        print("Records:", self.records)
        print("Research Topics:", list(self.research.keys()), "\n")


# -----------------------------------------------------------------------------
# Demonstration
# -----------------------------------------------------------------------------

def print_shop_demo():
    """
    Demonstration of the print shop workflow.
    """
    # Initialize supporting components
    ruler = Ruler(dpi=300)
    granulator = Granulator(base_unit=1.0)
    papermaker = PaperMaker(max_size_mb=500)
    inkmaker = Inkmaker(ruler, granulator, viscosity=0.3)

    # Define pigments
    black_pigment = Pigment(
        scattering=0.2,
        absorption=0.8,
        refractive_index=1.5,
        spectrum=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # Black pigment
    )
    inkmaker.add_pigment("black", black_pigment)

    # Initialize the Grand Printing Press
    press = GrandPrintingPress(page_width=800, page_height=1200, margin=50)

    # Initialize Shop
    shop = Shop("The Grand Print Shop", max_cache_size=5)

    # Add tools
    glyph_assembler = GlyphAssembler("Glyph Assembler")
    gradient_maker = GradientMaker("Gradient Maker")
    shop.add_tool("Glyph Assembler", glyph_assembler)
    shop.add_tool("Gradient Maker", gradient_maker)

    # Add materials
    standard_paper = Paper("Standard Paper", {"weight": "80gsm", "color": "white"})
    black_ink = Ink("Black Ink", {"viscosity": "medium", "color": "black"})
    shop.add_material("Standard Paper", standard_paper)
    shop.add_material("Black Ink", black_ink)

    # Add machines
    grand_press = GrandPress("Grand Press", ["Glyph Assembler", "Gradient Maker"])
    digitizer = Digitizer("Digitizer", [])
    shop.add_machine("Grand Press", grand_press)
    shop.add_machine("Digitizer", digitizer)

    # Add workers
    coordinator_trio = (shop.tool_coordinator, shop.material_coordinator, shop.machine_coordinator)
    ink_worker = InkWorker("Alice", ["Grand Press"], ["Gradient Maker"], coordinator_trio)
    paper_worker = PaperWorker("Bob", [], ["Glyph Assembler"], coordinator_trio)
    press_operator = PressOperator("Charlie", ["Grand Press"], [], coordinator_trio)
    shop.add_worker("Alice", ink_worker)
    shop.add_worker("Bob", paper_worker)
    shop.add_worker("Charlie", press_operator)

    # Conduct some research
    shop.conduct_research("Ink Diffusion", lambda: "Simulated ink diffusion parameters.")

    # Display inventory before running
    shop.display_inventory()

    # Run the shop
    shop.run()

    # Display final logs and research
    shop.display_inventory()

    # Cleanup Digitizer textures
    digitizer_instance = shop.machine_coordinator.get_machine("Digitizer").digitizer_instance
    digitizer_instance.cleanup()

class Material:
    """
    Represents a material with data and metadata for processing in the print shop.
    """
    def __init__(self, data: torch.Tensor, metadata: dict = None):
        self.data = data
        self.metadata = metadata or {}

    def transform(self, transformation: callable):
        """
        Apply a transformation to the material's data.

        Args:
            transformation (callable): Transformation function.

        Returns:
            None
        """
        self.data = transformation(self.data)



# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print_shop_demo()
