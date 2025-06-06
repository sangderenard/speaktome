import torch
import torch.nn.functional as F
from typing import Tuple, Dict
import sympy as sp
import numpy as np


class Ruler:
    """
    **Ruler**
    A versatile unit converter and coordinate translator for all system components.
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize the Ruler with a default DPI (dots per inch).
        
        Args:
            dpi (int): Resolution for conversions (dots per inch).
        """
        self.dpi = dpi
        self.base_unit = sp.Symbol('count')
        self.unit_registry = self._initialize_units()

    def _initialize_units(self):
        """
        Define unit relationships using SymPy for extensibility.
        
        Returns:
            dict: Registry of unit conversions relative to 'count'.
        """
        mm = sp.Symbol('mm')
        inch = sp.Symbol('inch')
        count = self.base_unit
        dpi_conversion = 1 / self.dpi
        return {
            'count': 1,
            'inch': dpi_conversion,
            'mm': dpi_conversion / 25.4,
        }

    def convert(self, value, from_unit, to_unit):
        """
        Convert a value between units.
        
        Args:
            value (float): Value to convert.
            from_unit (str): Source unit.
            to_unit (str): Target unit.
        
        Returns:
            float: Converted value.
        """
        if from_unit not in self.unit_registry or to_unit not in self.unit_registry:
            raise ValueError(f"Unknown units: {from_unit} or {to_unit}")
        return value * self.unit_registry[from_unit] / self.unit_registry[to_unit]

    def translate_coordinates(self, tensor: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Translate coordinates with a scaling factor.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            scale_factor (float): Scaling factor.
        
        Returns:
            torch.Tensor: Translated tensor.
        """
        return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode="bilinear").squeeze()

    def tensor_to_coordinates(self, x, y, resolution=(800, 1200)):
        """
        Convert tensor indices to physical coordinates.
        
        Args:
            x (int): X index.
            y (int): Y index.
            resolution (Tuple[int, int]): Tensor resolution (width, height).
        
        Returns:
            dict: Coordinates in various units (mm, inches, counts).
        """
        width, height = resolution
        coords = {
            "count": (x, y),
            "inch": (x * self.unit_registry['count'] / self.unit_registry['inch'],
                     y * self.unit_registry['count'] / self.unit_registry['inch']),
            "mm": (x * self.unit_registry['count'] / self.unit_registry['mm'],
                   y * self.unit_registry['count'] / self.unit_registry['mm']),
        }
        return coords


class Granulator:
    """
    **Granulator**
    Handles resolution and coordinate translation with advanced kernel definitions.
    """
    
    def __init__(self, base_unit: float):
        """
        Initialize the Granulator with a base unit.
        
        Args:
            base_unit (float): Base unit for grid resolution.
        """
        self.base_unit = base_unit

    def define_kernel(self, relative_size: float, ruler: Ruler) -> torch.Tensor:
        """
        Define a kernel based on relative size and convert it using Ruler.
        
        Args:
            relative_size (float): Relative size of the kernel (e.g., 0.1 for 10% of base unit).
            ruler (Ruler): Ruler instance for unit conversion.
        
        Returns:
            torch.Tensor: Defined kernel tensor.
        """
        # Convert relative size to counts using Ruler
        kernel_size_in_counts = int(relative_size / ruler.unit_registry['count'])
        kernel_size = max(1, kernel_size_in_counts)
        kernel = torch.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        return kernel

    def translate_spatial_region(self, tensor: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Translate spatial regions with a scaling factor.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            scale_factor (float): Scaling factor.
        
        Returns:
            torch.Tensor: Translated tensor.
        """
        return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode="bilinear").squeeze()


class Pigment:
    """
    **Pigment**
    Models the material and optical properties of a single pigment particle.
    """
    
    def __init__(self, scattering: float, absorption: float, refractive_index: float, spectrum: torch.Tensor):
        """
        Initialize a pigment with its optical properties.
        
        Args:
            scattering (float): Scattering coefficient.
            absorption (float): Absorption coefficient.
            refractive_index (float): Refractive index.
            spectrum (torch.Tensor): RGB spectrum as a 3-element tensor.
        """
        self.scattering = scattering
        self.absorption = absorption
        self.refractive_index = refractive_index
        self.spectrum = spectrum  # RGB tensor

    def mix_with(self, other_pigment, ratio: float):
        """
        Mix this pigment with another pigment.
        
        Args:
            other_pigment (Pigment): The pigment to mix with.
            ratio (float): Proportion of this pigment in the mix.
        
        Returns:
            Pigment: A new mixed pigment.
        """
        new_spectrum = (self.spectrum * ratio) + (other_pigment.spectrum * (1 - ratio))
        new_scattering = (self.scattering * ratio) + (other_pigment.scattering * (1 - ratio))
        new_absorption = (self.absorption * ratio) + (other_pigment.absorption * (1 - ratio))
        new_refractive_index = (self.refractive_index * ratio) + (other_pigment.refractive_index * (1 - ratio))
        return Pigment(new_scattering, new_absorption, new_refractive_index, new_spectrum)


class Inkstone:
    """
    **Inkstone (砚)**
    Manages ink gradients, pigments, and transfer properties.
    """
    
    def __init__(self, pigments: Dict[str, Pigment], viscosity: float, surface_tension: float):
        """
        Initialize the inkstone with pigments and properties.
        
        Args:
            pigments (Dict[str, Pigment]): Pigments and their proportions.
            viscosity (float): Ink viscosity.
            surface_tension (float): Ink surface tension.
        """
        self.pigments = pigments
        self.viscosity = viscosity
        self.surface_tension = surface_tension

    def compute_gradient(self, tensor_shape: Tuple[int, int], mode: str = "linear") -> torch.Tensor:
        """
        Compute a global ink gradient for the given tensor shape.
        
        Args:
            tensor_shape (Tuple[int, int]): Dimensions of the gradient tensor.
            mode (str): Gradient mode ("linear", "radial").
        
        Returns:
            torch.Tensor: Gradient tensor.
        """
        height, width = tensor_shape
        gradient = torch.zeros((3, height, width), dtype=torch.float32)

        if mode == "linear":
            for y in range(height):
                gradient[:, y, :] = torch.linspace(0, 1, width).unsqueeze(0).expand(3, -1)

        elif mode == "radial":
            center = torch.tensor([height // 2, width // 2], dtype=torch.float32)
            for y in range(height):
                for x in range(width):
                    distance = torch.norm(torch.tensor([y, x]) - center)
                    gradient[:, y, x] = torch.clamp(distance / (max(height, width) / 2), 0, 1)

        return gradient


class Xuanzhi:
    """
    **Xuanzhi (宣纸)**
    Represents the properties of absorbent solids, including paper.
    """
    
    def __init__(self, fiber_density: float, absorption: float, roughness: float):
        """
        Initialize Xuanzhi properties.
        
        Args:
            fiber_density (float): Density of fibers (kg/m³).
            absorption (float): Absorption coefficient (0-1 range).
            roughness (float): Surface roughness coefficient (0-1 range).
        """
        self.fiber_density = fiber_density
        self.absorption = absorption
        self.roughness = roughness

    def simulate_bleed(self, ink_tensor: torch.Tensor, bleed_factor: float):
        """
        Simulate the effect of ink bleed on this paper.
        
        Args:
            ink_tensor (torch.Tensor): Tensor representing ink application.
            bleed_factor (float): Modifier for bleed intensity (proportional to absorption).
        
        Returns:
            torch.Tensor: Tensor with bleed simulation applied.
        """
        kernel_size = max(1, int(bleed_factor * self.absorption * min(ink_tensor.shape[-2:])))
        kernel = torch.ones((kernel_size, kernel_size)) / kernel_size**2
        padded_tensor = F.pad(ink_tensor, (kernel_size // 2,) * 4, mode="reflect")
        bleed_tensor = F.conv2d(
            padded_tensor.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()
        return torch.clamp(bleed_tensor, 0, 1)

    def generate_great_sheet(self, dimensions: Tuple[int, int]):
        """
        Generate a tensor representing a great sheet.
        
        Args:
            dimensions (Tuple[int, int]): Sheet dimensions (height, width).
        
        Returns:
            torch.Tensor: Great sheet tensor initialized with absorption.
        """
        return torch.ones(dimensions, dtype=torch.float32) * self.absorption


class GrandPress:
    """
    **GrandPress**
    Core logic for applying gradients, masks, and generating sheets.
    """
    
    def __init__(self, ruler: Ruler):
        """
        Initialize GrandPress with a ruler for unit conversions.
        
        Args:
            ruler (Ruler): Ruler instance.
        """
        self.ruler = ruler

    def press(self, gradient: torch.Tensor, mask: torch.Tensor, target_tensor: torch.Tensor):
        """
        Apply a gradient and mask to a target tensor.
        
        Args:
            gradient (torch.Tensor): Gradient tensor.
            mask (torch.Tensor): Mask tensor.
            target_tensor (torch.Tensor): Target tensor.
        
        Returns:
            torch.Tensor: Tensor after application.
        """
        return target_tensor * mask + gradient * (1 - mask)

    def batch_press(self, sheets: list, masks: list, gradients: list):
        """
        Batch process multiple tensors for efficiency.
        
        Args:
            sheets (list): List of target sheets.
            masks (list): List of mask tensors.
            gradients (list): List of gradient tensors.
        
        Returns:
            list: Processed tensors.
        """
        return [self.press(gradient, mask, sheet) for sheet, mask, gradient in zip(sheets, masks, gradients)]


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

    def generate_gradient(self, shape: Tuple[int, int], gradient_type: str = "linear"):
        """
        Generate a gradient tensor for ink application.
        
        Args:
            shape (Tuple[int, int]): Dimensions of the gradient tensor.
            gradient_type (str): Type of gradient ('linear', 'radial', etc.).
        
        Returns:
            torch.Tensor: Gradient tensor.
        """
        if gradient_type == "linear":
            gradient = torch.linspace(0, 1, shape[1]).repeat(shape[0], 1)
        elif gradient_type == "radial":
            height, width = shape
            center = torch.tensor([height // 2, width // 2], dtype=torch.float32)
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
            distance = torch.sqrt((y - center[0])**2 + (x - center[1])**2)
            gradient = torch.clamp(distance / (max(height, width) / 2), 0, 1)
        else:
            raise ValueError(f"Unsupported gradient type: {gradient_type}")
        
        return gradient.unsqueeze(0).repeat(3, 1, 1)  # Repeat for RGB channels

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
        spectrum = self.pigments[pigment].spectrum.view(3, 1, 1)
        return target_tensor * (1 - mask) + spectrum * mask

    def simulate_bleed(self, paper: Xuanzhi, ink_tensor: torch.Tensor, bleed_factor: float = 0.05):
        """
        Simulate ink bleed using the paper's bleed simulation.
        
        Args:
            paper (Xuanzhi): Paper object for bleed simulation.
            ink_tensor (torch.Tensor): Tensor of ink application.
            bleed_factor (float): Proportional factor affecting bleed size.
        
        Returns:
            torch.Tensor: Tensor after simulating ink bleed.
        """
        return paper.simulate_bleed(ink_tensor, bleed_factor)

    def generate_noise(self, shape: Tuple[int, int], noise_type: str = "white"):
        """
        Generate noise with specified spectral characteristics.
        
        Args:
            shape (Tuple[int, int]): Dimensions of the noise tensor.
            noise_type (str): Type of noise ('white', 'pink', 'brown').
        
        Returns:
            torch.Tensor: Noise tensor.
        """
        noise = torch.randn(*shape)
        if noise_type == "pink":
            # Apply 1/f spectral decay
            freqs_y = torch.fft.fftfreq(shape[0]).reshape(-1, 1)
            freqs_x = torch.fft.fftfreq(shape[1]).reshape(1, -1)
            freqs = torch.sqrt(freqs_y**2 + freqs_x**2)
            scaling = torch.where(freqs == 0, torch.tensor(0.0), 1.0 / torch.sqrt(freqs))
            noise = torch.fft.ifft2(torch.fft.fft2(noise) * scaling).real
        elif noise_type == "brown":
            # Apply 1/f^2 spectral decay
            freqs_y = torch.fft.fftfreq(shape[0]).reshape(-1, 1)
            freqs_x = torch.fft.fftfreq(shape[1]).reshape(1, -1)
            freqs = torch.sqrt(freqs_y**2 + freqs_x**2)
            scaling = torch.where(freqs == 0, torch.tensor(0.0), 1.0 / (freqs ** 2))
            noise = torch.fft.ifft2(torch.fft.fft2(noise) * scaling).real
        # Normalize and clamp
        noise = (noise - noise.min()) / (noise.max() - noise.min())
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
        sheet = paper.generate_great_sheet((4096, 4096))
        gradient = self.generate_gradient(sheet.shape, gradient_type)
        inked_tensor = self.apply_ink_to_tensor(sheet, gradient, pigment)
        return self.simulate_bleed(paper, inked_tensor, bleed_factor=paper.absorption)


# Usage Example
if __name__ == "__main__":
    # Initialize Ruler and Granulator
    ruler = Ruler(dpi=300)
    granulator = Granulator(base_unit=1.0)
    inkmaker = Inkmaker(ruler, granulator, viscosity=0.3)

    # Add pigments
    red = Pigment(scattering=0.1, absorption=0.2, refractive_index=1.5, spectrum=torch.tensor([1.0, 0.0, 0.0]))
    blue = Pigment(scattering=0.2, absorption=0.1, refractive_index=1.4, spectrum=torch.tensor([0.0, 0.0, 1.0]))
    inkmaker.add_pigment("red", red)
    inkmaker.add_pigment("blue", blue)

    # Simulate full ink application
    paper = Xuanzhi(fiber_density=0.5, absorption=0.3, roughness=0.2)
    simulated_paper = inkmaker.full_simulation(paper, "red", "linear")
    print("Full simulation completed with shape:", simulated_paper.shape)
