import torch
import sympy as sp
import numpy as np

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
        self.base_unit = sp.Symbol('count')  # Base unit for tensors
        self.unit_registry = self._initialize_units()

    def _initialize_units(self):
        """
        Define the unit relationships using SymPy for clarity and extensibility.
        
        Returns:
            dict: A registry of unit conversions relative to the base unit.
        """
        mm = sp.Symbol('mm')  # Millimeters
        inch = sp.Symbol('inch')  # Inches
        count = self.base_unit  # Tensor index
        
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
        kernel_size = int(bleed_factor * min(tensor.shape))
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


# Usage Example
if __name__ == "__main__":
    ruler = Ruler()
    paper_maker = PaperMaker()

    # Convert 10 mm to tensor indices
    tensor_coords = ruler.coordinates_to_tensor(10, 10, unit="mm")
    print("Tensor coordinates for 10mm:", tensor_coords)

    # Allocate a Grand Sheet
    try:
        grand_sheet = paper_maker.allocate_grand_sheet(4096, 4096)
        print("Grand Sheet allocated successfully!")
    except MemoryError as e:
        print("Allocation failed:", e)

    # Simulate ink bleed
    ink_tensor = torch.rand((4096, 4096))
    ink_bleed = paper_maker.simulate_ink_bleed(ink_tensor)
    print("Simulated ink bleed applied.")
