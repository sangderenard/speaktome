import torch
import numpy as np
from PIL import Image
import math
import sys

class Noise:
    WHITE_NOISE = {"mean": 0.0, "std": 1.0}
    PINK_NOISE = {"mean": 0.0, "std": 0.5}
    BROWN_NOISE = {"mean": 0.0, "std": 0.3}
    BLUE_NOISE = {"mean": 0.0, "std": 0.7}
    VIOLET_NOISE = {"mean": 0.0, "std": 0.9}

    LASER_WAVELENGTHS = {
        "cheap_pen_light": {"wavelength": 650, "std": 2},  # nm (red), std deviation approximate
        "green_pointer": {"wavelength": 532, "std": 1},    # nm (green), std deviation approximate
        "blue_pointer": {"wavelength": 450, "std": 1},     # nm (blue), std deviation approximate
        "industrial_red": {"wavelength": 780, "std": 5},   # nm (near infrared), std deviation approximate
        "scientific_uv": {"wavelength": 355, "std": 0.5},  # nm (ultraviolet), std deviation approximate
        "fiber_laser": {"wavelength": 1064, "std": 3},     # nm (infrared), std deviation approximate
        "custom_lab_machine": {"wavelength": 1053, "std": 0.1} # nm (infrared, e.g., Lawrence Livermore), std deviation approximate
    }

    LIGHT_SPECTRA = {
        "sunlight": [
            {"wavelength": 400, "std": 10, "intensity": 0.3},  # Violet
            {"wavelength": 450, "std": 10, "intensity": 0.5},  # Blue
            {"wavelength": 500, "std": 10, "intensity": 0.8},  # Cyan
            {"wavelength": 550, "std": 10, "intensity": 1.0},  # Green
            {"wavelength": 600, "std": 10, "intensity": 0.9},  # Yellow
            {"wavelength": 650, "std": 10, "intensity": 0.7},  # Red
        ],
        "blue_sky_daylight": [
            {"wavelength": 450, "std": 15, "intensity": 1.0},  # Blue
            {"wavelength": 500, "std": 15, "intensity": 0.7},  # Cyan
            {"wavelength": 550, "std": 15, "intensity": 0.5},  # Green
        ],
        "tungsten": [
            {"wavelength": 600, "std": 20, "intensity": 1.0},  # Yellow-Red
            {"wavelength": 650, "std": 20, "intensity": 0.8},  # Red
            {"wavelength": 700, "std": 20, "intensity": 0.5},  # Deep Red
        ],
        "camera_flash": [
            {"wavelength": 450, "std": 10, "intensity": 0.7},  # Blue
            {"wavelength": 550, "std": 10, "intensity": 1.0},  # Green
            {"wavelength": 600, "std": 10, "intensity": 0.8},  # Yellow
        ],
        "phosphor_emitter": [
            {"wavelength": 430, "std": 10, "intensity": 0.6},  # Violet-Blue
            {"wavelength": 545, "std": 10, "intensity": 1.0},  # Green
            {"wavelength": 610, "std": 10, "intensity": 0.7},  # Orange-Red
        ],
        "moonlight": [
            {"wavelength": 450, "std": 10, "intensity": 0.6},  # Blue
            {"wavelength": 500, "std": 10, "intensity": 0.4},  # Cyan
            {"wavelength": 550, "std": 10, "intensity": 0.3},  # Green
        ],
        "sodium_street_lamp": [
            {"wavelength": 589, "std": 5, "intensity": 1.0},  # Yellow
        ],
        "halogen": [
            {"wavelength": 500, "std": 15, "intensity": 0.6},  # Cyan
            {"wavelength": 600, "std": 15, "intensity": 1.0},  # Yellow-Red
            {"wavelength": 650, "std": 15, "intensity": 0.8},  # Red
        ]
    }

    def __init__(self, mean=0.0, std=1.0, distribution="normal", histogram_profile=None):
        """
        Initializes the Noise class with the specified parameters or defaults.

        Args:
            mean (float): Mean of the normal distribution (used when distribution='normal').
            std (float): Standard deviation of the normal distribution (used when distribution='normal').
            distribution (str): Type of distribution to use ('normal', 'uniform', 'histogram').
            histogram_profile (np.ndarray): Histogram values for empirical noise generation (optional).
        """
        self.mean = mean
        self.std = std
        self.distribution = distribution
        self.histogram_profile = histogram_profile

        # Validate distribution type
        if self.distribution not in ["normal", "uniform", "histogram"]:
            raise ValueError("Unsupported distribution type. Use 'normal', 'uniform', or 'histogram'.")
        
        # Validate histogram profile
        if self.distribution == "histogram" and histogram_profile is None:
            raise ValueError("Histogram profile must be provided when using 'histogram' distribution.")

    @classmethod
    def from_named_spectrum(cls, spectrum_name, distribution="normal"):
        """
        Creates a Noise instance from a predefined noise spectrum.

        Args:
            spectrum_name (str): The name of the noise spectrum ('white', 'pink', 'brown', 'blue', 'violet').
            distribution (str): Type of distribution to use ('normal', 'uniform', 'histogram').

        Returns:
            Noise: A Noise instance with parameters corresponding to the specified spectrum.
        """
        spectrum_mapping = {
            "white": cls.WHITE_NOISE,
            "pink": cls.PINK_NOISE,
            "brown": cls.BROWN_NOISE,
            "blue": cls.BLUE_NOISE,
            "violet": cls.VIOLET_NOISE
        }
        if spectrum_name not in spectrum_mapping:
            raise ValueError("Unsupported spectrum name. Use 'white', 'pink', 'brown', 'blue', or 'violet'.")
        params = spectrum_mapping[spectrum_name]
        return cls(mean=params["mean"], std=params["std"], distribution=distribution)

    def generate(self, size):
        """
        Generates a noise tensor of the requested size based on the initialized parameters.

        Args:
            size (tuple): Shape of the tensor to be generated.

        Returns:
            torch.Tensor: A tensor filled with noise of the specified size.
        """
        if self.distribution == "normal":
            return self._generate_normal(size)
        elif self.distribution == "uniform":
            return self._generate_uniform(size)
        elif self.distribution == "histogram":
            return self._generate_histogram(size)

    def _generate_normal(self, size):
        """
        Generates a tensor of Gaussian noise with the specified mean and standard deviation.

        Args:
            size (tuple): Shape of the tensor to be generated.

        Returns:
            torch.Tensor: A tensor filled with Gaussian noise.
        """
        return torch.normal(mean=self.mean, std=self.std, size=size)

    def _generate_uniform(self, size):
        """
        Generates a tensor of noise from a uniform distribution between 0 and 1.

        Args:
            size (tuple): Shape of the tensor to be generated.

        Returns:
            torch.Tensor: A tensor filled with uniformly distributed noise.
        """
        return torch.rand(size)

    def _generate_histogram(self, size):
        """
        Generates a tensor of noise based on an empirical histogram profile.

        Args:
            size (tuple): Shape of the tensor to be generated.

        Returns:
            torch.Tensor: A tensor filled with noise following the histogram profile.
        """
        # Generate random indices based on the histogram profile probabilities
        histogram_probabilities = self.histogram_profile / np.sum(self.histogram_profile)
        values = np.arange(len(histogram_probabilities))
        sampled_indices = np.random.choice(values, size=int(np.prod(size)), p=histogram_probabilities)
        sampled_tensor = torch.tensor(sampled_indices, dtype=torch.float32)
        return sampled_tensor.view(size)

    def get_closest_color_hex(self, spectrum_name):
        """
        Returns the web-safe hex color code closest to the given light spectrum.

        Args:
            spectrum_name (str): The name of the light spectrum.

        Returns:
            str: The web-safe hex color code representing the spectrum.
        """
        if spectrum_name not in self.LIGHT_SPECTRA:
            raise ValueError("Unsupported spectrum name. Available options: {}".format(", ".join(self.LIGHT_SPECTRA.keys())))
        
        spectrum = self.LIGHT_SPECTRA[spectrum_name]
        total_intensity = sum([band["intensity"] for band in spectrum])
        
        # Calculate weighted RGB values based on intensity and wavelength contributions
        r, g, b = 0, 0, 0
        for band in spectrum:
            wavelength = band["wavelength"]
            intensity = band["intensity"] / total_intensity
            rgb = self.wavelength_to_rgb(wavelength)
            r += rgb[0] * intensity
            g += rgb[1] * intensity
            b += rgb[2] * intensity
        
        # Clamp values between 0 and 1
        r = min(max(r, 0), 1)
        g = min(max(g, 0), 1)
        b = min(max(b, 0), 1)
        
        # Convert to hex
        hex_color = '#{0:02x}{1:02x}{2:02x}'.format(int(r*255), int(g*255), int(b*255))
        return hex_color

    def wavelength_to_rgb(self, wavelength):
        """
        Converts a wavelength in the visible spectrum to an approximate RGB value.

        Args:
            wavelength (float): Wavelength in nanometers.

        Returns:
            tuple: RGB values in the range [0, 1].
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

        # Adjust intensity for the edges of the visible spectrum
        if 380 <= wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif 645 <= wavelength <= 780:
            factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
        else:
            factor = 1.0

        r = (r * factor)
        g = (g * factor)
        b = (b * factor)

        return (r, g, b)

def closest_16bit_color(r, g, b):
    """
    Convert 8-bit RGB to closest 16-bit RGB565 color.

    Args:
        r (int): Red channel (0-255)
        g (int): Green channel (0-255)
        b (int): Blue channel (0-255)

    Returns:
        tuple: RGB565 as (R, G, B) each scaled back to 8-bit for display.
    """
    # Convert to 5 bits for red, 6 bits for green, 5 bits for blue
    r5 = (r * 31) // 255
    g6 = (g * 63) // 255
    b5 = (b * 31) // 255
    # Convert back to 8 bits
    r8 = (r5 * 255) // 31
    g8 = (g6 * 255) // 63
    b8 = (b5 * 255) // 31
    return (r8, g8, b8)

def hex_to_rgb(hex_color):
    """
    Convert hex color to RGB tuple.

    Args:
        hex_color (str): Hex color string, e.g., '#ff0000'

    Returns:
        tuple: (R, G, B) each in range [0, 255]
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 digits.")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b)

def apply_alpha_blending(bg_color, fg_color, alpha):
    """
    Apply alpha blending between background and foreground colors.

    Args:
        bg_color (tuple): Background color (R, G, B)
        fg_color (tuple): Foreground color (R, G, B)
        alpha (float): Alpha value [0.0, 1.0]

    Returns:
        tuple: Blended color (R, G, B)
    """
    blended = tuple(
        int(fg_c * alpha + bg_c * (1 - alpha))
        for fg_c, bg_c in zip(fg_color, bg_color)
    )
    return blended

def generate_light_simulation(noise_instance, spectrum_name, image_size=(512, 512), num_photons=10000, std=10.0, luminance_scale=100.0, output_filename="light_simulation.png"):
    """
    Generate a light simulation image based on the specified parameters.

    Args:
        noise_instance (Noise): Instance of the Noise class.
        spectrum_name (str): Name of the light spectrum.
        image_size (tuple): Size of the output image (width, height).
        num_photons (int): Number of photonic units to simulate.
        std (float): Standard deviation for beam tightness.
        luminance_scale (float): Percentage to scale luminance [0, 100].
        output_filename (str): Filename for the output PNG image.
    """
    width, height = image_size
    # Initialize image with black background
    image = Image.new("RGB", (width, height), (0, 0, 0))
    pixels = image.load()

    # Generate photonic units positions using Gaussian distribution centered at the origin
    # Map the unit square [-1, 1] to image coordinates
    positions = noise_instance.generate((num_photons, 2)).numpy()
    # Normalize positions to have mean 0 and std deviation as specified
    positions = (positions - np.mean(positions, axis=0)) / np.std(positions, axis=0) * std
    # Scale positions to image coordinates
    positions = (positions + std) / (2 * std) * np.array([width, height])

    # Generate colors based on the spectrum
    hex_color = noise_instance.get_closest_color_hex(spectrum_name)
    fg_color = hex_to_rgb(hex_color)

    # Convert to closest 16-bit color
    fg_color_16bit = closest_16bit_color(*fg_color)

    # Luminance scaling
    luminance_scale = max(0.0, min(luminance_scale, 100.0)) / 100.0

    bg_color = (0, 0, 0)  # Black background

    for pos in positions:
        x, y = pos
        ix, iy = int(x), int(y)
        if 0 <= ix < width and 0 <= iy < height:
            # Apply Gaussian spread based on distance from center
            # Calculate distance from center
            dx = x - width / 2
            dy = y - height / 2
            distance = math.sqrt(dx*dx + dy*dy)
            # Calculate alpha based on distance and std
            alpha = math.exp(-(distance**2) / (2 * std**2))
            alpha = min(max(alpha * luminance_scale, 0.0), 1.0)
            # Apply alpha blending
            current_color = pixels[ix, iy]
            blended_color = apply_alpha_blending(current_color, fg_color_16bit, alpha)
            pixels[ix, iy] = blended_color

    # Save the image
    image.save(output_filename, format="PNG")
    print(f"Image saved as {output_filename}")

def main_menu():
    """
    Display the text-based menu and handle user inputs to generate light simulations.
    """
    noise_instance = Noise()

    while True:
        print("\n=== Light Simulation Menu ===")
        print("1. Select Light Spectrum")
        print("2. Configure Beam Tightness (std)")
        print("3. Configure Luminance Scale (%)")
        print("4. Configure Number of Photons")
        print("5. Generate Simulation")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            print("\nAvailable Light Spectra:")
            for idx, key in enumerate(Noise.LIGHT_SPECTRA.keys(), start=1):
                print(f"{idx}. {key}")
            spectrum_choice = input("Select a spectrum by number: ")
            try:
                spectrum_idx = int(spectrum_choice) - 1
                spectrum_name = list(Noise.LIGHT_SPECTRA.keys())[spectrum_idx]
                selected_spectrum = spectrum_name
                print(f"Selected spectrum: {selected_spectrum}")
            except (IndexError, ValueError):
                print("Invalid selection. Please try again.")
                continue

        elif choice == '2':
            std_input = input("Enter standard deviation for beam tightness (e.g., 10.0): ")
            try:
                std = float(std_input)
                if std <= 0:
                    raise ValueError
                beam_std = std
                print(f"Beam tightness std set to: {beam_std}")
            except ValueError:
                print("Invalid input. Please enter a positive number.")
                continue

        elif choice == '3':
            luminance_input = input("Enter luminance scale percentage (0-100, default 100): ")
            try:
                luminance_scale = float(luminance_input)
                if not (0 <= luminance_scale <= 100):
                    raise ValueError
                print(f"Luminance scale set to: {luminance_scale}%")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 100.")
                continue

        elif choice == '4':
            photons_input = input("Enter number of photonic units (e.g., 10000): ")
            try:
                num_photons = int(photons_input)
                if num_photons <= 0:
                    raise ValueError
                print(f"Number of photonic units set to: {num_photons}")
            except ValueError:
                print("Invalid input. Please enter a positive integer.")
                continue

        elif choice == '5':
            # Set defaults if not set
            spectrum_name = locals().get('selected_spectrum', 'sunlight')
            beam_std = locals().get('beam_std', 10.0)
            luminance_scale = locals().get('luminance_scale', 100.0)
            num_photons = locals().get('num_photons', 10000)

            output_filename = f"{spectrum_name}_std{beam_std}_lum{luminance_scale}.png"

            generate_light_simulation(
                noise_instance=noise_instance,
                spectrum_name=spectrum_name,
                image_size=(512, 512),
                num_photons=num_photons,
                std=beam_std,
                luminance_scale=luminance_scale,
                output_filename=output_filename
            )

        elif choice == '6':
            print("Exiting the simulation.")
            sys.exit(0)

        else:
            print("Invalid choice. Please select a number between 1 and 6.")

if __name__ == "__main__":
    main_menu()
