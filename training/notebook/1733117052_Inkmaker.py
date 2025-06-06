import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class Inkmaker:
    """
    Inkmaker: A class for applying FFT-driven rainbow ink effects to text tensors
    and generating spectrographic ink applications for illuminated text rendering.
    """

    def __init__(self, wavelength_range: Tuple[float, float] = (400, 700)):
        """
        Initialize the Inkmaker with wavelength range for rainbow colors.

        Args:
            wavelength_range (Tuple[float, float]): The range of wavelengths (in nm) to simulate.
        """
        self.wavelength_range = wavelength_range

    def wavelength_to_rgb(self, wavelength: float, intensity: float) -> torch.Tensor:
        """
        Convert a wavelength to an RGB color based on the physics of light.

        Args:
            wavelength (float): The wavelength in nanometers.
            intensity (float): The intensity of the color.

        Returns:
            torch.Tensor: The RGB representation of the wavelength.
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

        # Scale to intensity
        return torch.tensor([r, g, b]) * intensity * 255.0

    def generate_color_map(self, spectrum: torch.Tensor, fft_data: torch.Tensor) -> torch.Tensor:
        """
        Generate a color map based on FFT data and spectrum.

        Args:
            spectrum (torch.Tensor): Wavelengths across the page.
            fft_data (torch.Tensor): FFT data dictating intensity across the spectrum.

        Returns:
            torch.Tensor: A tensor representing the RGB color map.
        """
        colors = []
        for wavelength in spectrum:
            intensity = fft_data[int(wavelength) % len(fft_data)]  # Modulate by FFT
            color = self.wavelength_to_rgb(wavelength, intensity)
            colors.append(color)
        return torch.stack(colors, dim=1)  # Shape: (3, width)

    def generate_colored_noise(self, size: int = 1024, noise_type: str = "white") -> torch.Tensor:
        """
        Generate colored noise based on physics definitions and FFT energy distribution.

        Args:
            size (int): Size of the spectrum to generate.
            noise_type (str): Type of noise to generate ("white", "pink", "brown").

        Returns:
            torch.Tensor: RGB tensor representing the colored noise spectrum.
        """
        # Generate white noise (flat spectrum)
        white_noise = torch.randn(size)

        # Generate noise based on type
        if noise_type == "white":
            fft_result = torch.fft.fft(white_noise)
        elif noise_type == "pink":
            # Pink noise: Amplitude decreases ~1/sqrt(frequency)
            pink_filter = 1.0 / torch.sqrt(torch.arange(1, size + 1, dtype=torch.float32))
            fft_result = torch.fft.fft(white_noise) * pink_filter
        elif noise_type == "brown":
            # Brown noise: Amplitude decreases ~1/frequency
            brown_filter = 1.0 / torch.arange(1, size + 1, dtype=torch.float32)
            fft_result = torch.fft.fft(white_noise) * brown_filter
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        # Normalize FFT energy with standard deviation
        fft_magnitude = torch.abs(fft_result)
        energy_distribution = torch.norm(fft_magnitude, p=2) / size
        fft_magnitude_normalized = fft_magnitude / energy_distribution

        # Create a spectrum by mapping normalized energy to intensity
        spectrum = torch.linspace(self.wavelength_range[0], self.wavelength_range[1], size)
        fft_intensity = fft_magnitude_normalized.real  # Take only the real part

        # Map FFT intensity to RGB using wavelength
        colors = [self.wavelength_to_rgb(wavelength, intensity) for wavelength, intensity in zip(spectrum, fft_intensity)]
        return torch.stack(colors, dim=1)  # Shape: (3, size)

    def demonstrate_colored_noise(self, output_path: str = "colored_noise_demo.png", size: int = 1024):
        """
        Generate and save a demonstration of colored noise for white, pink, and brown noise.

        Args:
            output_path (str): Path to save the demonstration image.
            size (int): Size of the noise spectrum.
        """
        noise_types = ["white", "pink", "brown"]
        demo_height = 100  # Height for each noise type visualization
        demo_images = []

        for noise_type in noise_types:
            color_map = self.generate_colored_noise(size, noise_type=noise_type)
            # Stretch each noise horizontally for visualization
            noise_image = color_map.unsqueeze(1).repeat(1, demo_height, 1)
            demo_images.append(noise_image)

        # Combine all noise types vertically
        combined_image = torch.cat(demo_images, dim=1)
        combined_image = combined_image.permute(1, 2, 0).numpy().astype(np.uint8)

        # Save the demonstration image
        plt.imsave(output_path, combined_image)


if __name__ == "__main__":
    # Initialize Inkmaker
    inkmaker = Inkmaker()

    # Demonstrate colored noise
    inkmaker.demonstrate_colored_noise("colored_noise_demo.png", size=2048)
