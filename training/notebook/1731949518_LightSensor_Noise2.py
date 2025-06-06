import torch
import numpy as np
from PIL import Image
import math
import sys
import random

class Noise2:
    LIGHT_SPECTRA = {
        # Noise Sources
        "WHITE_NOISE": {
            "gameid": "noise_white",
            "type": "noise",
            "description": "White noise with uniform frequency distribution.",
            "parameters": {"mean": 0.0, "std": 1.0}
        },
        "PINK_NOISE": {
            "gameid": "noise_pink",
            "type": "noise",
            "description": "Pink noise with equal energy per octave.",
            "parameters": {"mean": 0.0, "std": 0.5}
        },
        "BROWN_NOISE": {
            "gameid": "noise_brown",
            "type": "noise",
            "description": "Brown noise with lower frequencies dominating.",
            "parameters": {"mean": 0.0, "std": 0.3}
        },
        "BLUE_NOISE": {
            "gameid": "noise_blue",
            "type": "noise",
            "description": "Blue noise with higher frequencies dominating.",
            "parameters": {"mean": 0.0, "std": 0.7}
        },
        "VIOLET_NOISE": {
            "gameid": "noise_violet",
            "type": "noise",
            "description": "Violet noise with extreme high-frequency emphasis.",
            "parameters": {"mean": 0.0, "std": 0.9}
        },

        # Lasers
        "cheap_pen_light": {
            "gameid": "laser_red_pen",
            "type": "laser",
            "description": "Basic red laser pointer for general use.",
            "wavelength": 650,
            "std": 2,
            "simulation": {
                "envelope": {"attack": 0.1, "sustain": 0.8, "decay": 0.1},
                "error_factor": 0.05
            }
        },
        "green_pointer": {
            "gameid": "laser_green_pointer",
            "type": "laser",
            "description": "Green laser pointer with higher visibility.",
            "wavelength": 532,
            "std": 1,
            "simulation": {
                "envelope": {"attack": 0.2, "sustain": 0.6, "decay": 0.2},
                "error_factor": 0.04
            }
        },
        "blue_pointer": {
            "gameid": "laser_blue_pointer",
            "type": "laser",
            "description": "Blue laser pointer for precise applications.",
            "wavelength": 450,
            "std": 1,
            "simulation": {
                "envelope": {"attack": 0.15, "sustain": 0.7, "decay": 0.15},
                "error_factor": 0.03
            }
        },
        "industrial_red": {
            "gameid": "laser_industrial_red",
            "type": "laser",
            "description": "Industrial red laser for specialized equipment.",
            "wavelength": 780,
            "std": 5,
            "simulation": {
                "envelope": {"attack": 0.3, "sustain": 0.5, "decay": 0.2},
                "error_factor": 0.06
            }
        },
        "scientific_uv": {
            "gameid": "laser_scientific_uv",
            "type": "laser",
            "description": "Scientific ultraviolet laser for precision research.",
            "wavelength": 355,
            "std": 0.5,
            "simulation": {
                "envelope": {"attack": 0.25, "sustain": 0.65, "decay": 0.1},
                "error_factor": 0.02
            }
        },
        "fiber_laser": {
            "gameid": "laser_fiber",
            "type": "laser",
            "description": "Infrared fiber laser for high-power applications.",
            "wavelength": 1064,
            "std": 3,
            "simulation": {
                "envelope": {"attack": 0.4, "sustain": 0.5, "decay": 0.1},
                "error_factor": 0.05
            }
        },
        "custom_lab_machine": {
            "gameid": "laser_custom_lab",
            "type": "laser",
            "description": "Custom infrared laser for advanced lab equipment.",
            "wavelength": 1053,
            "std": 0.1,
            "simulation": {
                "envelope": {"attack": 0.2, "sustain": 0.7, "decay": 0.1},
                "error_factor": 0.01
            }
        },

        # Spectra
        "helium_neon_laser": {
            "gameid": "laser_helium_neon",
            "type": "laser",
            "description": "Helium-neon laser emitting tight red light.",
            "bands": [{"wavelength": 632.8, "std": 0.1, "intensity": 1.0}],
            "beam_tightness": 2.0,
            "simulation": {"envelope": {"attack": 0.1, "sustain": 0.9, "decay": 0.1}, "error_factor": 0.02}
        },
        "argon_ion_laser": {
            "gameid": "laser_argon_ion",
            "type": "laser",
            "description": "Argon-ion laser emitting blue and green light.",
            "bands": [
                {"wavelength": 488, "std": 0.1, "intensity": 0.8},
                {"wavelength": 514.5, "std": 0.1, "intensity": 0.9}
            ],
            "beam_tightness": 2.0,
            "simulation": {"envelope": {"attack": 0.15, "sustain": 0.85, "decay": 0.15}, "error_factor": 0.03}
        },
        "fluorescent_lighting": {
            "gameid": "spectra_fluorescent",
            "type": "spectra",
            "description": "Fluorescent light with broad emission bands.",
            "bands": [
                {"wavelength": 436, "std": 15, "intensity": 0.6},
                {"wavelength": 546, "std": 15, "intensity": 1.0},
                {"wavelength": 611, "std": 15, "intensity": 0.4}
            ],
            "beam_tightness": 15.0,
            "simulation": {"envelope": {"attack": 0, "sustain": 1, "decay": 0}}
        },
        "led_street_lamp": {
            "gameid": "spectra_led_street",
            "type": "spectra",
            "description": "LED street lamp with broad but distinct bands.",
            "bands": [
                {"wavelength": 450, "std": 10, "intensity": 0.7},
                {"wavelength": 520, "std": 10, "intensity": 1.0},
                {"wavelength": 610, "std": 10, "intensity": 0.8}
            ],
            "beam_tightness": 12.0,
            "simulation": {"envelope": {"attack": 0, "sustain": 1, "decay": 0}}
        },
        "neon_sign": {
            "gameid": "spectra_neon_sign",
            "type": "spectra",
            "description": "Neon sign with prominent orange and red emissions.",
            "bands": [
                {"wavelength": 585, "std": 8, "intensity": 1.0},
                {"wavelength": 640, "std": 8, "intensity": 0.9}
            ],
            "beam_tightness": 5.0,
            "simulation": {"envelope": {"attack": 0, "sustain": 1, "decay": 0}}
        },
        "sunlight": {
            "gameid": "spectra_sunlight",
            "type": "spectra",
            "description": "Natural sunlight with a broad and diffused spectrum.",
            "bands": [
                {"wavelength": 400, "std": 10, "intensity": 0.3},
                {"wavelength": 450, "std": 10, "intensity": 0.5},
                {"wavelength": 500, "std": 10, "intensity": 0.8},
                {"wavelength": 550, "std": 10, "intensity": 1.0},
                {"wavelength": 600, "std": 10, "intensity": 0.9},
                {"wavelength": 650, "std": 10, "intensity": 0.7}
            ],
            "beam_tightness": 20.0,
            "simulation": {"envelope": {"attack": 0, "sustain": 1, "decay": 0}}
        },
        "blue_sky_daylight": {
            "gameid": "spectra_blue_sky",
            "type": "spectra",
            "description": "Daylight scattered by the atmosphere, creating a blue hue.",
            "bands": [
                {"wavelength": 450, "std": 15, "intensity": 1.0},
                {"wavelength": 500, "std": 15, "intensity": 0.7},
                {"wavelength": 550, "std": 15, "intensity": 0.5}
            ],
            "beam_tightness": 18.0,
            "simulation": {"envelope": {"attack": 0, "sustain": 1, "decay": 0}}
        },
        "tungsten": {
            "gameid": "spectra_tungsten",
            "type": "spectra",
            "description": "Tungsten incandescent light with warm, broad emissions.",
            "bands": [
                {"wavelength": 600, "std": 20, "intensity": 1.0},
                {"wavelength": 650, "std": 20, "intensity": 0.8},
                {"wavelength": 700, "std": 20, "intensity": 0.5}
            ],
            "beam_tightness": 22.0,
            "simulation": {"envelope": {"attack": 0.02, "sustain": 0.97, "decay": 0.01}, "error_factor": 0.04}
        },
        "sodium_street_lamp": {
            "gameid": "spectra_sodium_lamp",
            "type": "spectra",
            "description": "Sodium street lamp with a bright yellow emission.",
            "bands": [
                {"wavelength": 589, "std": 5, "intensity": 1.0}
            ],
            "beam_tightness": 10.0,
            "simulation": {"envelope": {"attack": 0.01, "sustain": 0.98, "decay": 0.01}, "error_factor": 0.02}
        },
        "halogen": {
            "gameid": "spectra_halogen",
            "type": "spectra",
            "description": "Halogen light with a warm, continuous spectrum.",
            "bands": [
                {"wavelength": 500, "std": 15, "intensity": 0.6},
                {"wavelength": 600, "std": 15, "intensity": 1.0},
                {"wavelength": 650, "std": 15, "intensity": 0.8}
            ],
            "beam_tightness": 17.0,
            "simulation": {"envelope": {"attack": 0.05, "sustain": 0.9, "decay": 0.05}, "error_factor": 0.03}
        },
        "moonlight": {
            "gameid": "spectra_moonlight",
            "type": "spectra",
            "description": "Diffused moonlight reflecting sunlight.",
            "bands": [
                {"wavelength": 450, "std": 10, "intensity": 0.6},
                {"wavelength": 500, "std": 10, "intensity": 0.4},
                {"wavelength": 550, "std": 10, "intensity": 0.3}
            ],
            "beam_tightness": 25.0,
            "simulation": {"envelope": {"attack": 0, "sustain": 1, "decay": 0}, "error_factor": 0.01}
        }
    }

    LIGHT_SENSORS = {
        "photon_cascade_vacuum_tube": {
            "sensor_id": "sensor_pcv",
            "description": "Photon cascade vacuum tube with high sensitivity to single photons.",
            "rgb_profile": {"r": 1.0, "g": 1.0, "b": 1.0},  # Linear response
            "range": "all"
        },
        "toy_digital_camera": {
            "sensor_id": "sensor_tdc",
            "description": "Toy digital camera with basic RGB detection capabilities.",
            "rgb_profile": {"r": 0.8, "g": 1.0, "b": 0.6},  # Simplified RGB sensitivity
            "range": "visible"
        },
        "advanced_nn_sensor": {
            "sensor_id": "sensor_nn",
            "description": "Advanced neural network-based sensor with learned RGB profiles.",
            "rgb_profile": {"r": 0.9, "g": 0.95, "b": 0.85},
            "range": "visible"
        }
        # Add more sensors as needed
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
            spectrum_name (str): The name of the noise spectrum (e.g., 'white', 'pink').
            distribution (str): Type of distribution to use ('normal', 'uniform', 'histogram').

        Returns:
            Noise: A Noise instance with parameters corresponding to the specified spectrum.
        """
        spectrum_mapping = {
            "white": cls.LIGHT_SPECTRA["WHITE_NOISE"],
            "pink": cls.LIGHT_SPECTRA["PINK_NOISE"],
            "brown": cls.LIGHT_SPECTRA["BROWN_NOISE"],
            "blue": cls.LIGHT_SPECTRA["BLUE_NOISE"],
            "violet": cls.LIGHT_SPECTRA["VIOLET_NOISE"]
        }
        if spectrum_name not in spectrum_mapping:
            raise ValueError("Unsupported spectrum name. Use 'white', 'pink', 'brown', 'blue', or 'violet'.")
        params = spectrum_mapping[spectrum_name]
        return cls(mean=params["parameters"]["mean"], std=params["parameters"]["std"], distribution=distribution)

    def generate_noise_tensor(self, size):
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

    def get_closest_color_hex(self, spectrum_name, normalize=False):
        """
        Returns the web-safe hex color code closest to the given light spectrum.

        Args:
            spectrum_name (str): The name of the light spectrum.
            normalize (bool): Whether to normalize the spectrum to the visible range.

        Returns:
            str: The web-safe hex color code representing the spectrum.
        """
        if spectrum_name not in self.LIGHT_SPECTRA:
            raise ValueError("Unsupported spectrum name. Available options: {}".format(", ".join(self.LIGHT_SPECTRA.keys())))

        spectrum = self.LIGHT_SPECTRA[spectrum_name]
        bands = spectrum.get("bands", [])

        if normalize:
            # Normalize intensities within the visible spectrum (400-700 nm)
            total_intensity = sum([band["intensity"] for band in bands if 400 <= band["wavelength"] <= 700])
            if total_intensity == 0:
                total_intensity = 1  # Prevent division by zero
            bands = [
                {**band, "intensity": band["intensity"] / total_intensity if 400 <= band["wavelength"] <= 700 else 0.0}
                for band in bands
            ]

        # Calculate weighted RGB values based on intensity and wavelength contributions
        r, g, b = 0.0, 0.0, 0.0
        for band in bands:
            wavelength = band["wavelength"]
            intensity = band["intensity"]
            rgb = self.wavelength_to_rgb(wavelength)
            r += rgb[0] * intensity
            g += rgb[1] * intensity
            b += rgb[2] * intensity

        # Clamp values between 0 and 1
        r = min(max(r, 0), 1)
        g = min(max(g, 0), 1)
        b = min(max(b, 0), 1)

        # Convert to hex
        hex_color = '#{0:02x}{1:02x}{2:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        return hex_color

    def wavelength_to_rgb(self, wavelength):
        """
        Converts a wavelength in the visible spectrum to an approximate RGB value.

        Args:
            wavelength (float): Wavelength in nanometers.

        Returns:
            tuple: RGB values in the range [0, 1].
        """
        gamma = 0.8
        intensity_max = 1.0
        factor = 0.0
        r, g, b = 0.0, 0.0, 0.0

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

        # Intensity correction
        if 380 <= wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif 645 <= wavelength <= 780:
            factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
        else:
            factor = 1.0

        r = (r * factor) ** gamma if r > 0 else 0
        g = (g * factor) ** gamma if g > 0 else 0
        b = (b * factor) ** gamma if b > 0 else 0

        return (r, g, b)

    def closest_16bit_color(self, r, g, b):
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

    def hex_to_rgb(self, hex_color):
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

    def apply_alpha_blending(self, bg_color, fg_color, alpha):
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

    def probabilistic_mixer(self, light_sources, image_size=(512, 512), scale_factor=2):
        """
        Mix multiple light sources probabilistically to generate a noisy, blended image.

        Args:
            light_sources (list): List of dictionaries representing light sources with their properties.
            image_size (tuple): Size of the output image (width, height).
            scale_factor (int): Scale factor for subpixel tensor generation.

        Returns:
            Image: Final noisy, lossy image as a PIL Image.
        """
        width, height = image_size
        scaled_size = (width * scale_factor, height * scale_factor)
        final_image = torch.zeros((scaled_size[1], scaled_size[0], 3), dtype=torch.float32)

        for source in light_sources:
            spectrum_name = source['name']
            properties = self.LIGHT_SPECTRA[spectrum_name]

            # Generate noise tensor for the source
            if properties['type'] == 'noise':
                noise = self.generate_noise_tensor(scaled_size)
                # Normalize based on std
                noise = (noise - properties['parameters']['mean']) / properties['parameters']['std']
                # Scale intensity
                intensity_scale = source.get('intensity_scale', 1.0)
                noise = noise * intensity_scale
                # Clamp noise
                noise = torch.clamp(noise, 0, 1)
                # Assign to RGB (assuming grayscale noise)
                final_image += noise.unsqueeze(2).repeat(1, 1, 3)
            elif properties['type'] in ['laser', 'spectra']:
                # Handle lasers and spectra
                bands = properties.get('bands', [])
                beam_tightness = properties.get('beam_tightness', 10.0)
                simulation = properties.get('simulation', {})
                envelope = simulation.get('envelope', {"attack": 0, "sustain": 1, "decay": 0})
                error_factor = simulation.get('error_factor', 0.05)

                # Simulate emission profile based on envelope
                attack, sustain, decay = envelope['attack'], envelope['sustain'], envelope['decay']
                total_duration = attack + sustain + decay

                # For simplicity, consider all emissions active (attack=0, sustain=1, decay=0 for natural sources)
                emission_probability = sustain

                # Generate positions based on beam tightness
                positions = self.generate_noise_tensor((scaled_size[1], scaled_size[0], 2))  # (y, x, 2)
                positions = (positions - 0.5) * 2 * beam_tightness  # Normalize to [-beam_tightness, beam_tightness]

                # Create Gaussian beam
                y = torch.linspace(-1, 1, steps=scaled_size[1]).unsqueeze(1).repeat(1, scaled_size[0])
                x = torch.linspace(-1, 1, steps=scaled_size[0]).unsqueeze(0).repeat(scaled_size[1], 1)
                distance_squared = x**2 + y**2
                beam_intensity = torch.exp(-distance_squared / (2 * (beam_tightness / scaled_size[0])**2))
                beam_intensity = beam_intensity * emission_probability

                # Add random noise based on error factor
                beam_intensity += self.generate_noise_tensor(scaled_size) * error_factor
                beam_intensity = torch.clamp(beam_intensity, 0, 1)

                # Convert wavelength to RGB
                rgb_hex = self.get_closest_color_hex(spectrum_name, normalize=False)
                rgb = torch.tensor(self.hex_to_rgb(rgb_hex)) / 255.0  # Normalize to [0,1]

                # Apply RGB profile based on distance
                # Here, assuming variable intensity related to distance is already handled by beam_intensity

                # Apply to final image
                for i in range(3):
                    final_image[:, :, i] += beam_intensity * rgb[i]

        # Normalize final image
        final_image = torch.clamp(final_image, 0, 1)

        # Resize down by scale factor
        final_image = final_image.view(scale_factor, height, scale_factor, width, 3).mean(dim=(0, 2))

        # Convert to 16-bit RGB
        final_image_16bit = (final_image * 65535).numpy().astype(np.uint16)

        # Convert to PIL Image
        final_pil_image = Image.fromarray(final_image_16bit, mode='I;16')
        return final_pil_image
import torch
from torch_sparse import SparseTensor

class Noise2:
    # Existing methods and constants are retained here ...

    def generate_emission_tensor(self, light_sources, image_size, wavelength_range=(380, 780), num_wavelengths=401):
        """
        Generate a sparse tensor representing photonic emissions.

        Args:
            light_sources (list): List of dictionaries representing light sources with properties.
            image_size (tuple): Tuple (width, height) of the emission space.
            wavelength_range (tuple): Range of wavelengths (start, end).
            num_wavelengths (int): Number of wavelength bins for precision.

        Returns:
            SparseTensor: COO-format tensor of emissions with structure:
                [Batch, Target Wavelength, Actual Wavelength, Loss Alpha, Intensity, X Arrival, Y Arrival].
        """
        width, height = image_size
        wavelength_bins = torch.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)

        emissions = []
        batch_index = 0  # For simplicity, using single batch (can be extended)

        for source in light_sources:
            source_name = source['name']
            properties = self.LIGHT_SPECTRA[source_name]
            beam_width = properties.get('beam_tightness', 10.0)
            intensity_scale = source.get('intensity_scale', 1.0)

            if properties['type'] in ['laser', 'spectra']:
                for band in properties.get('bands', []):
                    wavelength = band['wavelength']
                    std = band['std']
                    intensity = band['intensity'] * intensity_scale

                    # Generate sparse photon emissions
                    for _ in range(intensity * 1000):  # Scale number of emissions arbitrarily
                        x = torch.normal(mean=width // 2, std=beam_width / 2)
                        y = torch.normal(mean=height // 2, std=beam_width / 2)

                        distance = torch.sqrt(x**2 + y**2)
                        alpha = 1 / (distance + 1e-3)  # Avoid division by zero

                        emissions.append([
                            batch_index,  # Batch
                            wavelength,  # Target Wavelength
                            wavelength + torch.normal(0, std),  # Actual Wavelength
                            alpha,  # Loss Alpha
                            intensity,  # Original Intensity
                            x.item(),  # X Arrival
                            y.item(),  # Y Arrival
                        ])
        
        # Convert to sparse tensor
        emissions = torch.tensor(emissions, dtype=torch.float32)
        indices = emissions[:, :6].T.long()  # Use first 6 columns as indices
        values = emissions[:, 6]  # Use the 7th column as values (intensity)
        sparse_emission_tensor = torch.sparse_coo_tensor(indices, values, size=(1, num_wavelengths, num_wavelengths, width, height))

        return sparse_emission_tensor
    def generate_probabilistic_image(self, relative_start=0.0, duration=1.0, light_sources=None, image_size=(512, 512), scale_factor=2):
        """
        Generates a probabilistic image by mixing multiple light sources.

        Args:
            relative_start (float): Relative start time of the emission.
            duration (float): Duration of the emission.
            light_sources (list): List of light source names to include.
            image_size (tuple): Size of the output image (width, height).
            scale_factor (int): Scale factor for subpixel tensor generation.

        Returns:
            Image: Final noisy, blended image as a PIL Image.
        """
        if light_sources is None:
            light_sources = list(self.LIGHT_SPECTRA.keys())

        # Prepare light sources with additional parameters
        sources = []
        for name in light_sources:
            source = {
                "name": name,
                "relative_start": relative_start,
                "duration": duration,
                "intensity_scale": self.LIGHT_SPECTRA[name].get("simulation", {}).get("envelope", {}).get("sustain", 1.0)
            }
            sources.append(source)

        # Mix the sources probabilistically
        final_image = self.probabilistic_mixer(sources, image_size=image_size, scale_factor=scale_factor)
        return final_image

    def wavelength_to_rgb_custom(self, wavelength, intensity):
        """
        Custom wavelength to RGB conversion with intensity modulation.

        Args:
            wavelength (float): Wavelength in nanometers.
            intensity (float): Intensity scaling factor.

        Returns:
            tuple: RGB values in the range [0, 1] scaled by intensity.
        """
        r, g, b = self.wavelength_to_rgb(wavelength)
        return (r * intensity, g * intensity, b * intensity)

class LightSensor:
    def __init__(self, sensor_id, description, rgb_profile, range_type):
        """
        Initializes a LightSensor with specific properties.

        Args:
            sensor_id (str): Unique identifier for the sensor.
            description (str): Description of the sensor.
            rgb_profile (dict): RGB sensitivity profile.
            range_type (str): Detection range type ('visible', 'all', etc.).
        """
        self.sensor_id = sensor_id
        self.description = description
        self.rgb_profile = rgb_profile
        self.range_type = range_type

    def detect_light(self, image):
        """
        Simulates light detection based on the sensor's RGB profile.

        Args:
            image (Image): PIL Image representing the light emission.

        Returns:
            np.ndarray: Detected RGB values as a 16-bit array.
        """
        rgb = np.array(image, dtype=np.float32) / 65535.0  # Normalize
        detected_r = rgb[:, :, 0] * self.rgb_profile.get('r', 1.0)
        detected_g = rgb[:, :, 1] * self.rgb_profile.get('g', 1.0)
        detected_b = rgb[:, :, 2] * self.rgb_profile.get('b', 1.0)
        detected_rgb = np.stack([detected_r, detected_g, detected_b], axis=2)
        detected_rgb = np.clip(detected_rgb, 0, 1)
        detected_rgb_16bit = (detected_rgb * 65535).astype(np.uint16)
        return detected_rgb_16bit

    @classmethod
    def from_dict(cls, sensor_dict):
        """
        Creates a LightSensor instance from a dictionary.

        Args:
            sensor_dict (dict): Dictionary containing sensor properties.

        Returns:
            LightSensor: An instance of LightSensor.
        """
        return cls(
            sensor_id=sensor_dict["sensor_id"],
            description=sensor_dict["description"],
            rgb_profile=sensor_dict["rgb_profile"],
            range_type=sensor_dict["range"]
        )

def generate_light_simulation(noise_instance, sensor_instance, light_sources, image_size=(512, 512), scale_factor=2, output_filename="light_simulation.png"):
    """
    Generate a light simulation image based on the specified parameters and detect it using a sensor.

    Args:
        noise_instance (Noise): Instance of the Noise class.
        sensor_instance (LightSensor): Instance of the LightSensor class.
        light_sources (list): List of light source names to include in the simulation.
        image_size (tuple): Size of the output image (width, height).
        scale_factor (int): Scale factor for subpixel tensor generation.
        output_filename (str): Filename for the output PNG image.
    """
    # Generate the probabilistic image
    simulated_image = noise_instance.generate_probabilistic_image(
        light_sources=light_sources,
        image_size=image_size,
        scale_factor=scale_factor
    )

    # Save the simulated image
    simulated_image.save(output_filename, format="PNG")
    print(f"Simulated image saved as {output_filename}")

    # Detect the light using the sensor
    detected_rgb = sensor_instance.detect_light(simulated_image)

    # Convert detected RGB to image
    detected_image = Image.fromarray(detected_rgb, mode='RGB')
    detected_filename = f"detected_{output_filename}"
    detected_image.save(detected_filename, format="PNG")
    print(f"Detected image saved as {detected_filename}")

def main_menu():
    """
    Display the text-based menu and handle user inputs to generate light simulations.
    """
    noise_instance = Noise2()

    # Initialize sensors
    sensors = {k: LightSensor.from_dict(v) for k, v in Noise2.LIGHT_SENSORS.items()}

    selected_spectrum = "sunlight"
    beam_std = noise_instance.LIGHT_SPECTRA[selected_spectrum].get("beam_tightness", 10.0)
    luminance_scale = 100.0
    num_photons = 10000
    normalize_spectrum = False

    while True:
        print("\n=== Light Simulation Menu ===")
        print("1. Select Light Spectrum")
        print("2. Configure Beam Tightness (std)")
        print("3. Configure Luminance Scale (%)")
        print("4. Configure Number of Photons")
        print("5. Normalize Spectrum to Visible Range")
        print("6. Select Light Sensors")
        print("7. Generate Simulation")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            print("\nAvailable Light Spectra:")
            for idx, key in enumerate(Noise2.LIGHT_SPECTRA.keys(), start=1):
                print(f"{idx}. {key.replace('_', ' ').title()}")
            spectrum_choice = input("Select a spectrum by number: ")
            try:
                spectrum_idx = int(spectrum_choice) - 1
                if spectrum_idx < 0 or spectrum_idx >= len(Noise2.LIGHT_SPECTRA):
                    raise IndexError
                spectrum_name = list(Noise2.LIGHT_SPECTRA.keys())[spectrum_idx]
                selected_spectrum = spectrum_name
                beam_std = noise_instance.LIGHT_SPECTRA[selected_spectrum].get("beam_tightness", 10.0)
                print(f"Selected spectrum: {selected_spectrum.replace('_', ' ').title()}")
            except (IndexError, ValueError):
                print("Invalid selection. Please try again.")
                continue

        elif choice == '2':
            std_input = input("Enter beam tightness (std) (e.g., 10.0): ")
            try:
                std = float(std_input)
                if std <= 0:
                    raise ValueError
                beam_std = std
                # Update the beam tightness for the selected spectrum
                noise_instance.LIGHT_SPECTRA[selected_spectrum]["beam_tightness"] = beam_std
                print(f"Beam tightness (std) set to: {beam_std}")
            except ValueError:
                print("Invalid input. Please enter a positive number.")
                continue

        elif choice == '3':
            luminance_input = input("Enter luminance scale percentage (0-100, default 100): ")
            try:
                if luminance_input.strip() == "":
                    luminance_scale = 100.0
                else:
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
            normalize_input = input("Normalize spectrum to visible range? (y/n, default n): ").lower()
            if normalize_input in ['y', 'yes']:
                normalize_spectrum = True
                print("Spectrum normalization enabled.")
            elif normalize_input in ['n', 'no', '']:
                normalize_spectrum = False
                print("Spectrum normalization disabled.")
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue

        elif choice == '6':
            print("\nAvailable Light Sensors:")
            for idx, key in enumerate(Noise2.LIGHT_SENSORS.keys(), start=1):
                sensor = Noise2.LIGHT_SENSORS[key]
                print(f"{idx}. {key.replace('_', ' ').title()} - {sensor['description']}")
            sensor_choice = input("Select a sensor by number: ")
            try:
                sensor_idx = int(sensor_choice) - 1
                if sensor_idx < 0 or sensor_idx >= len(Noise2.LIGHT_SENSORS):
                    raise IndexError
                sensor_name = list(Noise2.LIGHT_SENSORS.keys())[sensor_idx]
                selected_sensor = sensors[sensor_name]
                print(f"Selected sensor: {sensor_name.replace('_', ' ').title()}")
            except (IndexError, ValueError):
                print("Invalid selection. Please try again.")
                continue

        elif choice == '7':
            # Set defaults if not set
            spectrum_name = selected_spectrum
            beam_std = noise_instance.LIGHT_SPECTRA[spectrum_name].get("beam_tightness", 10.0)
            output_filename = f"{spectrum_name}_std{beam_std}_lum{luminance_scale}_norm{normalize_spectrum}.png"

            # Gather all light sources (for simplicity, using the selected spectrum)
            light_sources = [spectrum_name]

            generate_light_simulation(
                noise_instance=noise_instance,
                sensor_instance=selected_sensor,
                light_sources=light_sources,
                image_size=(512, 512),
                scale_factor=2,
                output_filename=output_filename
            )

        elif choice == '8':
            print("Exiting the simulation.")
            sys.exit(0)

        else:
            print("Invalid choice. Please select a number between 1 and 8.")

if __name__ == "__main__":
    main_menu()
