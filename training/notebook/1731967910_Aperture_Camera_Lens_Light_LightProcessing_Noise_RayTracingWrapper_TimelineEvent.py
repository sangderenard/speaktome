import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import mitsuba as mi

# Initialize Mitsuba in spectral mode
mi.set_variant("scalar_spectral")  # Enable spectral rendering for arbitrary wavelengths

# Set up a logging configuration
logging.basicConfig(filename="emission_log.jsonl", level=logging.INFO, format="%(message)s")

def log_emission(emission_details: Dict[str, Any]):
    # Add a timestamp to the log
    emission_details["timestamp"] = datetime.utcnow().isoformat()
    # Serialize as JSON for structured logging
    logging.info(json.dumps(emission_details))

@dataclass
class TimelineEvent:
    source_name: str
    start_time: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))
    duration: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0], dtype=torch.float64))
    intensity: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0], dtype=torch.float64))
    a: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))
    s: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0], dtype=torch.float64))
    d: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))
    size: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0], dtype=torch.float64))
    beam_tightness: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0], dtype=torch.float64))

    def __post_init__(self):
        # Automatically convert numeric inputs to tensors
        for field_name, value in self.__dict__.items():
            if isinstance(value, (int, float)):
                setattr(self, field_name, torch.tensor([value], dtype=torch.float64))
            elif isinstance(value, torch.Tensor):
                continue  # Leave tensors unchanged

class RayTracingWrapper:
    def __init__(
        self,
        resolution: Tuple[int, int] = (512, 512),
        granularity: int = 100,
        stencil_matrix: Optional[torch.Tensor] = None,
        grid_tuple: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the RayTracingWrapper for handling ray tracing tasks.

        Args:
            resolution (tuple): Resolution of the output image (width, height).
            granularity (int): Number of rays per light source for point source summation.
            stencil_matrix (torch.Tensor, optional): Stencil matrix representing a gobo mask.
            grid_tuple (tuple, optional): (density, depth) for grid scrim effects.
        """
        self.resolution = resolution
        self.granularity = granularity
        self.stencil_matrix = stencil_matrix
        self.grid_tuple = grid_tuple

    def _create_sensor_plane(self):
        """
        Create a sensor plane (camera) in Mitsuba.

        Returns:
            mitsuba.Sensor: The sensor object.
        """
        # Define sensor (image plane)
        sensor_dict = {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]),
            "fov": 45,
            "film": {
                "type": "hdrfilm",
                "width": self.resolution[0],
                "height": self.resolution[1],
                "rfilter": {"type": "gaussian"},
                "pixel_format": "rgb"
            }
        }
        sensor = mi.load_dict(sensor_dict)
        return sensor

    def _apply_stencil(self, sensor_plane):
        """
        Apply stencil (gobo) effects to the sensor plane.

        Args:
            sensor_plane (mitsuba.Sensor): The sensor plane object.

        Returns:
            None
        """
        if self.stencil_matrix is not None:
            # Convert stencil_matrix to a Mitsuba bitmap texture
            stencil_image = self.stencil_matrix.numpy().astype(np.float32)
            # Normalize stencil image to range [0, 1]
            stencil_image /= stencil_image.max()

            # Create a bitmap texture from the stencil image
            stencil_texture = mi.Bitmap(stencil_image, dynamic=False)

            # Create a material that uses the stencil as a mask (transparency)
            stencil_material = mi.load_dict({
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "bitmap": stencil_texture
                }
            })

            # Add a plane with the stencil material in front of the sensor
            stencil_plane = mi.load_dict({
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f.translate([0, 0, -1]),
                "material": stencil_material
            })
            sensor_plane.add_child(stencil_plane)

    def _apply_grid_scrim(self, sensor_plane):
        """
        Apply grid scrim effects to the sensor plane.

        Args:
            sensor_plane (mitsuba.Sensor): The sensor plane object.

        Returns:
            None
        """
        if self.grid_tuple is not None:
            grid_density, grid_depth = self.grid_tuple

            # Create a grid pattern as a texture
            grid_image = np.ones((self.resolution[1], self.resolution[0]), dtype=np.float32)  # Note: Height first
            step_x = int(self.resolution[0] / grid_density)
            step_y = int(self.resolution[1] / grid_density)
            grid_width = max(1, int(grid_depth * min(self.resolution)))

            for x in range(0, self.resolution[0], step_x):
                end_x = min(x + grid_width, self.resolution[0])
                grid_image[:, x:end_x] = 0.5  # Darken grid lines
            for y in range(0, self.resolution[1], step_y):
                end_y = min(y + grid_width, self.resolution[1])
                grid_image[y:end_y, :] = 0.5  # Darken grid lines

            # Create a bitmap texture from the grid image
            grid_texture = mi.Bitmap(grid_image, dynamic=False)

            # Create a material that uses the grid texture
            grid_material = mi.load_dict({
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "bitmap": grid_texture
                }
            })

            # Add a plane with the grid material in front of the sensor
            grid_plane = mi.load_dict({
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f.translate([0, 0, -2]),
                "material": grid_material
            })
            sensor_plane.add_child(grid_plane)

    def trace(
        self,
        light_sources: List[Dict[str, Any]],
        wavelengths: List[float],
        time_slices: List[float],
        extended_sources: bool = True
    ) -> torch.Tensor:
        """
        Perform ray tracing for light sources, wavelengths, and time slices.

        Args:
            light_sources (list): List of light source configurations.
            wavelengths (list): List of wavelengths to trace.
            time_slices (list): Time slices for rendering.
            extended_sources (bool): Whether to simulate extended light sources.

        Returns:
            torch.Tensor: A tensor representing the simulated light features.
        """
        results = []

        for time_idx, time_slice in enumerate(time_slices):
            for light_source in light_sources:
                for wavelength in wavelengths:
                    # Define sensor plane
                    sensor = self._create_sensor_plane()

                    # Apply stencil and grid scrim effects
                    self._apply_stencil(sensor)
                    self._apply_grid_scrim(sensor)

                    # Create a scene and add sensor
                    scene = mi.load_dict({
                        "type": "scene",
                        "sensor": sensor
                    })

                    # Add light sources
                    if extended_sources:
                        # Simulate extended source by distributing rays across the source surface
                        for _ in range(self.granularity):
                            # Random position within the light source radius
                            angle = torch.rand(1).item() * 2 * np.pi
                            r = light_source["size"] * np.sqrt(torch.rand(1).item())
                            jittered_position = [
                                light_source["position"][0] + r * np.cos(angle),
                                light_source["position"][1] + r * np.sin(angle),
                                light_source["position"][2]
                            ]

                            # Create light source for the ray
                            light = mi.load_dict({
                                "type": "point",
                                "position": jittered_position,
                                "intensity": {
                                    "type": "spectrum",
                                    "value": wavelength
                                }
                            })

                            scene.add(light)
                    else:
                        # Simulate point source by emitting from a single position
                        light = mi.load_dict({
                            "type": "point",
                            "position": light_source["position"],
                            "intensity": {
                                "type": "spectrum",
                                "value": wavelength
                            }
                        })
                        scene.add(light)

                    # Perform ray tracing
                    integrator = mi.load_dict({"type": "path"})
                    image = mi.render(scene, integrator=integrator)

                    # Convert the result to a PyTorch tensor
                    image_tensor = torch.tensor(image, dtype=torch.float64)

                    # Append to results
                    results.append(image_tensor)

                    # Log emission details
                    emission_details = {
                        "source_name": light_source["name"],
                        "time_slice": time_slice,
                        "wavelength": wavelength,
                        "position": light_source["position"],
                        "intensity": light_source["intensity"],
                        "size": light_source.get("size", 1.0),
                        "beam_tightness": light_source.get("beam_tightness", 0.1),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    log_emission(emission_details)

        # Stack results into a tensor
        # Shape: [num_time_slices * num_light_sources * num_wavelengths, height, width, channels]
        emission_tensor = torch.stack(results, dim=0)

        return emission_tensor

class Light:
    """
    A photonic discriminator processing class.
    Encapsulates the entire process of generating, processing, and gamifying light emission data.
    """

    class Noise:
        def __init__(self):
            self.LIGHT_SPECTRA = {
                # Noise Sources
                "WHITE_NOISE": {
                    "name": "WHITE_NOISE",
                    "type": "noise",
                    "description": "White noise with uniform frequency distribution.",
                    "parameters": {"mean": 0.0, "std": 1.0}
                },
                "PINK_NOISE": {
                    "name": "PINK_NOISE",
                    "type": "noise",
                    "description": "Pink noise with equal energy per octave.",
                    "parameters": {"mean": 0.0, "std": 0.5}
                },
                "BROWN_NOISE": {
                    "name": "BROWN_NOISE",
                    "type": "noise",
                    "description": "Brown noise with lower frequencies dominating.",
                    "parameters": {"mean": 0.0, "std": 0.3}
                },
                "BLUE_NOISE": {
                    "name": "BLUE_NOISE",
                    "type": "noise",
                    "description": "Blue noise with higher frequencies dominating.",
                    "parameters": {"mean": 0.0, "std": 0.7}
                },
                "VIOLET_NOISE": {
                    "name": "VIOLET_NOISE",
                    "type": "noise",
                    "description": "Violet noise with extreme high-frequency emphasis.",
                    "parameters": {"mean": 0.0, "std": 0.9}
                },

                # Lasers
                "cheap_pen_light": {
                    "name": "cheap_pen_light",
                    "type": "laser",
                    "description": "Basic red laser pointer for general use.",
                    "wavelength": 650.0,
                    "std": 2.0,
                    "simulation": {
                        "envelope": {"attack": 0.1, "sustain": 0.8, "decay": 0.1},
                        "error_factor": 0.05
                    },
                    "size": 0.05,  # Relative to sensor view angle
                    "beam_tightness": 0.1
                },
                "green_pointer": {
                    "name": "green_pointer",
                    "type": "laser",
                    "description": "Green laser pointer with higher visibility.",
                    "wavelength": 532.0,
                    "std": 1.0,
                    "simulation": {
                        "envelope": {"attack": 0.2, "sustain": 0.6, "decay": 0.2},
                        "error_factor": 0.04
                    },
                    "size": 0.05,
                    "beam_tightness": 0.1
                },
                "blue_pointer": {
                    "name": "blue_pointer",
                    "type": "laser",
                    "description": "Blue laser pointer for precise applications.",
                    "wavelength": 450.0,
                    "std": 1.0,
                    "simulation": {
                        "envelope": {"attack": 0.15, "sustain": 0.7, "decay": 0.15},
                        "error_factor": 0.03
                    },
                    "size": 0.05,
                    "beam_tightness": 0.1
                },
                "industrial_red": {
                    "name": "industrial_red",
                    "type": "laser",
                    "description": "Industrial red laser for specialized equipment.",
                    "wavelength": 780.0,
                    "std": 5.0,
                    "simulation": {
                        "envelope": {"attack": 0.3, "sustain": 0.5, "decay": 0.2},
                        "error_factor": 0.06
                    },
                    "size": 0.05,
                    "beam_tightness": 0.1
                },
                "scientific_uv": {
                    "name": "scientific_uv",
                    "type": "laser",
                    "description": "Scientific ultraviolet laser for precision research.",
                    "wavelength": 355.0,
                    "std": 0.5,
                    "simulation": {
                        "envelope": {"attack": 0.25, "sustain": 0.65, "decay": 0.1},
                        "error_factor": 0.02
                    },
                    "size": 0.05,
                    "beam_tightness": 0.1
                },
                "fiber_laser": {
                    "name": "fiber_laser",
                    "type": "laser",
                    "description": "Infrared fiber laser for high-power applications.",
                    "wavelength": 1064.0,
                    "std": 3.0,
                    "simulation": {
                        "envelope": {"attack": 0.4, "sustain": 0.5, "decay": 0.1},
                        "error_factor": 0.05
                    },
                    "size": 0.05,
                    "beam_tightness": 0.1
                },
                "custom_lab_machine": {
                    "name": "custom_lab_machine",
                    "type": "laser",
                    "description": "Custom infrared laser for advanced lab equipment.",
                    "wavelength": 1053.0,
                    "std": 0.1,
                    "simulation": {
                        "envelope": {"attack": 0.2, "sustain": 0.7, "decay": 0.1},
                        "error_factor": 0.01
                    },
                    "size": 0.05,
                    "beam_tightness": 0.1
                },

                # Spectra
                "helium_neon_laser": {
                    "name": "helium_neon_laser",
                    "type": "laser",
                    "description": "Helium-neon laser emitting tight red light.",
                    "bands": [{"wavelength": 632.8, "std": 0.1, "intensity": 1.0}],
                    "beam_tightness": 2.0,
                    "simulation": {"envelope": {"attack": 0.1, "sustain": 0.9, "decay": 0.1}, "error_factor": 0.02}
                },
                "argon_ion_laser": {
                    "name": "argon_ion_laser",
                    "type": "laser",
                    "description": "Argon-ion laser emitting blue and green light.",
                    "bands": [
                        {"wavelength": 488.0, "std": 0.1, "intensity": 0.8},
                        {"wavelength": 514.5, "std": 0.1, "intensity": 0.9}
                    ],
                    "beam_tightness": 2.0,
                    "simulation": {"envelope": {"attack": 0.15, "sustain": 0.85, "decay": 0.15}, "error_factor": 0.03}
                },
                "fluorescent_lighting": {
                    "name": "fluorescent_lighting",
                    "type": "spectra",
                    "description": "Fluorescent light with broad emission bands.",
                    "bands": [
                        {"wavelength": 436.0, "std": 15.0, "intensity": 0.6},
                        {"wavelength": 546.0, "std": 15.0, "intensity": 1.0},
                        {"wavelength": 611.0, "std": 15.0, "intensity": 0.4}
                    ],
                    "beam_tightness": 15.0,
                    "simulation": {"envelope": {"attack": 0.0, "sustain": 1.0, "decay": 0.0}}
                },
                "led_street_lamp": {
                    "name": "led_street_lamp",
                    "type": "spectra",
                    "description": "LED street lamp with broad but distinct bands.",
                    "bands": [
                        {"wavelength": 450.0, "std": 10.0, "intensity": 0.7},
                        {"wavelength": 520.0, "std": 10.0, "intensity": 1.0},
                        {"wavelength": 610.0, "std": 10.0, "intensity": 0.8}
                    ],
                    "beam_tightness": 12.0,
                    "simulation": {"envelope": {"attack": 0.0, "sustain": 1.0, "decay": 0.0}}
                },
                "neon_sign": {
                    "name": "neon_sign",
                    "type": "spectra",
                    "description": "Neon sign with prominent orange and red emissions.",
                    "bands": [
                        {"wavelength": 585.0, "std": 8.0, "intensity": 1.0},
                        {"wavelength": 640.0, "std": 8.0, "intensity": 0.9}
                    ],
                    "beam_tightness": 5.0,
                    "simulation": {"envelope": {"attack": 0.0, "sustain": 1.0, "decay": 0.0}}
                },
                "sunlight": {
                    "name": "sunlight",
                    "type": "spectra",
                    "description": "Natural sunlight with a broad and diffused spectrum.",
                    "bands": [
                        {"wavelength": 400.0, "std": 10.0, "intensity": 0.3},
                        {"wavelength": 450.0, "std": 10.0, "intensity": 0.5},
                        {"wavelength": 500.0, "std": 10.0, "intensity": 0.8},
                        {"wavelength": 550.0, "std": 10.0, "intensity": 1.0},
                        {"wavelength": 600.0, "std": 10.0, "intensity": 0.9},
                        {"wavelength": 650.0, "std": 10.0, "intensity": 0.7}
                    ],
                    "beam_tightness": 20.0,
                    "simulation": {"envelope": {"attack": 0.0, "sustain": 1.0, "decay": 0.0}}
                },
                "blue_sky_daylight": {
                    "name": "blue_sky_daylight",
                    "type": "spectra",
                    "description": "Daylight scattered by the atmosphere, creating a blue hue.",
                    "bands": [
                        {"wavelength": 450.0, "std": 15.0, "intensity": 1.0},
                        {"wavelength": 500.0, "std": 15.0, "intensity": 0.7},
                        {"wavelength": 550.0, "std": 15.0, "intensity": 0.5}
                    ],
                    "beam_tightness": 18.0,
                    "simulation": {"envelope": {"attack": 0.0, "sustain": 1.0, "decay": 0.0}}
                },
                "tungsten": {
                    "name": "tungsten",
                    "type": "spectra",
                    "description": "Tungsten incandescent light with warm, broad emissions.",
                    "bands": [
                        {"wavelength": 600.0, "std": 20.0, "intensity": 1.0},
                        {"wavelength": 650.0, "std": 20.0, "intensity": 0.8},
                        {"wavelength": 700.0, "std": 20.0, "intensity": 0.5}
                    ],
                    "beam_tightness": 22.0,
                    "simulation": {"envelope": {"attack": 0.02, "sustain": 0.97, "decay": 0.01}, "error_factor": 0.04}
                },
                "sodium_street_lamp": {
                    "name": "sodium_street_lamp",
                    "type": "spectra",
                    "description": "Sodium street lamp with a bright yellow emission.",
                    "bands": [
                        {"wavelength": 589.0, "std": 5.0, "intensity": 1.0}
                    ],
                    "beam_tightness": 10.0,
                    "simulation": {"envelope": {"attack": 0.01, "sustain": 0.98, "decay": 0.01}, "error_factor": 0.02}
                },
                "halogen": {
                    "name": "halogen",
                    "type": "spectra",
                    "description": "Halogen light with a warm, continuous spectrum.",
                    "bands": [
                        {"wavelength": 500.0, "std": 15.0, "intensity": 0.6},
                        {"wavelength": 600.0, "std": 15.0, "intensity": 1.0},
                        {"wavelength": 650.0, "std": 15.0, "intensity": 0.8}
                    ],
                    "beam_tightness": 17.0,
                    "simulation": {"envelope": {"attack": 0.05, "sustain": 0.9, "decay": 0.05}, "error_factor": 0.03}
                },
                "moonlight": {
                    "name": "moonlight",
                    "type": "spectra",
                    "description": "Diffused moonlight reflecting sunlight.",
                    "bands": [
                        {"wavelength": 450.0, "std": 10.0, "intensity": 0.6},
                        {"wavelength": 500.0, "std": 10.0, "intensity": 0.4},
                        {"wavelength": 550.0, "std": 10.0, "intensity": 0.3}
                    ],
                    "beam_tightness": 25.0,
                    "simulation": {"envelope": {"attack": 0.0, "sustain": 1.0, "decay": 0.0}}
                }
            }

            self.LIGHT_SENSORS = {
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

            # Convert light spectra and sensors to tensors where applicable
            self.LIGHT_SPECTRA = self.convert_to_tensors(self.LIGHT_SPECTRA)
            self.LIGHT_SENSORS = self.convert_to_tensors(self.LIGHT_SENSORS)

        def convert_to_tensors(self, data):
            if isinstance(data, dict):
                return {k: self.convert_to_tensors(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self.convert_to_tensors(v) for v in data]
            elif isinstance(data, float):
                return torch.tensor(data, dtype=torch.float64, requires_grad=True)
            elif isinstance(data, str):
                return data
            elif isinstance(data, dict) and "wavelength" in data:
                # Handle bands for spectra
                return {k: self.convert_to_tensors(v) for k, v in data.items()}
            return data  # Return as is for unhandled types

        def normalize_envelope(self, attack: torch.Tensor, sustain: torch.Tensor, decay: torch.Tensor):
            total = attack + sustain + decay
            if torch.all(total == 0):
                raise ValueError("Envelope parameters cannot all be zero.")
            return attack / total, sustain / total, decay / total

        def generate_emission_tensor(
            self,
            light_sources: List[Dict[str, Any]],
            wavelengths: List[float],
            time_slices: List[float],
            extended_sources: bool = True,
            stencil_matrix: Optional[torch.Tensor] = None,
            grid_tuple: Optional[Tuple[float, float]] = None
        ) -> torch.Tensor:
            """
            Generate a tensor representing light emissions using RayTracingWrapper.

            Args:
                light_sources (list): List of light source configurations.
                wavelengths (list): List of wavelengths to trace.
                time_slices (list): Time slices for rendering.
                extended_sources (bool): Whether to simulate extended light sources.
                stencil_matrix (torch.Tensor, optional): Stencil matrix for gobo mask.
                grid_tuple (tuple, optional): (density, depth) for grid scrim effects.

            Returns:
                torch.Tensor: A tensor representing the simulated light features.
            """
            # Initialize RayTracingWrapper with optional stencil and grid parameters
            ray_tracer = RayTracingWrapper(
                resolution=(self.get_sensor_resolution()),
                granularity=100,
                stencil_matrix=stencil_matrix,
                grid_tuple=grid_tuple
            )

            # Prepare light source configurations
            formatted_light_sources = []
            for source in light_sources:
                if source["type"] == "laser":
                    # For lasers, assume a single wavelength
                    formatted_light_sources.append({
                        "name": source["name"],
                        "position": [0, 0, 0],  # Assuming origin for simplicity; adjust as needed
                        "intensity": source["intensity"].item(),
                        "wavelength": source["wavelength"].item(),
                        "size": source.get("size", 0.05).item(),
                        "beam_tightness": source.get("beam_tightness", 0.1).item()
                    })
                elif source["type"] == "spectra":
                    # For spectra, iterate over bands
                    for band in source["bands"]:
                        formatted_light_sources.append({
                            "name": source["name"],
                            "position": [0, 0, 0],  # Adjust as needed
                            "intensity": band["intensity"],
                            "wavelength": band["wavelength"],
                            "size": source.get("size", 0.05),
                            "beam_tightness": source.get("beam_tightness", 0.1)
                        })
                # Handle other types if necessary

            # Perform ray tracing
            emission_tensor = ray_tracer.trace(
                light_sources=formatted_light_sources,
                wavelengths=wavelengths,
                time_slices=time_slices,
                extended_sources=extended_sources
            )

            return emission_tensor

        def get_sensor_resolution(self):
            # Define sensor resolution (can be made configurable)
            return (512, 512)

class Lens:
    def process(self, emission_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dummy lens processing for emission tensor.

        Args:
            emission_tensor (torch.Tensor): Emission tensor.

        Returns:
            torch.Tensor: Processed emission tensor.
        """
        # Placeholder: Apply lens effects here if needed
        return emission_tensor

class Aperture:
    def process(self, emission_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dummy aperture processing for emission tensor.

        Args:
            emission_tensor (torch.Tensor): Emission tensor.

        Returns:
            torch.Tensor: Processed emission tensor.
        """
        # Placeholder: Apply aperture effects here if needed
        return emission_tensor

class Camera:
    def capture(self, emission_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process the emission tensor to generate a final image.

        Args:
            emission_tensor (torch.Tensor): Emission tensor.

        Returns:
            torch.Tensor: Final image tensor.
        """
        # Sum across the emission samples to simulate final image
        # Shape: [batch_size, height, width, channels] -> [channels, height, width]
        final_image = emission_tensor.sum(dim=0).permute(2, 0, 1)
        return final_image

class LightProcessing:
    """
    A photonic discriminator processing class.
    Encapsulates the entire process of generating, processing, and gamifying light emission data.
    """

    def __init__(self):
        self.noise = self.Noise()
        self.lens = Lens()
        self.aperture = Aperture()
        self.camera = Camera()

    class Noise:
        def __init__(self):
            self.LIGHT_SPECTRA = {
                # [Existing LIGHT_SPECTRA definitions as shown above]
                # Ensure all entries have 'name', 'type', 'description', etc.
                # Example:
                "cheap_pen_light": {
                    "name": "cheap_pen_light",
                    "type": "laser",
                    "description": "Basic red laser pointer for general use.",
                    "wavelength": torch.tensor(650.0, dtype=torch.float64, requires_grad=True),
                    "std": torch.tensor(2.0, dtype=torch.float64, requires_grad=True),
                    "simulation": {
                        "envelope": {"attack": torch.tensor(0.1, dtype=torch.float64),
                                     "sustain": torch.tensor(0.8, dtype=torch.float64),
                                     "decay": torch.tensor(0.1, dtype=torch.float64)},
                        "error_factor": torch.tensor(0.05, dtype=torch.float64)
                    },
                    "size": torch.tensor(0.05, dtype=torch.float64),
                    "beam_tightness": torch.tensor(0.1, dtype=torch.float64)
                },
                "green_pointer": {
                    "name": "green_pointer",
                    "type": "laser",
                    "description": "Green laser pointer with higher visibility.",
                    "wavelength": torch.tensor(532.0, dtype=torch.float64, requires_grad=True),
                    "std": torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
                    "simulation": {
                        "envelope": {"attack": torch.tensor(0.2, dtype=torch.float64),
                                     "sustain": torch.tensor(0.6, dtype=torch.float64),
                                     "decay": torch.tensor(0.2, dtype=torch.float64)},
                        "error_factor": torch.tensor(0.04, dtype=torch.float64)
                    },
                    "size": torch.tensor(0.05, dtype=torch.float64),
                    "beam_tightness": torch.tensor(0.1, dtype=torch.float64)
                },
                "blue_pointer": {
                    "name": "blue_pointer",
                    "type": "laser",
                    "description": "Blue laser pointer for precise applications.",
                    "wavelength": torch.tensor(450.0, dtype=torch.float64, requires_grad=True),
                    "std": torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
                    "simulation": {
                        "envelope": {"attack": torch.tensor(0.15, dtype=torch.float64),
                                     "sustain": torch.tensor(0.7, dtype=torch.float64),
                                     "decay": torch.tensor(0.15, dtype=torch.float64)},
                        "error_factor": torch.tensor(0.03, dtype=torch.float64)
                    },
                    "size": torch.tensor(0.05, dtype=torch.float64),
                    "beam_tightness": torch.tensor(0.1, dtype=torch.float64)
                },
                # Add other light spectra as needed
            }

            self.LIGHT_SENSORS = {
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

            # Convert light spectra and sensors to tensors where applicable
            self.LIGHT_SPECTRA = self.convert_to_tensors(self.LIGHT_SPECTRA)
            self.LIGHT_SENSORS = self.convert_to_tensors(self.LIGHT_SENSORS)

        def convert_to_tensors(self, data):
            if isinstance(data, dict):
                return {k: self.convert_to_tensors(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self.convert_to_tensors(v) for v in data]
            elif isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, float):
                return torch.tensor(data, dtype=torch.float64, requires_grad=True)
            elif isinstance(data, str):
                return data
            elif isinstance(data, dict) and "wavelength" in data:
                # Handle bands for spectra
                return {k: self.convert_to_tensors(v) for k, v in data.items()}
            return data  # Return as is for unhandled types

        def normalize_envelope(self, attack: torch.Tensor, sustain: torch.Tensor, decay: torch.Tensor):
            total = attack + sustain + decay
            if torch.all(total == 0):
                raise ValueError("Envelope parameters cannot all be zero.")
            return attack / total, sustain / total, decay / total

        def generate_emission_tensor(
            self,
            light_sources: List[Dict[str, Any]],
            wavelengths: List[float],
            time_slices: List[float],
            extended_sources: bool = True,
            stencil_matrix: Optional[torch.Tensor] = None,
            grid_tuple: Optional[Tuple[float, float]] = None
        ) -> torch.Tensor:
            """
            Generate a tensor representing light emissions using RayTracingWrapper.

            Args:
                light_sources (list): List of light source configurations.
                wavelengths (list): List of wavelengths to trace.
                time_slices (list): Time slices for rendering.
                extended_sources (bool): Whether to simulate extended light sources.
                stencil_matrix (torch.Tensor, optional): Stencil matrix for gobo mask.
                grid_tuple (tuple, optional): (density, depth) for grid scrim effects.

            Returns:
                torch.Tensor: A tensor representing the simulated light features.
            """
            # Initialize RayTracingWrapper with optional stencil and grid parameters
            ray_tracer = RayTracingWrapper(
                resolution=self.get_sensor_resolution(),
                granularity=100,
                stencil_matrix=stencil_matrix,
                grid_tuple=grid_tuple
            )

            # Prepare light source configurations
            formatted_light_sources = []
            for source in light_sources:
                if source["type"] == "laser":
                    # For lasers, assume a single wavelength
                    formatted_light_sources.append({
                        "name": source["name"],
                        "position": [0, 0, 0],  # Assuming origin for simplicity; adjust as needed
                        "intensity": source["intensity"].item(),
                        "wavelength": source["wavelength"].item(),
                        "size": source.get("size", 0.05).item(),
                        "beam_tightness": source.get("beam_tightness", 0.1).item()
                    })
                elif source["type"] == "spectra":
                    # For spectra, iterate over bands
                    for band in source["bands"]:
                        formatted_light_sources.append({
                            "name": source["name"],
                            "position": [0, 0, 0],  # Adjust as needed
                            "intensity": band["intensity"],
                            "wavelength": band["wavelength"],
                            "size": source.get("size", 0.05).item(),
                            "beam_tightness": source.get("beam_tightness", 0.1).item()
                        })
                # Handle other types if necessary

            # Perform ray tracing
            emission_tensor = ray_tracer.trace(
                light_sources=formatted_light_sources,
                wavelengths=wavelengths,
                time_slices=time_slices,
                extended_sources=extended_sources
            )

            return emission_tensor

        def get_sensor_resolution(self):
            # Define sensor resolution (can be made configurable)
            return (512, 512)

    def main_demo():
        # Initialize the light processing system
        game = LightProcessing()

        # Define timeline events
        timeline = [
            TimelineEvent(
                source_name="cheap_pen_light",
                start_time=torch.tensor([0.0], dtype=torch.float64),
                duration=torch.tensor([10.0], dtype=torch.float64),
                intensity=torch.tensor([1.0], dtype=torch.float64),
                size=torch.tensor([0.05], dtype=torch.float64),
                beam_tightness=torch.tensor([0.1], dtype=torch.float64)
            ),
            TimelineEvent(
                source_name="green_pointer",
                start_time=torch.tensor([10.0], dtype=torch.float64),
                duration=torch.tensor([10.0], dtype=torch.float64),
                intensity=torch.tensor([0.8], dtype=torch.float64),
                size=torch.tensor([0.05], dtype=torch.float64),
                beam_tightness=torch.tensor([0.1], dtype=torch.float64)
            ),
            TimelineEvent(
                source_name="blue_pointer",
                start_time=torch.tensor([20.0], dtype=torch.float64),
                duration=torch.tensor([10.0], dtype=torch.float64),
                intensity=torch.tensor([0.6], dtype=torch.float64),
                size=torch.tensor([0.05], dtype=torch.float64),
                beam_tightness=torch.tensor([0.1], dtype=torch.float64)
            )
        ]

        # Parameters for testing
        image_size = (512, 512)  # Must match RayTracingWrapper resolution
        num_time_batches = 3  # Number of time slices
        num_categories = 5
        wavelengths = [450.0, 550.0, 650.0]  # Example wavelengths: Blue, Green, Red
        time_slices = [0.0, 1.0, 2.0]  # Example time slices

        # Optional: Define stencil matrix and grid scrim
        # Example: Simple circular stencil
        stencil_size = 100
        stencil_matrix_np = np.ones((stencil_size, stencil_size), dtype=np.float64)
        cv = int(stencil_size / 2)
        radius = int(stencil_size / 4)
        y, x = np.ogrid[:stencil_size, :stencil_size]
        mask = (x - cv) ** 2 + (y - cv) ** 2 <= radius ** 2
        stencil_matrix_np[mask] = 0.5  # Dim the central region
        stencil_matrix = torch.tensor(stencil_matrix_np, dtype=torch.float64)

        # Grid scrim parameters: density and depth
        grid_tuple = (10, 0.3)  # Example values

        # Generate emission tensor
        emission_tensor = game.noise.generate_emission_tensor(
            light_sources=[{
                "name": event.source_name,
                "type": event.type if "type" in event.__dict__ else "laser",
                "intensity": event.intensity,
                "size": event.size,
                "beam_tightness": event.beam_tightness
            } for event in timeline],
            wavelengths=wavelengths,
            time_slices=time_slices,
            extended_sources=True,
            stencil_matrix=stencil_matrix,
            grid_tuple=grid_tuple
        )

        # Process emission tensor through lens and aperture
        emission_tensor = game.lens.process(emission_tensor)
        emission_tensor = game.aperture.process(emission_tensor)

        # Capture final image
        final_image = game.camera.capture(emission_tensor)

        # Initialize results
        correct_losses = []
        incorrect_losses = []
        random_losses = []
        num_tests = 10  # Reduced for demonstration

        for _ in range(num_tests):
            # Correct answer
            gt_dense = emission_tensor.clone().detach()
            correct_prediction = gt_dense.clone()  # Correct prediction matches ground truth

            # Example: Generate predicted categories based on wavelength
            # Assuming wavelength is the second feature; adjust as needed
            gt_wavelengths = gt_dense[:, :, 1]  # [batch, height, width, channels] -> channels=RGB
            # Here, we might need to extract wavelength differently based on how data is structured
            # For demonstration, let's assume 'wavelength' is stored in the green channel
            gt_wavelengths = gt_dense[:, :, 1, 1]  # [batch, height, width]

            min_wavelength = gt_wavelengths.min()
            max_wavelength = gt_wavelengths.max()
            true_categories = torch.floor(
                ((gt_wavelengths - min_wavelength) / (max_wavelength - min_wavelength + 1e-6)) * num_categories
            ).long()
            true_categories = true_categories.view(-1)

            # Simulated predicted categories (one-hot)
            predicted_categories = torch.nn.functional.one_hot(true_categories, num_classes=num_categories).float()

            # Incorrect prediction (random one-hot)
            incorrect_indices = torch.randint(0, num_categories, (true_categories.size(0),))
            incorrect_categories = torch.nn.functional.one_hot(incorrect_indices, num_classes=num_categories).float()

            # Random prediction (random probabilities)
            random_categories = torch.rand(true_categories.size(0), num_categories)

            # Compute losses
            correct_loss = game.calculate_category_loss(emission_tensor, predicted_categories, num_categories)
            incorrect_loss = game.calculate_category_loss(emission_tensor, incorrect_categories, num_categories)
            random_loss = game.calculate_category_loss(emission_tensor, random_categories, num_categories)

            # Record losses
            correct_losses.append(correct_loss.item())
            incorrect_losses.append(incorrect_loss.item())
            random_losses.append(random_loss.item())

        # Calculate statistics
        stats = {
            "correct_mean_loss": torch.tensor(correct_losses).mean().item(),
            "incorrect_mean_loss": torch.tensor(incorrect_losses).mean().item(),
            "random_mean_loss": torch.tensor(random_losses).mean().item(),
            "correct_std_dev": torch.tensor(correct_losses).std().item(),
            "incorrect_std_dev": torch.tensor(incorrect_losses).std().item(),
            "random_std_dev": torch.tensor(random_losses).std().item()
        }

        # Print statistics
        print("Loss Statistics:")
        print(f"Correct Predictions: Mean={stats['correct_mean_loss']:.4f}, StdDev={stats['correct_std_dev']:.4f}")
        print(f"Incorrect Predictions: Mean={stats['incorrect_mean_loss']:.4f}, StdDev={stats['incorrect_std_dev']:.4f}")
        print(f"Random Predictions: Mean={stats['random_mean_loss']:.4f}, StdDev={stats['random_std_dev']:.4f}")

    if __name__ == "__main__":
        main_demo()
