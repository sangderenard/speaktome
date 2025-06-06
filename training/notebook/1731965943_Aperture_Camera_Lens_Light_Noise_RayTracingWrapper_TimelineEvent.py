import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import logging
from datetime import datetime

# Set up a logging configuration
logging.basicConfig(filename="emission_log.jsonl", level=logging.INFO, format="%(message)s")

def log_emission(emission_details: Dict[str, Any]):
    # Add a timestamp to the log
    emission_details["timestamp"] = datetime.utcnow().isoformat()
    # Serialize as JSON for structured logging
    logging.info(json.dumps(emission_details))
from dataclasses import dataclass, field
import torch

@dataclass
class TimelineEvent:
    source_name: str
    start_time: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0]))
    duration: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0]))
    intensity: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0]))
    a: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0]))
    s: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0]))
    d: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0]))
    size: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0]))
    beam_tightness: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0]))

    def __post_init__(self):
        # Automatically convert numeric inputs to tensors
        for field_name, value in self.__dict__.items():
            if isinstance(value, (int, float)):
                setattr(self, field_name, torch.tensor([value]))
            elif isinstance(value, torch.Tensor):
                continue  # Leave tensors unchanged
import mitsuba as mi
import torch

mi.set_variant("scalar_spectral")  # Enable spectral rendering for arbitrary wavelengths


class RayTracingWrapper:
    def __init__(self, resolution=(512, 512), granularity=100):
        """
        Initialize the RayTracingWrapper for handling ray tracing tasks.

        Args:
            resolution (tuple): Resolution of the output image (width, height).
            granularity (int): Number of rays per light source for point source summation.
        """
        self.resolution = resolution
        self.granularity = granularity

    def _simulate_point_source(self, position, intensity, wavelength, sensor_plane):
        """
        Simulate a point light source by summing the contributions of multiple rays.

        Args:
            position (list): 3D position of the light source.
            intensity (float): Intensity of the light source.
            wavelength (float): Wavelength of the light source.
            sensor_plane (mitsuba.Scene): The sensor plane scene.

        Returns:
            torch.Tensor: Simulated light projection as a tensor.
        """
        total_image = torch.zeros((*self.resolution, 3), dtype=torch.float64)  # RGB tensor

        for _ in range(self.granularity):
            # Random jitter around the light position for granularity
            jittered_position = [
                position[0] + torch.normal(mean=0.0, std=0.01).item(),
                position[1] + torch.normal(mean=0.0, std=0.01).item(),
                position[2]
            ]

            # Create light source for the ray
            light = mi.load_dict({
                "type": "point",
                "position": jittered_position,
                "intensity": {"type": "spectrum", "value": wavelength}
            })

            # Add light to the scene
            sensor_plane.add(light)

            # Perform ray tracing
            integrator = mi.load_dict({"type": "path"})
            image = mi.render(sensor_plane, integrator=integrator)

            # Accumulate contributions
            total_image += torch.tensor(image, dtype=torch.float64)

        # Average over granularity
        return total_image / self.granularity

    def _simulate_extended_source(self, position, size, intensity, wavelength, sensor_plane):
        """
        Simulate an extended light source by distributing rays across the source surface.

        Args:
            position (list): 3D position of the light source.
            size (float): Radius of the extended light source.
            intensity (float): Intensity of the light source.
            wavelength (float): Wavelength of the light source.
            sensor_plane (mitsuba.Scene): The sensor plane scene.

        Returns:
            torch.Tensor: Simulated light projection as a tensor.
        """
        total_image = torch.zeros((*self.resolution, 3), dtype=torch.float64)  # RGB tensor

        for _ in range(self.granularity):
            # Random position within the light source radius
            angle = torch.rand(1).item() * 2 * torch.pi
            r = size * torch.sqrt(torch.rand(1)).item()
            jittered_position = [
                position[0] + r * torch.cos(angle).item(),
                position[1] + r * torch.sin(angle).item(),
                position[2]
            ]

            # Create light source for the ray
            light = mi.load_dict({
                "type": "point",
                "position": jittered_position,
                "intensity": {"type": "spectrum", "value": wavelength}
            })

            # Add light to the scene
            sensor_plane.add(light)

            # Perform ray tracing
            integrator = mi.load_dict({"type": "path"})
            image = mi.render(sensor_plane, integrator=integrator)

            # Accumulate contributions
            total_image += torch.tensor(image, dtype=torch.float64)

        # Average over granularity
        return total_image / self.granularity

    def trace(self, light_sources, wavelengths, time_slices, extended_sources=True):
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

        for light_source in light_sources:
            for wavelength in wavelengths:
                for time_slice in time_slices:
                    # Define sensor plane
                    sensor_plane = mi.Scene()

                    if extended_sources:
                        # Simulate extended source
                        image = self._simulate_extended_source(
                            position=light_source["position"],
                            size=light_source["size"],
                            intensity=light_source["intensity"],
                            wavelength=wavelength,
                            sensor_plane=sensor_plane
                        )
                    else:
                        # Simulate point source
                        image = self._simulate_point_source(
                            position=light_source["position"],
                            intensity=light_source["intensity"],
                            wavelength=wavelength,
                            sensor_plane=sensor_plane
                        )

                    results.append(image)

        # Stack results into a tensor
        return torch.stack(results, dim=0)

class Light:
    """
    A thread-locked photonic discriminator processing class.
    Encapsulates the entire process of generating, processing, and gamifying light emission data.
    """

    class Noise:
        def __init__(self):
            self.LIGHT_SPECTRA = {
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

            def convert_to_tensors(data):
                if isinstance(data, dict):
                    return {k: convert_to_tensors(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [convert_to_tensors(v) for v in data]
                elif isinstance(data, float):
                    return torch.tensor(data, requires_grad=True)
                elif isinstance(data, str):
                    return data
                return torch.tensor(float(data), requires_grad=True)

            self.LIGHT_SPECTRA = convert_to_tensors(self.LIGHT_SPECTRA)
            self.LIGHT_SENSORS = convert_to_tensors(self.LIGHT_SENSORS)

        def normalize_envelope(self, attack, sustain, decay):
            total = attack + sustain + decay
            if total == 0:
                raise ValueError("Envelope parameters cannot all be zero.")
            return attack / total, sustain / total, decay / total
        def generate_emission_tensor(light_sources, batch_size, num_features=10, dtype=torch.float64):
            """
            Generate a tensor representing light emissions in a batch-feature format.

            Args:
                light_sources (list): List of light source properties.
                batch_size (int): Number of emissions in the batch.
                num_features (int): Number of features per emission.
                dtype: Data type for the tensor (default: torch.float64).

            Returns:
                torch.Tensor: A tensor with shape [batch_size, num_features].
            """
            # Initialize an empty tensor for emissions
            emission_tensor = torch.zeros((batch_size, num_features), dtype=dtype, requires_grad=True)

            # Populate the tensor with emissions
            for batch_idx in range(batch_size):
                source = light_sources[batch_idx % len(light_sources)]  # Cycle through sources if needed

                # Source-specific properties
                intensity = torch.tensor(source.get("intensity", 1.0), dtype=dtype)
                wavelength = torch.tensor(source.get("wavelength", 650.0), dtype=dtype)
                beam_tightness = torch.tensor(source.get("beam_tightness", 1.0), dtype=dtype)

                # Spatial properties
                origin_x = torch.normal(mean=torch.tensor(0.5, dtype=dtype), std=beam_tightness)
                origin_y = torch.normal(mean=torch.tensor(0.5, dtype=dtype), std=beam_tightness)
                dest_x = torch.normal(mean=origin_x, std=beam_tightness)
                dest_y = torch.normal(mean=origin_y, std=beam_tightness)

                # Temporal and envelope features
                attack = torch.tensor(source["simulation"]["envelope"]["attack"], dtype=dtype)
                sustain = torch.tensor(source["simulation"]["envelope"]["sustain"], dtype=dtype)
                decay = torch.tensor(source["simulation"]["envelope"]["decay"], dtype=dtype)

                # Combine features into a tensor
                feature_vector = torch.stack([
                    intensity, wavelength, origin_x, origin_y, dest_x, dest_y,
                    attack, sustain, decay, beam_tightness
                ])

                # Assign to the batch tensor
                emission_tensor[batch_idx] = feature_vector

            return emission_tensor

        def generate_emission_tensor(
            self, light_sources, image_size, num_time_batches, wavelength_range=(380, 780)
        ):
            width, height = image_size
            emissions = []

            for time_idx in range(num_time_batches):
                for source_idx, source in enumerate(light_sources):
                    source_name = source["name"]
                    properties = self.LIGHT_SPECTRA[source_name]
                    wavelength = properties["wavelength"]
                    std = properties["std"]
                    intensity = source.get("intensity", torch.tensor([1.0]))
                    beam_tightness = properties.get("beam_tightness", torch.tensor([10.0]))

                    # Normalize envelope
                    envelope = properties["simulation"]["envelope"]
                    a, s, d = self.normalize_envelope(
                        envelope["attack"], envelope["sustain"], envelope["decay"]
                    )

                    # Temporal envelope scaling
                    total_duration = a + s + d
                    time_ratio = time_idx / max(1, num_time_batches - 1)
                    if time_ratio <= a:
                        std_scale = time_ratio / a
                    elif time_ratio <= a + s:
                        std_scale = torch.tensor([1.0])
                    else:
                        decay_ratio = (time_ratio - a - s) / d
                        std_scale = torch.max(torch.tensor([0.0]), torch.tensor([1.0]) - decay_ratio)

                    scaled_std = std * std_scale

                    # Generate emissions
                    emissions.extend(self._generate_emissions_for_source(
                        source_idx, width, height, intensity, beam_tightness, wavelength, scaled_std, (a,s,d)
                    ))

            # Convert to sparse tensor
            emissions = torch.tensor(emissions, dtype=torch.float32)
            indices = emissions[:, :3].T.long()  # Use first 3 columns as indices
            values = emissions[:, 3:]  # Use remaining columns as features
            size = (num_time_batches, len(light_sources), len(emissions), values.size(1))
            return torch.sparse_coo_tensor(indices, values, size=size)

        def _generate_emissions_for_source(self, source_idx, width, height, intensity, beam_tightness, wavelength, scaled_std, asd):
            a, s, d = asd
            emissions = []
            for _ in range(int(intensity * 1000)):
                origin_x = torch.normal(mean=width / 2, std=beam_tightness / 2)
                origin_y = torch.normal(mean=height / 2, std=beam_tightness / 2)
                arrival_x = torch.normal(mean=origin_x, std=beam_tightness / 2)
                arrival_y = torch.normal(mean=origin_y, std=beam_tightness / 2)
                actual_wavelength = wavelength + torch.normal(torch.tensor([0.0]), scaled_std)
                emissions.append([
                    source_idx,  # Source ID
                    origin_x, origin_y, arrival_x, arrival_y, actual_wavelength
                ])
                def log_emission_details(source_name, origin, destination, wavelength, intensity, envelope):
                    emission_details = {
                        "source_name": source_name,
                        "origin": {"x": origin[0], "y": origin[1]},
                        "destination": {"x": destination[0], "y": destination[1]},
                        "wavelength": wavelength,
                        "intensity": intensity,
                        "envelope": envelope,
                    }
                    log_emission(emission_details)
                log_emission_details(
                    source_name=source_idx,
                    origin=(origin_x.item(), origin_y.item()),
                    destination=(arrival_x.item(), arrival_y.item()),
                    wavelength=actual_wavelength.item(),
                    intensity=intensity.item(),
                    envelope={"attack": a.item(), "sustain": s.item(), "decay": d.item()}
                )
            return emissions

        def calculate_loss(self, emission_tensor, predicted_tensor, dims=None, reduction='mean'):
            dense_gt = emission_tensor.to_dense()
            dense_pred = predicted_tensor.to_dense()

            # Sum or reduce loss across specific dimensions
            if dims:
                loss = (dense_gt - dense_pred).abs() if reduction == 'sum' else (dense_gt - dense_pred).pow(2)
                loss = loss.sum(dim=dims) if reduction == 'sum' else loss.mean(dim=dims)
            else:
                loss = torch.nn.functional.mse_loss(dense_pred, dense_gt)
            return loss
        def calculate_category_loss(self, emission_tensor, predicted_categories, num_categories):
            """
            Calculates categorical loss between emission timeline data and predicted categories.

            Args:
                emission_tensor (torch.sparse.FloatTensor): Ground truth emission tensor.
                predicted_categories (torch.Tensor): Predicted category probabilities (N x num_categories).
                num_categories (int): Total number of categories.

            Returns:
                torch.Tensor: Category loss.
            """
            dense_gt = emission_tensor.to_dense()

            # Generate true categories for emission data
            # For demonstration, assign categories based on wavelength ranges
            gt_wavelengths = dense_gt[..., -2]  # Assuming wavelength is stored in the second last column
            true_categories = torch.floor((gt_wavelengths - gt_wavelengths.min()) /
                                        (gt_wavelengths.max() - gt_wavelengths.min()) * num_categories).long()

            # Flatten tensors to compare batch-wise
            true_categories = true_categories.view(-1)
            predicted_categories = predicted_categories.view(-1, num_categories)

            # Use Cross-Entropy Loss for category comparison
            loss_fn = torch.nn.CrossEntropyLoss()
            category_loss = loss_fn(predicted_categories, true_categories)
            return category_loss

        def generate_image_from_timeline(self, timeline, image_size, num_time_batches):
            light_sources = [{"name": event.source_name, "intensity": event.intensity} for event in timeline]
            emission_tensor = self.generate_emission_tensor(light_sources, image_size, num_time_batches)
            return self.camera.capture(emission_tensor)

        def compare_tensors(self, tensor1, tensor2, dims=None, reduction='mean'):
            return self.calculate_loss(tensor1, tensor2, dims, reduction)

    class Lens:
        def process(self, emission_tensor):
            """
            Dummy lens processing for emission tensor.
            """
            return emission_tensor

    class Aperture:
        def process(self, emission_tensor):
            """
            Dummy aperture processing for emission tensor.
            """
            return emission_tensor

    class Camera:
        def capture(self, emission_tensor):
            """
            Dummy camera processing for emission tensor.
            """
            dense_tensor = emission_tensor.to_dense()
            image = dense_tensor.sum(dim=(0, 1, 2))
            return image

    def __init__(self):
        self.noise = self.Noise()
        self.lens = self.Lens()
        self.aperture = self.Aperture()
        self.camera = self.Camera()

    def process_game_tick(self, light_sources, image_size, num_time_batches):
        """
        Simulates a single game tick by generating and processing emission data.

        Args:
            light_sources (list): List of light sources.
            image_size (tuple): Size of the image (width, height).
            num_time_batches (int): Number of time steps.

        Returns:
            torch.Tensor: Final image tensor.
        """
        emission_tensor = self.noise3.generate_emission_tensor(light_sources, image_size, num_time_batches)
        emission_tensor = self.lens.process(emission_tensor)
        emission_tensor = self.aperture.process(emission_tensor)
        final_image = self.camera.capture(emission_tensor)
        return final_image

    def calculate_loss(self, emission_tensor, predicted_parameters):
        """
        Calculates the loss between the original emission tensor and predicted parameters.

        Args:
            emission_tensor (torch.sparse.FloatTensor): Ground truth emission tensor.
            predicted_parameters (torch.Tensor): Model predictions.

        Returns:
            torch.Tensor: Calculated loss.
        """
        dense_gt = emission_tensor.to_dense()
        loss = torch.nn.functional.mse_loss(predicted_parameters, dense_gt)
        return loss

import random

def main_demo():
    # Initialize the game
    game = Light()

    # Define timeline events
    timeline = [
        TimelineEvent(source_name="cheap_pen_light", start_time=torch.tensor([0]), duration=torch.tensor([10]), intensity=torch.tensor([1.0])),
        TimelineEvent(source_name="green_pointer", start_time=torch.tensor([10]), duration=torch.tensor([10]), intensity=torch.tensor([0.8])),
        TimelineEvent(source_name="blue_pointer", start_time=torch.tensor([20]), duration=torch.tensor([10]), intensity=torch.tensor([0.6]))
    ]

    # Parameters for testing
    image_size = (64, 64)
    num_time_batches = 50
    num_categories = 5
    num_tests = 100

    # Generate emission tensor
    emission_tensor = game.noise.generate_emission_tensor(
        [{"name": e.source_name, "intensity": e.intensity} for e in timeline],
        image_size,
        num_time_batches
    )

    # Initialize results
    correct_losses = []
    incorrect_losses = []
    random_losses = []

    for _ in range(num_tests):
        # Correct answer
        gt_dense = emission_tensor.to_dense()
        correct_prediction = gt_dense.clone()  # Correct prediction matches ground truth
        correct_categories = torch.nn.functional.one_hot(
            torch.floor(gt_dense[..., -2] / (gt_dense[..., -2].max() + 1e-6) * num_categories).long(),
            num_classes=num_categories
        ).float()

        # Incorrect answer
        incorrect_prediction = correct_prediction.clone()
        incorrect_prediction *= torch.rand_like(incorrect_prediction)  # Add randomness

        # Random prediction
        random_prediction = torch.rand_like(correct_prediction)  # Totally random

        # Compute losses
        correct_loss = game.calculate_category_loss(emission_tensor, correct_categories, num_categories)
        incorrect_loss = game.calculate_category_loss(emission_tensor, incorrect_prediction, num_categories)
        random_loss = game.calculate_category_loss(emission_tensor, random_prediction, num_categories)

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
    print(f"Correct Predictions: Mean={stats['correct_mean_loss']}, StdDev={stats['correct_std_dev']}")
    print(f"Incorrect Predictions: Mean={stats['incorrect_mean_loss']}, StdDev={stats['incorrect_std_dev']}")
    print(f"Random Predictions: Mean={stats['random_mean_loss']}, StdDev={stats['random_std_dev']}")

if __name__ == "__main__":
    main_demo()