# Prometheus.py
from PIL import Image
import os
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import mitsuba as mi

# Initialize Mitsuba in spectral mode
mi.set_variant("scalar_spectral")  # Enable spectral rendering for arbitrary wavelengths
from mitsuba import Bitmap, Properties, Struct, ScalarVector2u, PyObjectWrapper

# Set up a logging configuration
logging.basicConfig(filename="emission_log.jsonl", level=logging.INFO, format="%(message)s")


def log_emission(emission_details: Dict[str, Any]):
    """
    Logs emission details with a timestamp.
    """
    emission_details["timestamp"] = datetime.utcnow().isoformat()
    logging.info(json.dumps(emission_details))


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
                setattr(self, field_name, torch.tensor([value], dtype=torch.float64))
            elif isinstance(value, torch.Tensor):
                continue  # Leave tensors unchanged

import numpy as np
import torch
import mitsuba as mi

# Initialize Mitsuba with the appropriate variant
# For spectral mode, you might need to use a spectral variant like 'scalar_spectral'
# For RGB mode, 'scalar_rgb' is appropriate
# The variant must be set before creating the class instances
# Example:
# mi.set_variant('scalar_rgb')
# mi.set_variant('scalar_spectral')  # Uncomment if using spectral mode

class FocusedLightSource:
    """
    A class to generate rays representing a focused light source with configurable beam width,
    Gaussian falloff and source errors, supporting both RGB and Spectral modes. Utilizes
    PyTorch for efficient batch generation of large ray collections.
    """
    
    def __init__(self, origin, direction, beam_width, num_rays,
                 gaussian_falloff_error=0.0,
                 gaussian_source_error=0.0,
                 mode='RGB',
                 wavelengths=None):
        """
        Initialize the FocusedLightSource.

        Parameters:
        - origin (list or tuple of 3 floats): The origin point (x, y, z) of the light source.
        - direction (list or tuple of 3 floats): The primary direction vector of the light source.
        - beam_width (float): Standard deviation of the Gaussian beam spread (radians).
                              Smaller values simulate tighter beams (e.g., lasers), while larger
                              values simulate more diffuse light sources.
        - num_rays (int): Number of rays to generate.
        - gaussian_falloff_error (float): Standard deviation for Gaussian perturbations in ray directions.
        - gaussian_source_error (float): Standard deviation for Gaussian perturbations in ray origins.
        - mode (str): Emission mode. Options:
                      - 'RGB': Each ray has RGB color.
                      - 'Spectral': Each ray has a single float wavelength.
        - wavelengths (torch.Tensor, optional): Tensor of shape (num_rays,) containing wavelengths for spectral mode.
                                               Required if mode is 'Spectral'.
        """
        self.origin = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).repeat(num_rays, 1)  # Shape: (num_rays, 3)
        self.direction = self._normalize(torch.tensor(direction, dtype=torch.float32))  # Shape: (3,)
        self.beam_width = beam_width
        self.num_rays = num_rays
        self.gaussian_falloff_error = gaussian_falloff_error
        self.gaussian_source_error = gaussian_source_error
        self.mode = mode.lower()
        
        # Validate mode
        valid_modes = ['rgb', 'spectral']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {valid_modes}")
        
        # Handle wavelengths for spectral mode
        if self.mode == 'spectral':
            if wavelengths is None:
                raise ValueError("Wavelengths must be provided for spectral mode.")
            if not isinstance(wavelengths, torch.Tensor):
                raise TypeError("Wavelengths must be a torch.Tensor.")
            if wavelengths.shape[0] != num_rays:
                raise ValueError("Length of wavelengths must match num_rays.")
            self.wavelengths = wavelengths.float()  # Shape: (num_rays,)
        else:
            # For RGB mode, you can set a default color or allow passing colors as a parameter
            # Here, we'll assume white light for simplicity
            self.colors = torch.ones(num_rays, 3, dtype=torch.float32)  # Shape: (num_rays, 3)
    
    def _normalize(self, vec):
        """Normalize a 3D vector."""
        norm = torch.norm(vec)
        if norm == 0:
            raise ValueError("Zero vector cannot be normalized.")
        return vec / norm
    
    def _normalize_vectors(self, vectors):
        """
        Normalize an array of 3D vectors.

        Parameters:
        - vectors (torch.Tensor): Tensor of shape (N, 3).

        Returns:
        - normalized_vectors (torch.Tensor): Tensor of shape (N, 3).
        """
        norms = torch.norm(vectors, dim=1, keepdim=True)  # Shape: (N, 1)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # Prevent division by zero
        return vectors / norms
    
    def generate_rays(self):
        """
        Generate a list of Mitsuba Ray3f objects based on the initialized parameters.

        Returns:
        - rays (list of mi.Ray3f): List containing generated rays.
        """
        # Generate Gaussian-distributed angles for beam spread
        # Assuming small angles, we can approximate the direction perturbations
        theta = torch.normal(mean=0.0, std=self.beam_width, size=(self.num_rays,))  # Shape: (num_rays,)
        phi = torch.rand(self.num_rays) * 2 * np.pi  # Uniformly distributed between 0 and 2pi
        
        # Convert spherical coordinates to Cartesian coordinates for direction perturbations
        delta_x = theta * torch.cos(phi)
        delta_y = theta * torch.sin(phi)
        delta_z = -torch.ones(self.num_rays)  # Directed towards negative z-axis (modifiable)
        
        # Combine perturbations into direction vectors
        directions = torch.stack((delta_x, delta_y, delta_z), dim=1)  # Shape: (num_rays, 3)
        directions = self._normalize_vectors(directions)  # Normalize directions
        
        # Apply Gaussian falloff error if specified
        if self.gaussian_falloff_error > 0.0:
            falloff_perturbations = torch.normal(
                mean=0.0,
                std=self.gaussian_falloff_error,
                size=directions.shape
            )
            directions += falloff_perturbations
            directions = self._normalize_vectors(directions)
        
        # Apply Gaussian source error if specified
        if self.gaussian_source_error > 0.0:
            origin_perturbations = torch.normal(
                mean=0.0,
                std=self.gaussian_source_error,
                size=self.origin.shape
            )
            origins = self.origin + origin_perturbations
        else:
            origins = self.origin.clone()  # Shape: (num_rays, 3)
        
        # Prepare colors or wavelengths based on mode
        if self.mode == 'spectral':
            # Spectral mode: Each ray has a single float wavelength
            # Mitsuba expects wavelengths in meters, so convert nanometers to meters if necessary
            # Here, we assume the input wavelengths are in meters
            # If in nanometers, uncomment the following line:
            # wavelengths_m = self.wavelengths * 1e-9  # Convert nm to meters
            wavelengths_m = self.wavelengths  # Assuming already in meters
        else:
            # RGB mode: Each ray has an RGB color
            colors = self.colors  # Shape: (num_rays, 3)
        
        # Convert torch tensors to numpy for Mitsuba
        origins_np = origins.numpy()  # Shape: (num_rays, 3)
        directions_np = directions.numpy()  # Shape: (num_rays, 3)
        
        # Create Mitsuba Ray3f objects in batch
        rays = []
        if self.mode == 'spectral':
            # Spectral Mode
            for i in range(self.num_rays):
                ray = mi.Ray3f(
                    o=mi.Point3f(*origins_np[i]),
                    d=mi.Vector3f(*directions_np[i]),
                    time=0.0,
                    wavelengths=mi.Float(wavelengths_m[i].item())
                )
                rays.append(ray)
        else:
            # RGB Mode
            for i in range(self.num_rays):
                ray = mi.Ray3f(
                    o=mi.Point3f(*origins_np[i]),
                    d=mi.Vector3f(*directions_np[i]),
                    time=0.0,
                    wavelengths=mi.Color3f(*colors[i].tolist())  # Assign RGB color
                )
                rays.append(ray)
        
        return rays

    @staticmethod
    def from_rgb(origin, direction, beam_width, num_rays,
                gaussian_falloff_error=0.0,
                gaussian_source_error=0.0,
                colors=None):
        """
        Alternative constructor for RGB mode with customizable colors.

        Parameters:
        - colors (torch.Tensor, optional): Tensor of shape (num_rays, 3) containing RGB colors.
                                           If None, defaults to white light.
        """
        if colors is not None:
            if not isinstance(colors, torch.Tensor):
                raise TypeError("Colors must be a torch.Tensor.")
            if colors.shape != (num_rays, 3):
                raise ValueError("Colors tensor must have shape (num_rays, 3).")
        else:
            colors = torch.ones(num_rays, 3, dtype=torch.float32)  # Default to white light
        
        obj = FocusedLightSource(
            origin=origin,
            direction=direction,
            beam_width=beam_width,
            num_rays=num_rays,
            gaussian_falloff_error=gaussian_falloff_error,
            gaussian_source_error=gaussian_source_error,
            mode='RGB'
        )
        obj.colors = colors  # Override default colors if provided
        return obj


class RayTracingWrapper:
    def __init__(
        self,
        resolution: Tuple[int, int] = (512, 512),
        granularity: int = 100000,
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

    def _create_sampling_plane(self, position=(0, 0, 0), size=(1, 1)):
        """
        Create a physical sampling plane for ray interactions.

        Args:
            position (tuple): Center position of the sampling plane.
            size (tuple): Dimensions of the sampling plane.

        Returns:
            mitsuba.Shape: A rectangle for ray interaction sampling.
        """
        plane_dict = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate(position) @ mi.ScalarTransform4f.scale(mi.Vector3f(size[0], size[1], 1.0)),
        }
        return mi.load_dict(plane_dict)


    def generate_gaussian_rays(self, origin, num_rays=1000, std_dev=0.1):
        """
        Generate rays from a Gaussian distribution around the origin, directed towards the sampling plane.

        Args:
            origin (tuple): Origin of the rays [x, y, z].
            num_rays (int): Number of rays to generate.
            std_dev (float): Standard deviation for the Gaussian spread.

        Returns:
            List[mitsuba.Ray3f]: Generated rays.
        """
        rays = []
        for _ in range(num_rays):
            offset = np.random.normal(loc=0.0, scale=std_dev, size=2)  # Only x and y deviations
            direction = mi.Vector3f(offset[0], offset[1], -1.0)  # Directed towards negative z-axis
            #direction = mi.normalize(direction)  # Normalize to unit vector
            rays.append(
                mi.Ray3f(
                    o=mi.Point3f(*origin),  # Origin point
                    d=direction,            # Direction vector
                    time=0.0                # Time parameter (optional)
                )
            )
        return rays

    def trace_rays(self, scene, rays, wavelength=None):
        """
        Trace rays and record intersection data as tensors, including the angle between
        the surface normal and ray direction.

        Args:
            scene (mitsuba.Scene): Scene containing the sampling plane.
            rays (List[mitsuba.Ray3f]): Rays to trace.
            wavelength (float, optional): Wavelength associated with these rays.

        Returns:
            List[Dict[str, torch.Tensor]]: Intersection data as tensors.
        """
        results = []
        for ray in rays:
            si = scene.ray_intersect(ray)
            if si.is_valid():
                # Convert all relevant data to torch.Tensor
                normal = torch.tensor(si.sh_frame.n, dtype=torch.float64)
                d = torch.tensor(ray.d, dtype=torch.float64)
                o = torch.tensor(ray.o, dtype=torch.float64)
                p = torch.tensor(si.p, dtype=torch.float64) if hasattr(si, 'p') else torch.zeros(3, dtype=torch.float64)

                # Compute the angle between the ray direction and the surface normal
                direction_normalized = d / torch.norm(d)
                normal_normalized = normal / torch.norm(normal)
                dot_product = torch.clamp(torch.dot(direction_normalized, normal_normalized), -1.0, 1.0)
                angle = torch.acos(dot_product)
                distance = d - o

                # Append the results
                results.append({
                    "normal": normal,
                    "direction": d,
                    "origin": o,
                    "position": p,
                    "distance": distance,
                    "wavelength": torch.tensor(wavelength, dtype=torch.float64) if wavelength is not None else torch.tensor(0.0),
                    "angle": angle,  # Include the computed angle
                })
            else:
                print("No valid intersection.")  # Debugging statement
        return results



    def accumulate_exposure(self, scene, rays, num_passes=10, wavelength=None):
        """
        Simulate progressive exposure by accumulating ray interactions.

        Args:
            scene (mitsuba.Scene): Scene containing the sampling plane.
            rays (List[mitsuba.Ray3f]): Rays to trace.
            num_passes (int): Number of exposure passes.
            wavelength (float, optional): Wavelength associated with these rays.

        Returns:
            List[Dict]: Accumulated intersection data.
        """
        exposure_data = []
        for pass_num in range(num_passes):
            pass_results = self.trace_rays(scene, rays, wavelength=wavelength)
            exposure_data.extend(pass_results)
            print(f"Completed pass {pass_num + 1}/{num_passes}")
        return exposure_data

    def trace(
        self,
        light_sources: List[Dict[str, Any]],
        wavelengths: List[float],
        time_slices: List[float],
        extended_sources: bool = True
    ) -> List[Dict]:
        """
        Perform ray tracing and accumulate interactions.

        Args:
            light_sources (list): List of light source configurations.
            wavelengths (list): List of wavelengths to trace.
            time_slices (list): List of time slices for rendering.
            extended_sources (bool): Whether to simulate extended light sources.

        Returns:
            List[Dict]: Accumulated intersection data.
        """
        print(f"Tracing scene with {len(light_sources)} light sources and {len(wavelengths)} wavelengths.")
        all_results = []

        for light_source in light_sources:
            for wavelength in wavelengths:
                print(f"Processing light source at wavelength {wavelength}")
                # Define the initial scene dictionary with the sampling plane
                sampling_plane = self._create_sampling_plane(position=(0, 0, 0), size=(1, 1))
                scene_dict = {
                    "type": "scene",
                    "shape": sampling_plane,  # Use a single shape instead of a list
                    "emitter": {
                        "type": "point",
                        "position": [float(x.item()) for x in light_source["position"]],
                        "intensity": {"type": "spectrum", "value": 1.0},
                    },
                }


                # Apply stencil effects to the scene
                #scene_dict = self._apply_stencil(scene_dict)
                print("Scene dictionary configured.")

                # Create the scene
                scene = mi.load_dict(scene_dict)
                print("Scene loaded.")

                # Generate Gaussian-distributed rays from the light source
                rays = self.generate_gaussian_rays(origin=light_source["position"].tolist(), num_rays=self.granularity, std_dev=0.1)
                print(f"Generated {len(rays)} Gaussian rays.")

                # Accumulate exposure over multiple passes
                interaction_data = self.accumulate_exposure(scene, rays, num_passes=5, wavelength=wavelength)  # Adjust num_passes as needed
                print(f"Accumulated {len(interaction_data)} interactions.")

                all_results.extend(interaction_data)

        print(f"Total interactions collected: {len(all_results)}")
        return all_results

    def _apply_stencil(self, scene_dict):
        """
        Add stencil effects to the scene dictionary.

        Args:
            scene_dict (dict): The Mitsuba scene dictionary.

        Returns:
            dict: Updated scene dictionary with stencil effects.
        """
        if self.stencil_matrix is not None:
            stencil_image = self.stencil_matrix.numpy().astype(np.float32)

            # Flatten the stencil data and normalize to [0, 1]
            stencil_data = stencil_image.flatten() / stencil_image.max()

            # Define a raw bitmap texture for the stencil
            stencil_texture = {
                "type": "bitmap",
                "raw": {
                    "width": stencil_image.shape[1],
                    "height": stencil_image.shape[0],
                    "channel_count": 1,  # Single channel (grayscale)
                    "data": stencil_data.tolist(),
                },
                "wrap_mode_u": "clamp",
                "wrap_mode_v": "clamp"
            }

            # Add the stencil plane to the scene
            scene_dict["stencil_plane"] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f.translate([0, 0, -1]),  # Place in front of the sampling plane
                "material": {
                    "type": "diffuse",
                    "reflectance": stencil_texture,
                }
            }

        return scene_dict

    def get_sensor_resolution(self):
        # Define sensor resolution (can be made configurable)
        return self.resolution


class PhysicalSensor:
    def __init__(self, resolution: Tuple[int, int], position: Tuple[float, float, float]):
        """
        A sensor plane for full-scene light interaction.

        Args:
            resolution (tuple): Resolution of the sensor (width, height).
            position (tuple): Position of the sensor in the scene.
        """
        self.resolution = resolution
        self.position = position

    def create_sampling_plane(self):
        """
        Create a physical sampling plane capable of capturing all ray interactions.

        Returns:
            dict: Mitsuba-compatible sampling plane definition.
        """
        sensor_dict = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate(self.position) @ mi.ScalarTransform4f.scale((1, 1)),  # Adjust scale as needed
        }
        return mi.load_dict(sensor_dict)

    def trace_full_scene(self, emission_data: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Convert emission data into a structured tensor for processing.

        Args:
            emission_data (List[Dict[str, torch.Tensor]]): Accumulated intersection data.

        Returns:
            torch.Tensor: A tensor representation of the emission data.
        """
        data = []
        for entry in emission_data:
            # Ensure all values are scalars or 1D tensors
            position = entry.get("position", torch.zeros(3, dtype=torch.float64))
            normal = entry.get("normal", torch.zeros(3, dtype=torch.float64))
            direction = entry.get("direction", torch.zeros(3, dtype=torch.float64))
            wavelength = entry.get("wavelength", torch.tensor(0.0, dtype=torch.float64))
            angle = entry.get("angle", torch.tensor(0.0, dtype=torch.float64))
            origin = entry.get("origin", torch.zeros(3, dtype=torch.float64))

            # Compute distance as a scalar
            distance = torch.norm(position - origin)

            # Concatenate all features into a single 1D tensor
            entry_tensor = torch.cat([
                position.view(-1),  # Flatten position (x, y, z)
                normal.view(-1),    # Flatten normal (x, y, z)
                direction.view(-1), # Flatten direction (x, y, z)
                angle.view(-1),     # Angle as a scalar
                distance.view(-1),  # Distance as a scalar
                wavelength.view(-1) # Wavelength as a scalar
            ])
            data.append(entry_tensor)

        # Stack all entries into a 2D tensor
        emission_tensor = torch.stack(data, dim=0)
        return emission_tensor


class Energy:
    """
    A thread-locked photonic discriminator processing class.
    Encapsulates the entire process of generating, processing, and gamifying light emission data.
    """

    class Noise:
        def __init__(self):
            self.LIGHT_SPECTRA = {
                # Lasers and Spectra as defined earlier
                "cheap_pen_light": {
                    "gameid": "laser_red_pen",
                    "type": "laser",
                    "description": "Basic red laser pointer for general use.",
                    "wavelength": 650,
                    "std": 2,
                    "position": torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64),  # Position in space
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
                    "position": torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64),
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
                    "position": torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64),
                    "simulation": {
                        "envelope": {"attack": 0.15, "sustain": 0.7, "decay": 0.15},
                        "error_factor": 0.03
                    }
                },
                # Add other light sources as needed
            }

            self.LIGHT_SENSORS = {
                "photon_cascade_vacuum_tube": {
                    "sensor_id": "sensor_pcv",
                    "description": "Photon cascade vacuum tube with high sensitivity to single photons.",
                    "rgb_profile": {"r": 1.0, "g": 1.0, "b": 1.0},  # Linear response
                    "range": "all"
                },
                # Remove dummy sensors if focusing solely on the cascade sensor
                # "toy_digital_camera": { ... },
                # "advanced_nn_sensor": { ... }
                # Add more sensors as needed
            }

            # Convert light spectra and sensors to tensors where applicable
            self.LIGHT_SPECTRA = self.convert_to_tensors(self.LIGHT_SPECTRA)
            self.LIGHT_SENSORS = self.convert_to_tensors(self.LIGHT_SENSORS)

        def convert_to_tensors(self, data):
            """
            Recursively convert applicable parts of the data structure to tensors.

            Args:
                data: The data to convert.

            Returns:
                The data with tensors where appropriate.
            """
            if isinstance(data, dict):
                return {k: self.convert_to_tensors(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self.convert_to_tensors(v) for v in data]
            elif isinstance(data, float):
                return torch.tensor(data, dtype=torch.float64, requires_grad=True)
            elif isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, str):
                return data
            return torch.tensor(float(data), dtype=torch.float64, requires_grad=True)

        def normalize_envelope(self, attack: torch.Tensor, sustain: torch.Tensor, decay: torch.Tensor):
            """
            Normalize envelope parameters.

            Args:
                attack (torch.Tensor): Attack parameter.
                sustain (torch.Tensor): Sustain parameter.
                decay (torch.Tensor): Decay parameter.

            Returns:
                Tuple of normalized (attack, sustain, decay).
            """
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
        ) -> List[Dict]:
            """
            Generate a list of intersection data representing light emissions using RayTracingWrapper.

            Args:
                light_sources (list): List of light source configurations.
                wavelengths (list): List of wavelengths to trace.
                time_slices (list): List of time slices for rendering.
                extended_sources (bool): Whether to simulate extended light sources.
                stencil_matrix (torch.Tensor, optional): Stencil matrix for gobo mask.
                grid_tuple (tuple, optional): (density, depth) for grid scrim effects.

            Returns:
                List[Dict]: Accumulated intersection data.
            """
            # Initialize RayTracingWrapper with optional stencil and grid parameters
            ray_tracer = RayTracingWrapper(
                resolution=self.get_sensor_resolution(),
                granularity=10000,
                stencil_matrix=stencil_matrix,
                grid_tuple=grid_tuple
            )

            # Prepare light source configurations
            formatted_light_sources = []
            for source in light_sources:
                spectrum = self.LIGHT_SPECTRA[source["name"]]
                formatted_light_sources.append({
                    "position": spectrum["position"],
                    "intensity": source.get("intensity", 1.0),
                    "wavelength": spectrum["wavelength"],
                    "size": source.get("size", 0.05),
                    "beam_tightness": source.get("beam_tightness", 0.1),
                })

            # Perform ray tracing
            interaction_data = ray_tracer.trace(
                light_sources=formatted_light_sources,
                wavelengths=wavelengths,
                time_slices=time_slices,
                extended_sources=extended_sources
            )

            return interaction_data

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
    def __init__(self, output_dir="output_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sensors: Dict[str, PhysicalSensor] = {}  # Dictionary to manage sensors
        self.active_sensor: Optional[PhysicalSensor] = None  # Currently active sensor

    def add_sensor(self, sensor_name: str, sensor: PhysicalSensor):
        """
        Add a PhysicalSensor to the Camera.

        Args:
            sensor_name (str): Unique name for the sensor.
            sensor (PhysicalSensor): Instance of PhysicalSensor.
        """
        self.sensors[sensor_name] = sensor
        logging.info(f"Sensor '{sensor_name}' added to Camera.")
        print(f"Sensor '{sensor_name}' added to Camera.")  # Debug Statement

    def select_sensor(self, sensor_name: str):
        """
        Select an active sensor by name.

        Args:
            sensor_name (str): Name of the sensor to activate.
        """
        if sensor_name not in self.sensors:
            raise ValueError(f"Sensor '{sensor_name}' not found in Camera.")
        self.active_sensor = self.sensors[sensor_name]
        logging.info(f"Sensor '{sensor_name}' is now active.")
        print(f"Sensor '{sensor_name}' is now active.")  # Debug Statement

    def remove_sensor(self, sensor_name: str):
        """
        Remove a sensor from the Camera.

        Args:
            sensor_name (str): Name of the sensor to remove.
        """
        if sensor_name in self.sensors:
            del self.sensors[sensor_name]
            logging.info(f"Sensor '{sensor_name}' removed from Camera.")
            print(f"Sensor '{sensor_name}' removed from Camera.")  # Debug Statement
            if self.active_sensor and self.active_sensor.position == self.sensors[sensor_name].position:
                self.active_sensor = None
                logging.info(f"Active sensor cleared as it was removed.")
                print(f"Active sensor cleared as it was removed.")  # Debug Statement
        else:
            raise ValueError(f"Sensor '{sensor_name}' not found in Camera.")

    def capture(self, emission_data: List[Dict[str, Any]], sensor_name: str, wavelengths: List[float]) -> torch.Tensor:
        """
        Process the emission data to generate and save final tensor using the specified sensor.

        Args:
            emission_data (List[Dict[str, Any]]): Accumulated intersection data.
            sensor_name (str): Name of the sensor to use.
            wavelengths (list): Wavelengths associated with the emission tensor.

        Returns:
            torch.Tensor: Emission tensor.
        """
        if sensor_name not in self.sensors:
            raise ValueError(f"Sensor '{sensor_name}' not found in Camera.")
        sensor = self.sensors[sensor_name]
        print(f"Capturing emission data with sensor '{sensor_name}'")  # Debug Statement
        # Use PhysicalSensor's trace_full_scene to process emission data
        emission_tensor = sensor.trace_full_scene(emission_data)
        self.save_tensor(emission_tensor, sensor_name)
        return emission_tensor

    def save_tensor(self, tensor: torch.Tensor, filename_prefix: str, normalize=True):
        """
        Save the emission tensor as a JSON file for further analysis.

        Args:
            tensor (torch.Tensor): Tensor to save.
            filename_prefix (str): Prefix for the output file name.
            normalize (bool): Whether to normalize tensor values.
        """
        # Convert tensor to CPU and detach
        tensor = tensor.cpu().detach()

        # Optionally normalize
        if normalize and tensor.numel() > 0:
            tensor = tensor - tensor.min()
            tensor = (tensor / tensor.max()) * 1.0  # Normalize to [0, 1]

        # Convert to list for JSON serialization
        data = tensor.tolist()

        # Save as JSON
        filepath = os.path.join(self.output_dir, f"{filename_prefix}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved tensor data: {filepath}")  # Debug Statement

    def save_image(self, tensor: torch.Tensor, filename_prefix: str, normalize=True):
        """
        Save a batch of tensors as image files.

        Args:
            tensor (torch.Tensor): Tensor to save, shape (B, H, W) or (B, H, W, C).
            filename_prefix (str): Prefix for the output file names.
            normalize (bool): Whether to normalize tensor values to [0, 255].
        """
        # Ensure the tensor is on the CPU and convert to numpy
        tensor = tensor.cpu().detach()

        # Normalize if needed
        if normalize and tensor.numel() > 0:
            tensor = tensor - tensor.min()  # Ensure all values are >= 0
            tensor = (tensor / tensor.max() * 255).clamp(0, 255).byte()

        # Handle batch of images
        if tensor.dim() == 3:  # Batch of grayscale images (B, H, W)
            for i, single_image in enumerate(tensor):
                np_array = single_image.numpy()
                image = Image.fromarray(np_array, mode="L")  # Grayscale mode
                filepath = os.path.join(self.output_dir, f"{filename_prefix}_{i}.png")
                image.save(filepath)
                print(f"Saved image: {filepath}")
        elif tensor.dim() == 4:  # Batch of RGB images (B, H, W, C)
            for i, single_image in enumerate(tensor):
                np_array = single_image.numpy()
                image = Image.fromarray(np_array, mode="RGB")  # RGB mode
                filepath = os.path.join(self.output_dir, f"{filename_prefix}_{i}.png")
                image.save(filepath)
                print(f"Saved image: {filepath}")
        else:
            raise ValueError(f"Unsupported tensor shape for saving images: {tensor.shape}")


class Light:
    """
    A photonic discriminator processing class.
    Encapsulates the entire process of generating, processing, and gamifying light emission data.
    """

    def __init__(self, energy: Energy):
        self.noise = energy.Noise()
        self.lens = Lens()
        self.aperture = Aperture()
        self.camera = Camera()

        # Initialize Camera and add only the photon cascade vacuum tube sensor
        cascade_sensor = PhysicalSensor(resolution=(512, 512), position=(0, 0, 0))
        self.camera.add_sensor("photon_cascade_vacuum_tube", cascade_sensor)

        # Set the cascade sensor as active
        self.camera.select_sensor("photon_cascade_vacuum_tube")

    def process_game_tick(
        self,
        timeline: List[TimelineEvent],
        wavelengths: List[float],
        time_slices: List[float],
        stencil_matrix: Optional[torch.Tensor] = None,
        grid_tuple: Optional[Tuple[float, float]] = None,
        extended_sources: bool = True
    ) -> torch.Tensor:
        """
        Simulates a single game tick by generating and processing emission data.

        Args:
            timeline (list): List of TimelineEvent objects.
            wavelengths (list): List of wavelengths to trace.
            time_slices (list): List of time slices for rendering.
            stencil_matrix (torch.Tensor, optional): Stencil matrix for gobo mask.
            grid_tuple (tuple, optional): (density, depth) for grid scrim effects.
            extended_sources (bool): Whether to simulate extended light sources.

        Returns:
            torch.Tensor: Final emission tensor.
        """
        print("Processing game tick...")  # Debug Statement
        # Convert timeline events to light source configurations
        light_sources = []
        for event in timeline:
            light_sources.append({
                "name": event.source_name,
                "intensity": event.intensity.item(),
                "size": event.size.item(),
                "beam_tightness": event.beam_tightness.item(),
            })
        print(f"Light sources configured: {light_sources}")  # Debug Statement

        # Generate emission data using ray tracing
        emission_data = self.noise.generate_emission_tensor(
            light_sources=light_sources,
            wavelengths=wavelengths,
            time_slices=time_slices,
            extended_sources=extended_sources,
            stencil_matrix=stencil_matrix,
            grid_tuple=grid_tuple
        )
        print(f"Emission data generated with {len(emission_data)} interactions.")  # Debug Statement

        # Convert emission data to tensor via camera.capture
        emission_tensor = self.camera.capture(emission_data, "photon_cascade_vacuum_tube", wavelengths)
        print(f"Emission tensor shape: {emission_tensor.shape}")  # Debug Statement

        # Optionally process emission tensor through lens and aperture
        emission_tensor = self.lens.process(emission_tensor)
        emission_tensor = self.aperture.process(emission_tensor)

        return emission_tensor

    def calculate_loss(self, emission_tensor: torch.Tensor, predicted_parameters: torch.Tensor):
        """
        Calculates the loss between the original emission tensor and predicted parameters.

        Args:
            emission_tensor (torch.Tensor): Ground truth emission tensor.
            predicted_parameters (torch.Tensor): Model predictions.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss = torch.nn.functional.mse_loss(predicted_parameters, emission_tensor)
        return loss

    def calculate_category_loss(
        self, emission_tensor: torch.Tensor, predicted_categories: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        """
        Calculates categorical loss between emission timeline data and predicted categories.

        Args:
            emission_tensor (torch.Tensor): Ground truth emission tensor.
            predicted_categories (torch.Tensor): Predicted category probabilities (N x num_categories).
            num_categories (int): Total number of categories.

        Returns:
            torch.Tensor: Category loss.
        """
        # Assign categories based on wavelength ranges
        gt_wavelengths = emission_tensor[:, 8]  # Assuming wavelength is the ninth feature
        min_wavelength = gt_wavelengths.min()
        max_wavelength = gt_wavelengths.max()
        true_categories = torch.floor(
            ((gt_wavelengths - min_wavelength) / (max_wavelength - min_wavelength + 1e-6)) * num_categories
        ).long()

        # Flatten tensors for loss computation
        true_categories = true_categories.reshape(-1)
        predicted_categories = predicted_categories.view(-1, num_categories)

        # Use Cross-Entropy Loss for category comparison
        loss_fn = torch.nn.CrossEntropyLoss()
        category_loss = loss_fn(predicted_categories, true_categories)

        return category_loss


def main_demo():
        # Example 1: RGB Mode
    print("Generating RGB Rays...")
    mi.set_variant('scalar_rgb')  # Set Mitsuba to RGB mode

    # Define light source parameters
    origin_rgb = [0.0, 0.0, 0.0]
    direction_rgb = [0.0, 0.0, -1.0]
    beam_width_rgb = 0.01  # radians
    num_rays_rgb = 100000  # Large number of rays for performance testing
    gaussian_falloff_error_rgb = 0.005
    gaussian_source_error_rgb = 0.001
    colors_rgb = torch.ones(num_rays_rgb, 3, dtype=torch.float32) * torch.tensor([1.0, 0.0, 0.0])  # Red color

    # Instantiate the light source in RGB mode
    light_source_rgb = FocusedLightSource.from_rgb(
        origin=origin_rgb,
        direction=direction_rgb,
        beam_width=beam_width_rgb,
        num_rays=num_rays_rgb,
        gaussian_falloff_error=gaussian_falloff_error_rgb,
        gaussian_source_error=gaussian_source_error_rgb,
        colors=colors_rgb  # Optional: specify colors per ray
    )

    # Generate rays
    rays_rgb = light_source_rgb.generate_rays()
    print(f"Generated {len(rays_rgb)} RGB rays.")

    # Example 2: Spectral Mode
    print("\nGenerating Spectral Rays...")
    mi.set_variant('scalar_spectral')  # Set Mitsuba to Spectral mode

    # Define light source parameters
    origin_spec = [0.0, 0.0, 0.0]
    direction_spec = [0.0, 0.0, -1.0]
    beam_width_spec = 0.01  # radians
    num_rays_spec = 100000  # Large number of rays for performance testing
    gaussian_falloff_error_spec = 0.005
    gaussian_source_error_spec = 0.001
    wavelengths_spec = torch.linspace(400e-9, 700e-9, steps=num_rays_spec)  # Wavelengths from 400nm to 700nm

    # Instantiate the light source in Spectral mode
    light_source_spec = FocusedLightSource(
        origin=origin_spec,
        direction=direction_spec,
        beam_width=beam_width_spec,
        num_rays=num_rays_spec,
        gaussian_falloff_error=gaussian_falloff_error_spec,
        gaussian_source_error=gaussian_source_error_spec,
        mode='Spectral',
        wavelengths=wavelengths_spec
    )

    # Generate rays
    rays_spec = light_source_spec.generate_rays()
    print(f"Generated {len(rays_spec)} Spectral rays.")
    try:
        print("Starting main_demo")  # Debug Statement
        # Initialize Energy
        energy = Energy()
        # Initialize Light with the Energy instance
        light = Light(energy=energy)

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

        # Generate emission data
        emission_tensor = light.process_game_tick(
            timeline=timeline,
            wavelengths=wavelengths,
            time_slices=time_slices,
            stencil_matrix=stencil_matrix,
            grid_tuple=grid_tuple,
            extended_sources=True
        )

        # Initialize results
        correct_losses = []
        incorrect_losses = []
        random_losses = []
        num_tests = 10  # Reduced for demonstration

        for i in range(num_tests):
            print(f"Starting test {i + 1}/{num_tests}")  # Debug Statement
            # Correct answer
            gt_dense = emission_tensor.clone().detach()
            correct_prediction = gt_dense.clone()  # Correct prediction matches ground truth

            # Example: Generate predicted categories based on wavelength
            gt_wavelengths = gt_dense[:, 8]  # Assuming wavelength is the ninth feature
            if gt_wavelengths.numel() == 0:
                print("No wavelength data available. Skipping loss calculation.")
                continue  # Skip this iteration

            min_wavelength = gt_wavelengths.min()
            max_wavelength = gt_wavelengths.max()
            true_categories = torch.floor(
                ((gt_wavelengths - min_wavelength) / (max_wavelength - min_wavelength + 1e-6)) * num_categories
            ).long()

            # Simulated predicted categories (one-hot)
            predicted_categories = torch.nn.functional.one_hot(true_categories, num_classes=num_categories).float()

            # Incorrect prediction (random one-hot)
            incorrect_indices = torch.randint(0, num_categories, (true_categories.size(0),))
            incorrect_categories = torch.nn.functional.one_hot(incorrect_indices, num_classes=num_categories).float()

            # Random prediction (random probabilities)
            random_categories = torch.rand(true_categories.size(0), num_categories)

            # Compute losses
            correct_loss = light.calculate_category_loss(emission_tensor, predicted_categories, num_categories)
            incorrect_loss = light.calculate_category_loss(emission_tensor, incorrect_categories, num_categories)
            random_loss = light.calculate_category_loss(emission_tensor, random_categories, num_categories)

            # Record losses
            correct_losses.append(correct_loss.item())
            incorrect_losses.append(incorrect_loss.item())
            random_losses.append(random_loss.item())

        # Calculate statistics after all tests
        if correct_losses and incorrect_losses and random_losses:
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
        else:
            print("No loss data collected.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main_demo()
