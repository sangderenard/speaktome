import mitsuba as mi
mi.set_variant('scalar_spectral')  # Use scalar_spectral for high physical accuracy

import numpy as np
import torch
from PIL import Image


class MitsubaToPILConverter:
    """
    A helper class to convert Mitsuba output (NumPy arrays) into PIL images.
    """

    @staticmethod
    def mitsuba_to_pil(mitsuba_array):
        """
        Converts a Mitsuba-rendered NumPy array into a PIL image.

        Args:
            mitsuba_array (numpy.ndarray): The Mitsuba-rendered image array. Expected shape is (height, width, channels).

        Returns:
            PIL.Image: The converted PIL image.
        """
        if mitsuba_array.ndim == 2:  # Grayscale image
            mitsuba_array = np.stack([mitsuba_array] * 3, axis=-1)
        elif mitsuba_array.ndim != 3 or mitsuba_array.shape[-1] not in [3, 4]:
            raise ValueError("Mitsuba array must have shape (H, W, 3) or (H, W, 4) for RGB or RGBA images.")

        mitsuba_array = np.clip(mitsuba_array, 0, 1)
        mitsuba_array = (mitsuba_array * 255).astype(np.uint8)

        return Image.fromarray(mitsuba_array)

    @staticmethod
    def save_image(pil_image, filepath):
        """
        Saves a PIL image to a file.

        Args:
            pil_image (PIL.Image): The image to save.
            filepath (str): The path where the image will be saved.
        """
        pil_image.save(filepath)
        print(f"Image saved to {filepath}")


class SceneBuilderWithFiberOpticIlluminator:
    """
    A class to build a Mitsuba 3 scene with embedded sensors and a forward-facing
    fiber optic illuminator for diagnostic snapshots.
    """

    def __init__(self, base_scene, sensor_field, illuminator_intensity=10.0, illuminator_cutoff=20.0):
        """
        Initializes the scene builder with base scene, sensor field, and illuminator properties.

        Args:
            base_scene (dict): Base Mitsuba-compatible scene dictionary.
            sensor_field (list): List of sensor definitions as [position, orientation, resolution].
            illuminator_intensity (float): Intensity of the fiber optic illuminator.
            illuminator_cutoff (float): Cutoff angle (in degrees) for the spotlight beam.
        """
        self.base_scene = base_scene
        self.sensor_field = sensor_field
        self.illuminator_intensity = illuminator_intensity
        self.illuminator_cutoff = illuminator_cutoff
        self.snapshot_dump_queue = []  # Stores snapshots for later inspection

    def build_scene(self):
        """
        Build the scene with sensors and illuminator embedded.

        Returns:
            mitsuba.Scene: Scene object with sensors and illuminator embedded.
        """
        scene_with_sensors = self.base_scene.copy()

        for idx, sensor_def in enumerate(self.sensor_field):
            position, orientation, resolution = sensor_def

            # Sensor configuration
            sensor_dict = {
                f'sensor_{idx}': {
                    'type': 'rectangle',
                    'to_world': mi.ScalarTransform4f.look_at(
                        origin=position,
                        target=[position[i] + orientation[i] for i in range(3)],
                        up=[0, 1, 0]
                    ),
                    'bsdf': {
                        'type': 'diffuse',
                        'reflectance': {'type': 'rgb', 'value': [0.9, 0.1, 0.1]}
                    }
                }
            }

            # Add sensor to the scene
            scene_with_sensors.update(sensor_dict)

        # Add the fiber optic illuminator
        illuminator_position = [0, 2, 5]
        illuminator_direction = [0, -1, -1]
        illuminator_dict = {
            'fiber_optic_illuminator': {
                'type': 'spot',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=illuminator_position,
                    target=[illuminator_position[i] + illuminator_direction[i] for i in range(3)],
                    up=[0, 1, 0]
                ),
                'intensity': {
                    'type': 'spectrum',
                    'value': self.illuminator_intensity
                },
                'cutoff_angle': self.illuminator_cutoff,
                'beam_width': 0.5  # Controls the tightness of the beam
            }
        }

        scene_with_sensors.update(illuminator_dict)
        return mi.load_dict(scene_with_sensors)

    def perform_diagnostics(self, scene, exposures=3, blend_mode="add"):
        """
        Perform diagnostics with the fiber optic illuminator, blending multiple exposures.

        Args:
            scene (mitsuba.Scene): The scene object to diagnose.
            exposures (int): Number of exposures to blend.
            blend_mode (str): Blend mode ('add', 'average').
        """
        print("Performing diagnostic snapshot with fiber optic illumination...")

        diagnostic_camera = mi.load_dict({
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 10, 10],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 128,
                'height': 128,
                'rfilter': {'type': 'box'}
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 32
            }
        })

        # Collect exposures
        exposure_images = []
        for _ in range(exposures):
            img = mi.render(scene, sensor=diagnostic_camera)
            exposure_images.append(torch.tensor(np.array(img), dtype=torch.float32))

        # Blend exposures
        if blend_mode == "add":
            blended_image = torch.clamp(sum(exposure_images), 0, 1)
        elif blend_mode == "average":
            blended_image = torch.clamp(sum(exposure_images) / exposures, 0, 1)
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")

        # Store the blended snapshot in the dump queue
        self.snapshot_dump_queue.append(blended_image.numpy())
        print("Blended diagnostic snapshot captured.")

    def build_and_diagnose_scene(self, exposures=3, blend_mode="add"):
        """
        Build the scene and perform diagnostics with blended exposures.

        Args:
            exposures (int): Number of exposures to blend.
            blend_mode (str): Blend mode ('add', 'average').

        Returns:
            mitsuba.Scene: Scene object with sensors and illuminator embedded.
        """
        scene = self.build_scene()
        self.perform_diagnostics(scene, exposures, blend_mode)
        return scene
import mitsuba as mi
mi.set_variant('scalar_spectral')  # Use scalar_spectral for high physical accuracy

class ThreeFiberExplorationCamera:
    """
    A class to define and manage a three-fiber exploration camera setup with a camera lens,
    a spotlight, and a floodlight.
    """

    def __init__(self, position, orientation, resolution, spotlight_intensity=50.0,
                 floodlight_intensity=20.0, lens_aperture=2.8):
        """
        Initializes the three-fiber exploration camera.

        Args:
            position (list): Position of the camera in the scene [x, y, z].
            orientation (list): Orientation vector [dx, dy, dz].
            resolution (list): Resolution of the camera sensor [width, height].
            spotlight_intensity (float): Intensity of the spotlight.
            floodlight_intensity (float): Intensity of the floodlight.
            lens_aperture (float): Aperture size for the camera lens.
        """
        self.position = position
        self.orientation = orientation
        self.resolution = resolution
        self.spotlight_intensity = spotlight_intensity
        self.floodlight_intensity = floodlight_intensity
        self.lens_aperture = lens_aperture

    def get_camera(self):
        """
        Returns the camera configuration.

        Returns:
            dict: Mitsuba-compatible dictionary for the camera.
        """
        return {
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=self.position,
                target=[self.position[i] + self.orientation[i] for i in range(3)],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': self.resolution[0],
                'height': self.resolution[1],
                'rfilter': {'type': 'box'}
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'aperture_radius': 1.0 / self.lens_aperture  # Simplified aperture control
        }

    def get_spotlight(self):
        """
        Returns the spotlight configuration.

        Returns:
            dict: Mitsuba-compatible dictionary for the spotlight.
        """
        return {
            'type': 'spot',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=self.position,
                target=[self.position[i] + self.orientation[i] for i in range(3)],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'spectrum',
                'value': self.spotlight_intensity
            },
            'cutoff_angle': 20.0,  # Narrow beam
            'beam_width': 0.5
        }

    def get_floodlight(self):
        """
        Returns the floodlight configuration.

        Returns:
            dict: Mitsuba-compatible dictionary for the floodlight.
        """
        return {
            'type': 'diffuse',
            'to_world': mi.ScalarTransform4f.translate(self.position),
            'intensity': {
                'type': 'spectrum',
                'value': self.floodlight_intensity
            }
        }

    def integrate_into_scene(self, scene_dict):
        """
        Integrates the camera, spotlight, and floodlight into an existing scene dictionary.

        Args:
            scene_dict (dict): Base scene dictionary to modify.
        """
        scene_dict.update({
            'three_fiber_camera': self.get_camera(),
            'spotlight': self.get_spotlight(),
            'floodlight': self.get_floodlight()
        })
        print("Three Fiber Exploration Camera and lights integrated into the scene.")


def main():
    """
    Main function to demonstrate the SceneBuilderWithFiberOpticIlluminator class with configurable parameters.
    """
    # Define base scene
    base_scene = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 10},
        'shape_plane': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([10, 10, 1]),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        },
        'shape_sphere': {
            'type': 'sphere',
            'radius': 1.0,
            'to_world': mi.ScalarTransform4f.translate([0, 1, 0]),
            'bsdf': {'type': 'dielectric', 'int_ior': 1.5, 'ext_ior': 1.0}
        }
    }

    # Define sensor field: [position, orientation, resolution]
    sensor_field = [
        [[-2, 2, 5], [0, -1, -1], [256, 256]],
        [[0, 2, 5], [0, -1, -1], [256, 256]],
        [[2, 2, 5], [0, -1, -1], [256, 256]]
    ]

    # Define configuration parameters
    config = {
        "illuminator_intensity": 50.0,  # Bright light
        "illuminator_cutoff": 15.0,  # Tight beam
        "exposures": 10,  # Long exposure
        "blend_mode": "add"  # Additive blending
    }

    # Instantiate the builder with configurable parameters
    builder = SceneBuilderWithFiberOpticIlluminator(
        base_scene=base_scene,
        sensor_field=sensor_field,
        illuminator_intensity=config["illuminator_intensity"],
        illuminator_cutoff=config["illuminator_cutoff"]
    )

    # Build and diagnose the scene with blended exposures
    print("Building and diagnosing the scene with configured parameters...")
    scene = builder.build_and_diagnose_scene(
        exposures=config["exposures"],
        blend_mode=config["blend_mode"]
    )

    # Retrieve and save snapshots
    snapshots = builder.snapshot_dump_queue
    if snapshots:
        converter = MitsubaToPILConverter()
        for i, snapshot in enumerate(snapshots):
            pil_image = converter.mitsuba_to_pil(snapshot)
            filepath = f"configured_snapshot_{i + 1}.png"
            converter.save_image(pil_image, filepath)
    else:
        print("No snapshots were captured.")

if __name__ == "__main__":
    main()

