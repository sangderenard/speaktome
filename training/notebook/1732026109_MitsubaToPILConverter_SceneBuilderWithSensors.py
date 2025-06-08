import mitsuba as mi
mi.set_variant('scalar_spectral')  # Use scalar_spectral for high physical accuracy

from PIL import Image
import numpy as np


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


class SceneBuilderWithSensors:
    """
    A class to build a Mitsuba 3 scene with embedded sensors, perform diagnostics,
    and store diagnostic snapshots in a dump queue.
    """

    def __init__(self, base_scene, sensor_field):
        """
        Initializes the scene builder with base scene and sensor field.

        Args:
            base_scene (dict): Base Mitsuba-compatible scene dictionary.
            sensor_field (list): List of sensor definitions as [position, orientation, resolution].
        """
        self.base_scene = base_scene
        self.sensor_field = sensor_field  # Format: [position, orientation, resolution]
        self.snapshot_dump_queue = []

    def build_scene(self):
        """
        Build the scene with sensors embedded as thin rectangular prisms.

        Returns:
            mitsuba.Scene: Scene object with sensors embedded.
        """
        scene_with_sensors = self.base_scene.copy()

        for idx, sensor_def in enumerate(self.sensor_field):
            position, orientation, resolution = sensor_def

            # Add physical sensor representation as a thin rectangular prism
            sensor_representation = {
                f'sensor_rep_{idx}': {
                    'type': 'rectangle',
                    'to_world': mi.ScalarTransform4f.look_at(
                        origin=position,
                        target=[position[i] + orientation[i] for i in range(3)],
                        up=[0, 1, 0]
                    ) @ mi.ScalarTransform4f.scale([0.2, 0.01, 1.0]),  # Thin rectangle
                    'bsdf': {
                        'type': 'diffuse',
                        'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]}  # Red color
                    }
                }
            }

            # Add the sensor representation to the scene
            scene_with_sensors.update(sensor_representation)

        return mi.load_dict(scene_with_sensors)

    def perform_diagnostics(self, scene, fast_pass=0):
        """
        Perform diagnostics on the scene.

        Args:
            scene (mitsuba.Scene): The scene object to diagnose.
            fast_pass (int): Determines the level of diagnostics:
                - 0: Full Mitsuba diagnostics.
                - 1: Low-resolution snapshot of the sensor field.
                - 2+: Skip diagnostics.
        """
        if fast_pass == 0:
            print("Performing full Mitsuba diagnostics...")
            print("Shapes:", scene.shapes())
            print("Emitters:", scene.emitters())
            print("Sensors:", scene.sensors())
            bbox = scene.bbox()
            print("Bounding box:", bbox)

        elif fast_pass == 1:
            print("Performing low-resolution snapshot of the sensor field...")

            # Add a temporary diagnostic light
            diagnostic_light = {
                'type': 'point',
                'position': [0, 10, 10],
                'intensity': {
                    'type': 'rgb',
                    'value': [50, 50, 50]  # Bright diagnostic light
                }
            }
            scene_with_light = self.base_scene.copy()
            scene_with_light.update({"diagnostic_light": diagnostic_light})
            scene = mi.load_dict(scene_with_light)

            # Diagnostic camera
            diagnostic_camera = mi.load_dict({
                'type': 'perspective',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=[0, 10, 20],
                    target=[0, 0, 0],
                    up=[0, 1, 0]
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': 64,  # Low resolution
                    'height': 64,
                    'rfilter': {'type': 'box'}
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': 1  # Minimal samples for quick snapshot
                }
            })
            img = mi.render(scene, sensor=diagnostic_camera)

            # Store the snapshot in the dump queue
            snapshot = np.array(img)
            self.snapshot_dump_queue.append(snapshot)
            print("Snapshot added to dump queue with diagnostic light.")

        elif fast_pass >= 2:
            print("Skipping diagnostics.")


    def build_and_diagnose_scene(self, fast_pass=0):
        """
        Build the scene and perform diagnostics.

        Args:
            fast_pass (int): Diagnostic level (0: full, 1: low-res snapshot, 2+: skip).

        Returns:
            mitsuba.Scene: Scene object with sensors embedded.
        """
        scene = self.build_scene()
        self.perform_diagnostics(scene, fast_pass)
        return scene

    def get_snapshot_dump_queue(self):
        """
        Retrieve all snapshots stored in the dump queue.

        Returns:
            list: List of snapshots as NumPy arrays.
        """
        return self.snapshot_dump_queue


def main():
    """
    Main function to demonstrate the SceneBuilderWithSensors class and MitsubaToPILConverter utility.
    """
    base_scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 10
        },
        'shape_plane': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([10, 10, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        }
    }

    sensor_field = [
        [[-2, 2, 5], [0, -1, -1], [256, 256]],
        [[0, 2, 5], [0, -1, -1], [256, 256]],
        [[2, 2, 5], [0, -1, -1], [256, 256]]
    ]

    builder = SceneBuilderWithSensors(base_scene, sensor_field)
    print("Building and diagnosing the scene...")
    scene = builder.build_and_diagnose_scene(fast_pass=1)

    snapshots = builder.get_snapshot_dump_queue()

    if snapshots:
        converter = MitsubaToPILConverter()
        for i, snapshot in enumerate(snapshots):
            pil_image = converter.mitsuba_to_pil(snapshot)
            filepath = f"snapshot_{i + 1}.png"
            converter.save_image(pil_image, filepath)
    else:
        print("No snapshots were captured.")


if __name__ == "__main__":
    main()
