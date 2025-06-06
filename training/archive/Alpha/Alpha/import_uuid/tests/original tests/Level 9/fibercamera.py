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
        if mitsuba_array.ndim == 2:  # Grayscale image
            mitsuba_array = np.stack([mitsuba_array] * 3, axis=-1)
        elif mitsuba_array.ndim != 3 or mitsuba_array.shape[-1] not in [3, 4]:
            raise ValueError("Mitsuba array must have shape (H, W, 3) or (H, W, 4) for RGB or RGBA images.")

        mitsuba_array = np.clip(mitsuba_array, 0, 1)
        mitsuba_array = (mitsuba_array * 255).astype(np.uint8)

        return Image.fromarray(mitsuba_array)

    @staticmethod
    def save_image(pil_image, filepath):
        pil_image.save(filepath)
        print(f"Image saved to {filepath}")


class ThreeFiberExplorationCamera:
    """
    A class to define and manage a three-fiber exploration camera setup with a camera lens,
    a spotlight, and a floodlight.
    """

    def __init__(self, position, orientation, resolution, spotlight_intensity=50.0,
                 floodlight_intensity=20.0, lens_aperture=2.8, flood_width=3, flood_height=1):
        self.position = position
        self.orientation = orientation
        self.resolution = resolution
        self.spotlight_intensity = spotlight_intensity
        self.floodlight_intensity = floodlight_intensity
        self.lens_aperture = lens_aperture
        self.flood_width = flood_width
        self.flood_height = flood_height

    def get_camera(self):
        return {
            'type': 'thinlens',
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
            'aperture_radius': 1.0 / self.lens_aperture
        }

    def get_spotlight(self):
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
            'cutoff_angle': 20.0,
            'beam_width': 0.5
        }


    def get_floodlight(self):
        return {
            'type': 'area',
            'to_world': mi.ScalarTransform4f.translate(self.position),
            'emitter': {
                'type': 'spectrum',
                'radiance': self.floodlight_intensity  # Ensure this is a spectrum-compatible value
            },
            'shape': {
                'type': 'rectangle',  # You can choose other shapes like 'disk', 'sphere', etc.
                'width': self.flood_width,   # Define appropriate dimensions
                'height': self.flood_height
            }
        }


    def integrate_into_scene(self, scene_dict):
        scene_dict.update({
            'three_fiber_camera': self.get_camera(),
            'spotlight': self.get_spotlight(),
            'floodlight': self.get_floodlight()
        })
        print("Three Fiber Exploration Camera and lights integrated into the scene.")


class SceneBuilderWithLongExposure:
    """
    A class to build a Mitsuba 3 scene, perform diagnostics, and simulate long exposure.
    """

    def __init__(self, base_scene, camera_class, exposure_blend_mode='add'):
        self.base_scene = base_scene
        self.camera_class = camera_class
        self.exposure_blend_mode = exposure_blend_mode
        self.snapshots = []

    def build_scene_with_camera(self):
        self.camera_class.integrate_into_scene(self.base_scene)
        return mi.load_dict(self.base_scene)

    def perform_long_exposure(self, scene, exposures=5):
        print(f"Simulating long exposure with {exposures} exposures...")
        long_exposure = None

        for i in range(exposures):
            img = mi.render(scene, sensor=scene['three_fiber_camera'])
            img_array = np.array(img)

            if long_exposure is None:
                long_exposure = img_array
            else:
                if self.exposure_blend_mode == 'add':
                    long_exposure += img_array
                elif self.exposure_blend_mode == 'max':
                    long_exposure = np.maximum(long_exposure, img_array)
                else:
                    raise ValueError("Unsupported blend mode. Use 'add' or 'max'.")

        # Normalize the final exposure
        long_exposure = np.clip(long_exposure / exposures, 0, 1)
        return long_exposure


def main():
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

    camera = ThreeFiberExplorationCamera(
        position=[0, 2, 5],
        orientation=[0, -1, -1],
        resolution=[256, 256],
        spotlight_intensity=50.0,
        floodlight_intensity=20.0,
        lens_aperture=2.8
    )

    builder = SceneBuilderWithLongExposure(base_scene, camera)
    scene = builder.build_scene_with_camera()

    long_exposure_image = builder.perform_long_exposure(scene, exposures=10)

    converter = MitsubaToPILConverter()
    pil_image = converter.mitsuba_to_pil(long_exposure_image)
    converter.save_image(pil_image, "long_exposure_output.png")


if __name__ == "__main__":
    main()
