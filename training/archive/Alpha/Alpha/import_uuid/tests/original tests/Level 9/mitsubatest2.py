import mitsuba as mi
import numpy as np
from PIL import Image
import os

# Activate Mitsuba in spectral mode
mi.set_variant('scalar_spectral')


def generate_spectral_response(histogram, frequency_map, std_devs, wavelength_range=(400, 700), resolution=1):
    """
    Generate a spectral response profile based on a histogram and frequency data.
    """
    if not (len(histogram) == len(frequency_map) == len(std_devs)):
        raise ValueError("histogram, frequency_map, and std_devs must have the same length.")

    wavelengths = np.arange(wavelength_range[0], wavelength_range[1] + resolution, resolution)
    spectral_response = np.zeros_like(wavelengths, dtype=np.float64)

    for weight, center, std in zip(histogram, frequency_map, std_devs):
        spectral_response += weight * np.exp(-0.5 * ((wavelengths - center) / std) ** 2)

    return wavelengths, spectral_response


def save_to_spd(filename, wavelengths, spectral_response):
    """
    Save a spectral response to an .spd file.
    """
    with open(filename, 'w') as f:
        for wl, resp in zip(wavelengths, spectral_response):
            f.write(f"{wl} {resp}\n")


class BaseScene:
    """
    Base class containing shared scene components.
    """
    def __init__(self, resolution=(1024, 768), spp=64):
        self.resolution = resolution
        self.spp = spp
        self.objects = self.define_objects()
        self.camera_transform = self.define_camera_transform()

    def define_objects(self):
        """
        Define shared objects in the scene.
        Modify this method to add objects as needed.
        """
        # Example: Adding a simple sphere
        return {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.7, 0.2, 0.2]
                }
            }
        }

    def define_camera_transform(self):
        """
        Define the camera transformation.
        """
        return mi.ScalarTransform4f.look_at(
            origin=[0, 0, -5],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )


class RGBScene(BaseScene):
    """
    Scene configuration for RGB rendering with environmental lighting.
    """
    def __init__(self, output_path="rgb_output.png", resolution=(1024, 768), spp=64):
        super().__init__(resolution, spp)
        self.output_path = output_path
        self.scene_dict = self.define_scene()

    def define_scene(self):
        """
        Define the RGB scene.
        """
        scene = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': 10
            },
            'sensor': {
                'type': 'perspective',
                'fov': 45,
                'to_world': self.camera_transform,
                'film': {
                    'type': 'hdrfilm',
                    'width': self.resolution[0],
                    'height': self.resolution[1],
                    'pixel_format': 'rgb',
                    
                }
            },
            'emitter': {
                'type': 'constant',
                'radiance': {
                    'type': 'rgb',
                    'value': [1.0, 1.0, 1.0]  # White environmental light
                },
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90)  # Adjust orientation if needed
            },
            'shape': self.objects
        }
        return scene

    def render(self):
        """
        Render the RGB scene.
        """
        scene = mi.load_dict(self.scene_dict)
        print("Rendering RGB Scene...")
        image = mi.render(scene, spp=self.spp)
        print("RGB Rendering completed.")

        # Convert to NumPy array
        image_np = np.array(image)
        
        # Normalize and convert to 8-bit RGB
        image_np = np.clip(image_np / np.max(image_np), 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # Save the image
        img = Image.fromarray(image_np, 'RGB')
        img.save(self.output_path)
        print(f"RGB Image saved to {self.output_path}")


class SpectralScene(BaseScene):
    """
    Scene configuration for Spectral rendering with a blackbody emitter.
    """
    def __init__(self, output_path="spectral_output.exr", resolution=(1024, 768), spp=64):
        super().__init__(resolution, spp)
        self.output_path = output_path
        self.generate_spectral_response_files()
        self.scene_dict = self.define_scene()

    def generate_spectral_response_files(self):
        """
        Generate spectral response files for RGB channels.
        """
        # Define the frequency centers and standard deviations for RGB channels
        frequency_maps = [
            (450, 20),  # Red
            (550, 30),  # Green
            (650, 25)   # Blue
        ]

        # Create directory for SPD files if it doesn't exist
        os.makedirs('spectral_data', exist_ok=True)

        # Generate and save .spd files
        for idx, (center, std_dev) in enumerate(frequency_maps):
            wavelengths, spectral_response = generate_spectral_response(
                histogram=[1.0],
                frequency_map=[center],
                std_devs=[std_dev]
            )
            filename = os.path.join('spectral_data', f"data_{['red', 'green', 'blue'][idx]}.spd")
            save_to_spd(filename, wavelengths, spectral_response)
            print(f"Spectral response saved to {filename}")

    def define_scene(self):
        """
        Define the Spectral scene.
        """
        scene = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': 10
            },
            'sensor': {
                'type': 'thinlens',
                'fov': 45,
                'aperture_radius': 0.1,
                'to_world': self.camera_transform,
                'film': {
                    'type': 'specfilm',
                    'width': self.resolution[0],
                    'height': self.resolution[1],
                    'component_format': 'float32',
                    
                    'band1_red': {
                        'type': 'spectrum',
                        'filename': 'spectral_data/data_red.spd'
                    },
                    'band2_green': {
                        'type': 'spectrum',
                        'filename': 'spectral_data/data_green.spd'
                    },
                    'band3_blue': {
                        'type': 'spectrum',
                        'filename': 'spectral_data/data_blue.spd'
                    }
                }
            },
            'emitter': {
                'type': 'blackbody',
                'temperature': 3000,  # Warm, orange light
                'to_world': mi.ScalarTransform4f.translate([0, 0, 2]) @ mi.ScalarTransform4f.scale([0.5, 0.5, 1])
            },
            'shape': self.objects
        }
        return scene

    def render(self):
        """
        Render the Spectral scene.
        """
        scene = mi.load_dict(self.scene_dict)
        print("Rendering Spectral Scene...")
        image = mi.render(scene, spp=self.spp)
        print("Spectral Rendering completed.")

        # Save the spectral image as EXR
        mi.util.write_bitmap(self.output_path, image)
        print(f"Spectral Image saved to {self.output_path}")


def main():
    # Define rendering parameters
    resolution = (1024, 768)
    spp_rgb = 64
    spp_spectral = 1280

    # Render RGB Scene for Verification
    rgb_scene = RGBScene(
        output_path="rgb_output.png",
        resolution=resolution,
        spp=spp_rgb
    )
    rgb_scene.render()

    # Render Spectral Scene for Physics Simulation
    spectral_scene = SpectralScene(
        output_path="spectral_output.exr",
        resolution=resolution,
        spp=spp_spectral
    )
    spectral_scene.render()


if __name__ == "__main__":
    main()
