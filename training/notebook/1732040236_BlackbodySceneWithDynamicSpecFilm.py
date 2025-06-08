import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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


class BlackbodySceneWithDynamicSpecFilm:
    def __init__(self, output_path="output.exr", resolution=(1024, 768), spp=64):
        self.output_path = output_path
        self.resolution = resolution
        self.spp = spp

        # Dynamically generate spectral response files
        self.generate_spectral_response_files()

        # Define the spectral film
        self.film = {
            'type': 'specfilm',
            'width': resolution[0],
            'height': resolution[1],
            'component_format': 'float32',
            'band1_red': {
                'type': 'spectrum',
                'filename': 'data_red.spd'
            },
            'band2_green': {
                'type': 'spectrum',
                'filename': 'data_green.spd'
            },
            'band3_blue': {
                'type': 'spectrum',
                'filename': 'data_blue.spd'
            }
        }

        # Define the blackbody emitter
        self.emitter = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 2]) @ mi.ScalarTransform4f.scale([0.5, 0.5, 1]),
            'emitter': {
                'type': 'blackbody',
                'temperature': 3000  # Warm, orange light
            }
        }

        # Define the camera
        self.camera = {
            'type': 'thinlens',
            'fov': 45,
            'aperture_radius': 0.1,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, -5],  # Position behind the blackbody emitter
                target=[0, 0, 0],   # Look at the origin
                up=[0, 1, 0]        # Up direction
            ),
            'film': self.film
        }

        # Define the integrator
        self.integrator = {
            'type': 'path',
            'max_depth': 10
        }

        # Combine into a scene
        self.scene_dict = {
            'type': 'scene',
            'emitter': self.emitter,
            'sensor': self.camera,
            'integrator': self.integrator
        }

    def generate_spectral_response_files(self):
        """
        Dynamically generate spectral response profiles and save them as .spd files.
        """
        # Define the histogram, frequency centers, and std deviations for RGB channels
        histogram = [1.0, 0.8, 0.6]  # Relative amplitudes
        frequency_maps = [
            (450, 20),  # Red channel center and spread
            (550, 30),  # Green channel center and spread
            (650, 25)   # Blue channel center and spread
        ]

        # Generate and save .spd files for each channel
        for idx, (center, std_dev) in enumerate(frequency_maps):
            wavelengths, spectral_response = generate_spectral_response(
                histogram=[1.0], frequency_map=[center], std_devs=[std_dev]
            )
            filename = f"data_{['red', 'green', 'blue'][idx]}.spd"
            save_to_spd(filename, wavelengths, spectral_response)



    def render(self):
        # Load the scene
        scene = mi.load_dict(self.scene_dict)

        # Render the scene
        print("Rendering started...")
        image_tensor = mi.render(scene, spp=self.spp)
        print("Rendering completed.")
        # Convert the Mitsuba tensor to a NumPy array
        image_array = np.array(image_tensor)
        # Log tensor statistics
        print("Tensor Statistics:")
        print(f"Shape: {image_array.shape}")
        print(f"Min: {image_array.min()}")
        print(f"Max: {image_array.max()}")
        print(f"Mean: {image_array.mean()}")
        print(f"Std: {image_array.std()}")

        # Check if the tensor is all zeros
        if image_array.max() == 0:
            print("Warning: Rendered image contains only zero values!")


        # Map tensor data to RGB channels (if multi-channel)
        if len(image_array.shape) == 3 and image_array.shape[-1] == 3:
            # Scale the data to [0, 255] for 8-bit RGB
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)

            # Convert to a PIL image
            image = Image.fromarray(image_array, mode="RGB")

            # Save the image
            image.save(self.output_path.replace(".exr", ".png"))
            print(f"Image saved to {self.output_path.replace('.exr', '.png')}")
        else:
            print("Error: Rendered image tensor is not in the expected RGB format.")



def main():
    # Create the scene with dynamic spectral film
    scene = BlackbodySceneWithDynamicSpecFilm(
        output_path="blackbody_spectral_dynamic.exr",
        resolution=(1024, 768),
        spp=1280
    )

    # Render the scene
    scene.render()


if __name__ == "__main__":
    main()
