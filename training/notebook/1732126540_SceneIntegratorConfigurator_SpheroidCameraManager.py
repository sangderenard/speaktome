import mitsuba as mi
import numpy as np

mi.set_variant("scalar_spectral")
class SceneIntegratorConfigurator:
    def __init__(self):
        # Default values and supported integrators
        self.integrator = None
        self.integrator_options = {
            "aov": {"aovs": []},
            "path": {"max_depth": -1, "rr_depth": 5, "hide_emitters": False},
            "direct": {"shading_samples": None, "emitter_samples": None, "bsdf_samples": None, "hide_emitters": False},
            "volpath": {"max_depth": -1, "rr_depth": 5, "hide_emitters": False},
            "volpathmis": {"max_depth": -1, "rr_depth": 5, "hide_emitters": False},
            "prb": {"max_depth": 6, "rr_depth": 5},
            "prb_basic": {"max_depth": 6},
            "direct_projective": {
                "sppc": 32,
                "sppp": 32,
                "sppi": 128,
                "guiding": "octree",
                "guiding_proj": True,
                "guiding_rounds": 1,
            },
            "prb_projective": {
                "max_depth": -1,
                "rr_depth": 5,
                "sppc": 32,
                "sppp": 32,
                "sppi": 128,
                "guiding": "octree",
                "guiding_proj": True,
                "guiding_rounds": 1,
            },
            "prbvolpath": {"max_depth": 8, "rr_depth": 5, "hide_emitters": False},
            "stokes": {"integrator": None},
            "moment": {"integrator": None},
            "ptracer": {"max_depth": -1, "rr_depth": 5, "hide_emitters": False, "samples_per_pass": None},
            "depth": {},
        }

    def configure_integrator(self, integrator_type, **kwargs):
        """
        Configure the integrator dynamically based on the integrator type and parameters.
        :param integrator_type: Type of integrator to configure.
        :param kwargs: Parameters for the selected integrator.
        """
        if integrator_type not in self.integrator_options:
            raise ValueError(f"Unsupported integrator type: {integrator_type}")

        # Base configuration with default values
        config = {"type": integrator_type}
        config.update(self.integrator_options[integrator_type])

        # Override with user-provided parameters
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
            else:
                raise ValueError(f"Invalid parameter '{key}' for integrator type '{integrator_type}'")

        self.integrator = config
        return self.integrator

class SpheroidCameraManager:
    def __init__(self, resolution=(512, 512), aovs=None):
        """
        Initialize the Spheroid Camera Manager with an inner unit sphere camera and
        an outer 10x unit sphere with inverted normals.
        
        :param resolution: Resolution of the film (width, height).
        :param aovs: List of AOV variables to output (e.g., 'radiance', 'normals').
        """
        self.resolution = resolution
        self.aovs = aovs or ["radiance"]  # Default AOV
        self.scene = None
        self.integrator = None

    def build_inner_camera(self):
        """
        Define the inner unit sphere camera.
        """
        sensor = {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0],  # Center of the unit sphere
                target=[0, 0, -1],  # Looking forward
                up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm",
                "width": self.resolution[0],
                "height": self.resolution[1],
                "rfilter": {"type": "gaussian"},
            },
            "sampler": {"type": "independent", "sample_count": 64},
        }
        return sensor

    def build_outer_sphere(self):
        """
        Define the outer 10x unit sphere with inverted normals and emissive procedural noise texture.
        The upper hemisphere represents the sky, and the lower hemisphere represents the earth.
        """
        # Generate a uniform white noise texture
        noise_texture = self.generate_noise_texture("noise_texture.exr")

        # Create the sphere with emissive properties
        sphere = {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.scale(10.0),  # Scale to 10x the size of the inner sphere
            "flip_normals": True,  # Invert the normals to point inward
            "emitter": {  # Make the sphere emissive
                "type": "area",
                "radiance": {
                    "type": "bitmap",
                    "filename": noise_texture,  # Use the generated noise texture
                    "filter_type": "nearest",  # Ensure no interpolation
                },
            },
        }
        return sphere

    def generate_noise_texture(self, filename):
        """
        Generate a uniform white noise texture and save it as an EXR file.

        Parameters:
        - filename: Path to save the generated texture file.

        Returns:
        - str: The filename of the generated texture.
        """
        # Create a 256x256 uniform white noise array (RGB)
        width, height = 256, 256
        noise = np.random.rand(height, width, 3).astype(np.float32)  # Ensure Float32 type

        # Create a Bitmap directly from the noise array
        bitmap = mi.Bitmap(noise, pixel_format=mi.Bitmap.PixelFormat.RGB)

        # Save the Bitmap to an EXR file
        bitmap.write(filename, mi.Bitmap.FileFormat.OpenEXR)

        return filename




    def configure_integrator(self):
        """
        Configure the AOV integrator.
        """
        self.integrator = {
            "type": "aov",
            "aovs": ",".join(self.aovs),  # Combine requested AOVs into a single string
        }

    def build_scene(self):
        """
        Assemble the local rendering scene with concentric spheres.
        """
        self.scene = {
            "type": "scene",
            "outer_sphere": self.build_outer_sphere(),
            "sensor": self.build_inner_camera(),
            "integrator": self.configure_integrator(),
        }

    def render(self):
        """
        Perform the local rendering.
        :return: Rendered image as a NumPy array.
        """
        if self.scene is None:
            raise ValueError("Scene is not built. Call build_scene() first.")
        
        # Load and render the scene
        scene = mi.load_dict(self.scene)
        rendered = mi.render(scene)
        
        # Process output (convert Mitsuba output to a NumPy array)
        rendered = np.clip(rendered, 0.0, 1.0)  # Clamp values to [0, 1]
        return (rendered * 255).astype(np.uint8)

    def visualize(self, image):
        """
        Visualize the rendered image using matplotlib.
        :param image: Rendered image as a NumPy array.
        """
        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.axis("off")
        plt.show()


# Example Usage
def main():
    # Initialize the manager with custom settings
    manager = SpheroidCameraManager(
        resolution=(1024, 1024),
        aovs=["radiance"],  # Output radiance
    )
    
    # Build and render the scene
    manager.configure_integrator()
    manager.build_scene()
    rendered_image = manager.render()
    
    # Visualize the output
    manager.visualize(rendered_image)


if __name__ == "__main__":
    main()
