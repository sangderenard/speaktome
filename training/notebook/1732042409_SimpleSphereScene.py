import mitsuba as mi
import os

# Activate Mitsuba's Python bindings in spectral mode
mi.set_variant('scalar_spectral')


class SimpleSphereScene:
    def __init__(self, spot_distance=1.0, point_distance_multiplier=10.0):
        """
        Initializes the scene with a small sphere, a thin lens camera, a constant environment emitter,
        a forward-facing spot light, and multiple point light sources.

        Parameters:
            spot_distance (float): Distance of the spot light from the camera.
            point_distance_multiplier (float): Multiplier to determine the distance of point lights relative to the spot light.
        """
        # Define the sphere geometry with a simple diffuse BSDF
        sphere = {
            'sphere': {  # Unique identifier for the sphere
                'type': 'sphere',
                'radius': 0.5,
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': [0.7, 0.7, 0.7]  # Grey color
                    }
                }
            }
        }

        # Define the constant environment emitter for uniform spectral illumination
        env_emitter = {
            'env_emitter': {  # Unique identifier for the environment emitter
                'type': 'constant',
                'radiance': {
                    'type': 'spectrum',
                    'value': 1.0  # Spectral radiance
                }
            }
        }

        # Define the forward-facing spot light positioned to the left of the camera
        spot_emitter = {
            'spot_emitter': {  # Unique identifier for the spot emitter
                'type': 'spot',
                'intensity': {
                    'type': 'spectrum',
                    'value': 1.0  # Spectral intensity
                },
                'cutoff_angle': 20.0,  # Degrees
                'beam_width': 30.0,    # Degrees
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=[-spot_distance, 0, 0],  # Positioned to the left along the X-axis
                    target=[0, 0, 0],                # Looking towards the origin (sphere)
                    up=[0, 1, 0]                      # Up direction
                )
            }
        }

        # Define point light emitters positioned to the right, below, and behind the camera
        point_emitters = {}
        # Calculate positions based on the spot_distance and multiplier
        point_positions = {
            'point_emitter_right': [spot_distance * point_distance_multiplier, 0, 0],    # Right of the camera
            'point_emitter_below': [0, -spot_distance * point_distance_multiplier, 0],   # Below the camera
            'point_emitter_behind': [0, 0, spot_distance * point_distance_multiplier]     # Behind the camera
        }

        # Intensity for point lights is 10x that of the spot light
        point_intensity = 10.0

        for name, pos in point_positions.items():
            point_emitters[name] = {
                'type': 'point',
                'position': pos,
                'intensity': {
                    'type': 'spectrum',
                    'value': point_intensity
                }
            }

        # Define the thin lens camera positioned a short distance away from the sphere
        camera = {
            'sensor': {  # Unique identifier for the sensor
                'type': 'thinlens',
                'fov': 45,  # Field of view in degrees
                'aperture_radius': 0.1,
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=[0, 0, -spot_distance * 1.5],  # Positioned along the negative Z-axis
                    target=[0, 0, 0],                      # Looking towards the origin (sphere)
                    up=[0, 1, 0]                            # Up direction
                )
            },
                    # Define the film (image output settings)
            'film' : {
                'type': 'specfilm',
                    'width': 1024,
                    'height': 768,
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
        }


        # Compile all emitters: environment, spot, and point lights
        emitters = {**env_emitter, **spot_emitter, **point_emitters}

        # Construct the scene dictionary with unique keys for each component
        scene_dict = {
            'type': 'scene',
            **emitters,  # Unpack all emitters into the scene
            **sphere,    # Unpack the sphere into the scene
            **camera     # Unpack the sensor into the scene
        }

        # Load the scene using Mitsuba's load_dict function
        self.scene = mi.load_dict(scene_dict)

    def render(self, output_path='output.exr', spp=16, resolution=(800, 600)):
        """
        Renders the scene and saves the output image.

        Parameters:
            output_path (str): File path to save the rendered image.
            spp (int): Samples per pixel for rendering.
            resolution (tuple): Resolution of the output image (width, height).
        """
        # Define the integrator (path tracing)
        integrator = {
            'type': 'path',
            'max_depth': 5  # Maximum recursion depth
        }



        # Retrieve the sensor from the scene
        sensor = self.scene.sensors()[0]
        # Assign the film to the sensor
        

        # Define the integrator settings
        integrator_obj = mi.load_dict(integrator)

        # Create the rendering scene with the integrator and sensor
        scene = mi.Scene(
            objects=self.scene.objects(),
            sensors=[sensor],
            integrator=integrator_obj
        )

        # Perform the rendering
        print("Rendering started...")
        image = mi.render(scene, spp=spp)
        print("Rendering completed.")

        # Save the rendered image
        image.write(output_path)
        print(f"Image saved to {output_path}")


def main():
    """
    Demonstrates the SimpleSphereScene by rendering the scene.
    """
    # Instantiate the scene with default distances
    scene = SimpleSphereScene(spot_distance=1.0, point_distance_multiplier=10.0)
    
    # Define the output path and rendering parameters
    output_image_path = 'rendered_sphere.exr'
    samples_per_pixel = 64
    image_resolution = (1024, 768)
    
    # Render the scene
    scene.render(output_path=output_image_path, spp=samples_per_pixel, resolution=image_resolution)


if __name__ == '__main__':
    main()
