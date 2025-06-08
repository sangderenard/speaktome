import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")

import pymunk
import numpy as np
import numpy as np
import torch
import pymunk

class VertexBufferManager:
    def __init__(self, initial_vertices=None):
        """
        Initialize the VertexBufferManager with an optional initial set of vertices.
        :param initial_vertices: A list or array of initial vertices [(x, y, z), ...].
        """
        if initial_vertices is None:
            initial_vertices = []
        self.vertex_buffer = torch.tensor(initial_vertices, dtype=torch.float32)
    
    def update_vertex_buffer(self, new_vertices):
        """
        Update the vertex buffer with new vertices.
        :param new_vertices: A list or array of new vertices [(x, y, z), ...].
        """
        self.vertex_buffer = torch.tensor(new_vertices, dtype=torch.float32)
    
    def get_2d_slice(self):
        """
        Get a 2D slice of the vertex buffer in the XY plane centered at the origin.
        :return: A PyTorch tensor of 2D vertices [(x, y), ...].
        """
        # Extract the XY components and center at the origin
        xy_vertices = self.vertex_buffer[:, :2]
        center = torch.mean(xy_vertices, dim=0)
        return xy_vertices - center
    
    def create_pymunk_shape(self):
        """
        Create a Pymunk shape from the 2D slice of the vertex buffer.
        :return: A Pymunk Poly shape.
        """
        xy_slice = self.get_2d_slice().numpy()
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        shape = pymunk.Poly(body, xy_slice.tolist())
        return body, shape
    
    def write_obj_file(self, filename):
        """
        Write the current vertex buffer to an OBJ file.
        :param filename: Name of the .obj file to create.
        """
        vertices = self.vertex_buffer.numpy()
        with open(filename, 'w') as obj_file:
            # Write vertices
            for vertex in vertices:
                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # Create a default triangulation (fan triangulation for simplicity)
            for i in range(2, len(vertices)):
                obj_file.write(f"f 1 {i} {i+1}\n")

    def __str__(self):
        """
        String representation of the vertex buffer.
        """
        return f"VertexBufferManager with {len(self.vertex_buffer)} vertices:\n{self.vertex_buffer}"


class PymunkVertexExtractor:
    def __init__(self, circle_radius=30, circle_position=(256, 256), num_segments=32):
        """
        Initialize Pymunk and prepare for vertex extraction.
        :param circle_radius: Radius of the circle.
        :param circle_position: Initial position of the circle.
        :param num_segments: Number of segments to approximate the circle.
        """
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.num_segments = num_segments

        # Create a circle in the Pymunk simulation
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = circle_position
        self.shape = pymunk.Circle(self.body, circle_radius)
        self.shape.density = 1
        self.space.add(self.body, self.shape)

    def step_simulation(self, dt=1 / 60.0):
        """
        Step the Pymunk simulation forward.
        :param dt: Time step for the simulation.
        """
        self.space.step(dt)
    def extract_vertices(self, thickness=0.1):
        """
        Extract the vertices and indices approximating the circle with added thickness.
        :param thickness: The thickness of the extruded shape along the z-axis.
        :return: A tuple of (vertices, indices, normals) for Mitsuba.
        """
        vertices = []
        indices = []
        normals = []

        center = self.body.position
        radius = self.shape.radius

        # Generate vertices for the top and bottom layers
        top_layer = []
        bottom_layer = []

        for i in range(self.num_segments):
            angle = 2 * np.pi * i / self.num_segments
            x = center.x + radius * np.cos(angle)
            y = center.y + radius * np.sin(angle)
            z_top = thickness / 2
            z_bottom = -thickness / 2

            # Add vertices to top and bottom layers
            top_layer.append((x, y, z_top))
            bottom_layer.append((x, y, z_bottom))

            # Approximate normals (pointing outward)
            normal = (np.cos(angle), np.sin(angle), 0)
            normals.append(normal)

        vertices.extend(top_layer)
        vertices.extend(bottom_layer)

        # Triangulate the top and bottom layers (fan triangulation)
        for i in range(1, self.num_segments - 1):
            # Top layer
            indices.extend([0, i, i + 1])
            # Bottom layer (reversed order for correct orientation)
            indices.extend([self.num_segments, self.num_segments + i, self.num_segments + i + 1])

        # Connect the layers to form the side walls
        for i in range(self.num_segments):
            next_i = (i + 1) % self.num_segments

            # Side triangles
            indices.extend([i, next_i, self.num_segments + i])       # Top to bottom
            indices.extend([self.num_segments + i, next_i, self.num_segments + next_i])  # Bottom to top

        return np.array(vertices, dtype=np.float32), indices, np.array(normals, dtype=np.float32)



class SceneIntegratorConfigurator:
    def __init__(self):
        # Default values and supported integrators
        self.integrator = None
        self.integrator_options = {
            "aov": {"aovs": []},
            "path": {"max_depth": 1, "rr_depth": 5, "hide_emitters": False},
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
        self.integrator_configurator = SceneIntegratorConfigurator()

    def build_inner_camera(self, sensor_type="thinlens", film_type="hdrfilm"):
        """
        Define the inner unit sphere camera with configurable sensor and film types.

        :param sensor_type: Type of sensor ('perspective' or 'thinlens').
        :param film_type: Type of film ('hdrfilm' or 'specfilm').
        """
        # Validate input parameters
        if sensor_type not in ["perspective", "thinlens"]:
            raise ValueError(f"Invalid sensor type: {sensor_type}. Choose 'perspective' or 'thinlens'.")
        if film_type not in ["hdrfilm", "specfilm"]:
            raise ValueError(f"Invalid film type: {film_type}. Choose 'hdrfilm' or 'specfilm'.")

        # Base sensor configuration
        sensor = {
            "type": sensor_type,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0],  # Center of the unit sphere
                target=[0, 0, -1],  # Looking forward
                up=[0, 1, 0]
            ),
        }

        # Add aperture radius for thinlens sensor
        if sensor_type == "thinlens":
            sensor["aperture_radius"] = 0.2

        # Configure film based on the chosen type
        if film_type == "hdrfilm":
            sensor["film"] = {
                "type": "hdrfilm",
                "width": self.resolution[0],
                "height": self.resolution[1],
                "rfilter": {"type": "gaussian"},
            }
        elif film_type == "specfilm":
            sensor["film"] = {
                "type": "specfilm",
                "width": self.resolution[0],
                "height": self.resolution[1],
                "component_format": "float32",
                "band1_red": {
                    "type": "spectrum",
                    "filename": "spectral_data/data_red.spd"
                },
                "band2_green": {
                    "type": "spectrum",
                    "filename": "spectral_data/data_green.spd"
                },
                "band3_blue": {
                    "type": "spectrum",
                    "filename": "spectral_data/data_blue.spd"
                },
            }

        # Add default sampler
        sensor["sampler"] = {"type": "independent", "sample_count": 1}

        return sensor

    def vertex_buffer_to_mesh_dict(self, vertices, indices, normals, bsdf=None):
        """
        Converts a vertex buffer, indices, and normals into a Mitsuba mesh dictionary.
        """
        if len(vertices) < 3 or len(indices) < 3:
            raise ValueError("Vertex buffer and indices must define a valid mesh.")

        bsdf = bsdf or {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.5},  # Default spectral reflectance
        }

        return {
            "type": "mesh",
            "vertex_count": len(vertices),
            "face_count": len(indices) // 3,
            "positions": vertices.flatten().tolist(),
            "indices": indices,
            "normals": normals.flatten().tolist(),
            #"bsdf": bsdf,
        }


    def build_outer_sphere(self, resolution):
        """
        Define the outer 10x unit sphere with inverted normals and emissive procedural noise texture.
        The upper hemisphere represents the sky, and the lower hemisphere represents the earth.
        """
        # Generate a uniform white noise texture
        noise_texture = self.generate_noise_texture("noise_texture.exr", resolution)

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
    def update(self):
        """
        Update the scene by regenerating the noise texture and updating the outer sphere's configuration.
        """
        # Regenerate the noise texture
        noise_texture = self.generate_noise_texture("noise_texture.exr")

        # Update the emissive texture of the outer sphere
        self.scene["outer_sphere"]["emitter"]["radiance"]["filename"] = noise_texture
    def generate_noise_texture(self, filename, size=(100,100)):
        """
        Generate a uniform white noise texture and save it as an EXR file.

        Parameters:
        - filename: Path to save the generated texture file.

        Returns:
        - str: The filename of the generated texture.
        """
        # Create a 256x256 uniform white noise array (RGB)
        width, height = size
        noise = np.random.rand(height, width).astype(np.float32)  # Ensure Float32 type

        # Create a Bitmap directly from the noise array
        bitmap = mi.Bitmap(noise, pixel_format=mi.Bitmap.PixelFormat.Y)

        # Save the Bitmap to an EXR file
        bitmap.write(filename, mi.Bitmap.FileFormat.OpenEXR)

        return filename

    def configure_integrator(self, integrator_type="path", **kwargs):
        """
        Configure the integrator using the SceneIntegratorConfigurator instance.
        """
        return self.integrator_configurator.configure_integrator(integrator_type, **kwargs)

    def build_scene(self, integrator_type="path", **integrator_kwargs):
        """
        Assemble the local rendering scene with concentric spheres.
        :param integrator_type: Type of integrator to configure.
        :param integrator_kwargs: Additional parameters for the integrator.
        """
        integrator = self.configure_integrator(integrator_type, **integrator_kwargs)
        self.scene = {
            "type": "scene",
            "outer_sphere": self.build_outer_sphere(self.resolution),
            "sensor": self.build_inner_camera(),
            "integrator": integrator,
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

        print("hello")
        plt.imshow(image)
        plt.axis("off")
        plt.show()


import pygame
import numpy as np
import mitsuba as mi

def render_to_pygame(manager, vertex_extractor, width=512, height=512, fps=30):
    """
    Continuously render the scene to a pygame screen, updating the texture with Pymunk vertices.

    :param manager: An instance of SpheroidCameraManager.
    :param vertex_extractor: An instance of PymunkVertexExtractor.
    :param width: Width of the pygame window.
    :param height: Height of the pygame window.
    :param fps: Frames per second for the rendering loop.
    """
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mitsuba Render Viewer")
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step the Pymunk simulation
        vertex_extractor.step_simulation(dt=1 / fps)

        # Extract vertices and create a mesh
        vertices, indices, normals = vertex_extractor.extract_vertices()
        mesh_dict = manager.vertex_buffer_to_mesh_dict(vertices, indices, normals)
        print(mesh_dict)
        # Update the scene with the new mesh
        manager.scene["dynamic_mesh"] = mesh_dict

        # Render the updated scene
        rendered_image = manager.render()

        # Convert the rendered image to a pygame surface
        rendered_image = pygame.surfarray.make_surface(rendered_image.transpose(1, 0, 2))

        # Scale the surface to fit the pygame window
        rendered_image = pygame.transform.scale(rendered_image, (width, height))

        # Draw the surface to the screen
        screen.blit(rendered_image, (0, 0))
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(fps)

    pygame.quit()



if __name__ == "__main__":
    # Initialize the Pymunk vertex extractor
    # List available plugins

    vertex_extractor = PymunkVertexExtractor(
        circle_radius=30, 
        circle_position=(256, 256), 
        num_segments=32
    )

    # Initialize the SpheroidCameraManager
    manager = SpheroidCameraManager(
        resolution=(512, 512),
        aovs=["radiance"],  # Output radiance
    )
    
    # Build the initial scene
    manager.build_scene(integrator_type="path", max_depth=2)

    # Render continuously to a pygame window with dynamic vertex updates
    render_to_pygame(manager, vertex_extractor, width=512, height=512, fps=30)
