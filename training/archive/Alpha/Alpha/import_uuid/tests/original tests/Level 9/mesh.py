import mitsuba as mi
import numpy as np
import pygame

mi.set_variant("scalar_spectral")
import io
import pymunk
import numpy as np
import torch
class WASDForceController:
    def __init__(self, body, force_magnitude=100):
        """
        Initialize the WASD force controller.
        :param body: The Pymunk body to apply forces to.
        :param force_magnitude: Magnitude of the applied force for each key press.
        """
        self.body = body
        self.force_magnitude = force_magnitude
        self.force_vector = pymunk.Vec2d(0, 0)

    def handle_event(self, event):
        """
        Process key press or release events to update the force vector.
        :param event: A pygame event.
        """
        if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
            increment = self.force_magnitude if event.type == pygame.KEYDOWN else -self.force_magnitude

            if event.key == pygame.K_w:  # Move up
                self.force_vector += pymunk.Vec2d(0, increment)
            elif event.key == pygame.K_s:  # Move down
                self.force_vector += pymunk.Vec2d(0, -increment)
            elif event.key == pygame.K_a:  # Move left
                self.force_vector += pymunk.Vec2d(-increment, 0)
            elif event.key == pygame.K_d:  # Move right
                self.force_vector += pymunk.Vec2d(increment, 0)

    def apply_force(self):
        """
        Apply the accumulated force to the body.
        """
        self.body.apply_force_at_local_point(self.force_vector)
def calculate_normals(vertices, indices):
    """
    Calculate normals for a mesh using vertices and indices.
    :param vertices: A NumPy array of shape (N, 3) for vertex positions.
    :param indices: A NumPy array of shape (M, 3) for triangle indices.
    :return: A NumPy array of shape (N, 3) for vertex normals.
    """
    normals = np.zeros_like(vertices, dtype=np.float32)

    for face in indices:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Calculate the normal for the triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        face_normal /= np.linalg.norm(face_normal)  # Normalize the face normal

        # Accumulate the normal for each vertex
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal

    # Normalize all vertex normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

class SceneIntegratorConfigurator:
    def __init__(self):
        # Default values and supported integrators
        self.integrator = None
        self.integrator_options = {
            "aov": {"aovs": []},
            "path": {"max_depth": 2, "rr_depth": 5, "hide_emitters": False},
            "direct": {"shading_samples": None, "emitter_samples": 1, "bsdf_samples": None, "hide_emitters": False},
            "volpath": {"max_depth": 2, "rr_depth": 5, "hide_emitters": False},
            "volpathmis": {"max_depth": 2, "rr_depth": 5, "hide_emitters": False},
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
        #if integrator_type not in self.integrator_options:
        #    raise ValueError(f"Unsupported integrator type: {integrator_type}")

        # Base configuration with default values
        config = {"type": integrator_type}
        #config.update(self.integrator_options[integrator_type])

        # Override with user-provided parameters
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
            else:
                raise ValueError(f"Invalid parameter '{key}' for integrator type '{integrator_type}'")

        self.integrator = config
        return self.integrator

from fs.memoryfs import MemoryFS
class SpheroidCameraManager:
    def __init__(self, resolution=(512, 512), aovs=None, camera_origin=(0,0,0)):
        """
        Initialize the Spheroid Camera Manager with an inner unit sphere camera and
        an outer 10x unit sphere with inverted normals.
        
        :param resolution: Resolution of the film (width, height).
        :param aovs: List of AOV variables to output (e.g., 'radiance', 'normals').
        """
        self.camera_origin = np.array(camera_origin, dtype=np.float32)  # Fixed camera position
        self.resolution = resolution
        self.aovs = aovs or ["radiance"]  # Default AOV
        self.scene = None
        self.integrator_configurator = SceneIntegratorConfigurator()
        self.memfs = MemoryFS()  # Initialize in-memory filesystem

    def build_inner_camera(self, sensor_type="perspective", film_type="specfilm"):
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

    def vertex_buffer_to_mesh_dict(self, vertices, indices, normals=None, bsdf=None):
        """
        Converts a vertex buffer, indices, and normals into a Mitsuba mesh dictionary.
        """
        if len(vertices) < 3 or len(indices) < 3:
            raise ValueError("Vertex buffer and indices must define a valid mesh.")

        # Calculate normals if they are missing
        if normals is None:
            normals = calculate_normals(vertices, indices)

        bsdf = bsdf or {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.5},  # Default reflectance
        }

        return {
            "type": "mesh",
            "vertex_count": len(vertices),
            "face_count": len(indices),
            "positions": vertices.flatten().tolist(),
            "indices": indices.flatten().tolist(),
            "normals": normals.flatten().tolist(),
            "bsdf": bsdf,
        }



    def build_outer_sphere(self, resolution):
        """
        Define the outer 10x unit sphere with inverted normals and emissive procedural noise texture.
        The upper hemisphere represents the sky, and the lower hemisphere represents the earth.
        """
        # Generate a uniform white noise texture
        noise_texture = self.generate_noise_texture("noise_texture.exr", resolution)
        # Create the sphere with shiny reflective properties
        sphere = {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.scale(10.0),  # Scale to 10x the size of the inner sphere
            "flip_normals": True,  # Invert the normals to point inward
            "bsdf": {  # Define the material properties
                "type": "roughplastic",  # Use a shiny, reflective material
                "diffuse_reflectance": {
                    "type": "bitmap",
                    "filename": noise_texture,  # Use the generated noise texture as the color
                    "filter_type": "nearest",  # Ensure no interpolation
                },
                "specular_reflectance": {
                    "type": "rgb",
                    "value": [0.9, 0.9, 0.9],  # Define the reflectiveness (close to white for shiny)
                },
                "alpha": 0.1,  # Control roughness (lower value = shinier)
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
        self.scene["outer_sphere"]["bsdf"]["diffuse_reflectance"]["filename"] = noise_texture
    def generate_noise_texture(self, filename, size=(100,100)):
        """
        Generate a uniform white noise texture and save it as an EXR file.

        Parameters:
        - filename: Path to save the generated texture file.

        Returns:
        - str: The filename of the generated texture.
        """
        # Create a 256x256 uniform white noise array (RGB)
        width, height = 9, 9#size
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

    def render(self, obj_data=None):
        """
        Perform the local rendering with updated geometry.
        Automatically calculates normals if they are missing.
        """
        if obj_data:
            # Parse OBJ data and ensure normals are calculated
            vertices, indices, normals = self.parse_obj_data(obj_data)

            print(vertices.shape)
            print(indices.shape)
            print(normals.shape)

            # Convert to Mitsuba-compatible mesh dictionary
            dynamic_mesh = self.vertex_buffer_to_mesh_dict(vertices.flatten().flatten().numpy(), indices.flatten().flatten().numpy(), normals.flatten().flatten().numpy())

            # Add the dynamic mesh to the scene
            self.scene["dynamic_object"] = dynamic_mesh

        print("rendered scene")
        self.update()
        # Rebuild and render the scene
        scene = mi.load_dict(self.scene)
        rendered = mi.render(scene)

        # Process output (convert Mitsuba output to a NumPy array)
        rendered = np.clip(rendered, 0.0, 1.0)
        return (rendered * 255).astype(np.uint8)

    def parse_obj_data(self, obj_data):
        """
        Parse OBJ data into vertices, indices, and normals.
        Automatically calculates normals if not provided.
        """
        vertices = []
        normals = []
        indices = []

        for line in obj_data.splitlines():
            if line.startswith("v "):  # Vertex position
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith("vn "):  # Vertex normal
                normals.append(list(map(float, line.split()[1:])))
            elif line.startswith("f "):  # Face indices
                face = [int(part.split("//")[0]) - 1 for part in line.split()[1:]]
                indices.append(face)

        vertices = torch.tensor(vertices, dtype=torch.float32)
        indices = torch.tensor(indices, dtype=torch.int32)

        if normals:
            normals = torch.tensor(normals, dtype=torch.float32)
        else:
            normals = torch.tensor(calculate_normals(vertices.numpy(), indices.numpy()), dtype=torch.float32)

        return vertices, indices, normals

    def update_camera_look_at(self, target_position, up=[0, 1, 0]):
        """
        Update the camera's look-at target dynamically.

        :param target_position: The position the camera should look at (e.g., cube center).
        :param up: The up direction for the camera.
        """
        self.scene["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
            origin=self.camera_origin.tolist(),
            target=target_position,
            up=up
        )
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

def normalize_rendered(rendered, mode='standard', n_unique=256):
    """
    Normalize the rendered image using the specified mode.

    Parameters:
    - rendered (np.ndarray): The rendered image as a NumPy array with float values.
    - mode (str): The normalization mode ('standard' or 'quantized_histogram').
    - n_unique (int): Number of unique values for quantized histogram normalization.

    Returns:
    - np.ndarray: The normalized image as a uint8 NumPy array.
    """
    if mode == 'standard':
        # Standard Normalization
        min_val = rendered.min()
        max_val = rendered.max()
        if max_val - min_val > 1e-15:
            normalized = (rendered - min_val) / (max_val - min_val)
        else:
            # Avoid division by zero if the image is flat
            normalized = np.zeros_like(rendered)
    
    elif mode == 'quantized_histogram':
        # Quantized Histogram Normalization
        # Flatten the array to sort the values
        flattened = rendered.flatten()
        # Get sorted unique values
        unique_vals = np.unique(flattened)
        # Sort the unique values
        sorted_unique = np.sort(unique_vals)
        # Number of quantization levels
        n_levels = n_unique
        # If unique values are less than n_levels, adjust
        if len(sorted_unique) < n_levels:
            n_levels = len(sorted_unique)
        
        # Determine the indices to sample
        indices = np.linspace(0, len(sorted_unique) - 1, n_levels).astype(int)
        # Select the quantized values
        quantized_values = sorted_unique[indices]
        
        # Create a mapping from original to quantized values
        # Use interpolation for values in between quantized levels
        normalized = np.interp(rendered, quantized_values, np.linspace(0, 1, n_levels))
    
    else:
        raise ValueError("Unsupported normalization mode. Choose 'standard' or 'quantized_histogram'.")
    
    # Ensure the normalized values are within [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Convert to 8-bit unsigned integer
    return (normalized * 255).astype(np.uint8)
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
    
    def apply_pymunk_transform(self, position, angle):
        """
        Apply transformations from Pymunk's position and angle to the 3D vertex buffer.
        :param position: New position (x, y) from Pymunk.
        :param angle: Rotation angle (in radians) from Pymunk.
        """
        # Apply 2D translation and rotation to XY components
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        translation = torch.tensor([position[0], position[1], 0], dtype=torch.float32)
        self.vertex_buffer = (rotation_matrix @ self.vertex_buffer.T).T + translation

    def get_2d_slice(self):
        """
        Get a 2D slice of the vertex buffer in the XY plane centered at the origin.
        :return: A PyTorch tensor of 2D vertices [(x, y), ...].
        """
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
    def save_obj_to_memory(self):
        """
        Save the current vertex buffer to an in-memory OBJ format.
        :return: A string containing the OBJ file data.
        """
        vertices = self.vertex_buffer.numpy()
        obj_file = io.StringIO()

        # Write vertices
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Create a default triangulation (fan triangulation for simplicity)
        for i in range(2, len(vertices)):
            obj_file.write(f"f 1 {i} {i+1}\n")
        
        obj_file.seek(0)  # Move to the beginning of the StringIO object
        return obj_file.getvalue()
class PymunkVertexExtractor:
    def __init__(self, circle_radius=30, circle_position=(256, 256), num_segments=32):
        self.space = pymunk.Space()
        self.space.gravity = (0.1, 0.1)
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = circle_position
        self.shape = pymunk.Circle(self.body, circle_radius)
        self.shape.density = 1
        self.space.add(self.body, self.shape)
    
    def step_simulation(self, dt=1 / 60.0):
        self.space.step(dt)
        self.body.angle = self.body.angle + .01
        self.space.gravity = (-self.body.position[0], -self.body.position[1])
    
    def get_pymunk_transform(self):
        """
        Get the current position and angle of the Pymunk body.
        :return: (position, angle) where position is (x, y) and angle is in radians.
        """
        return self.body.position, self.body.angle
def create_mitsuba_cube(position, scale=1.0, rotation=None):
    """
    Create a Mitsuba cube with a given position, scale, and rotation.
    :param position: (x, y, z) position of the cube.
    :param scale: Uniform scale factor.
    :param rotation: Optional rotation (Quaternion or Euler angles).
    :return: Mitsuba-compatible cube dictionary.
    """
    # Start with the default transformation
    to_world = mi.ScalarTransform4f.translate(position) @ mi.ScalarTransform4f.scale(scale)

    # Add rotation if provided
    if rotation:
        # Assuming rotation is a list of Euler angles [x, y, z] in radians
        rot_x = mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=np.degrees(rotation[0]))
        rot_y = mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=np.degrees(rotation[1]))
        rot_z = mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=np.degrees(rotation[2]))
        to_world = to_world @ rot_x @ rot_y @ rot_z

    # Create the cube primitive
    # Create the cube primitive
    cube = {
        "type": "cube",
        "to_world": to_world,
        "emitter":{
            "type": "area",
            "radiance":{
            "type": "blackbody",
            "temperature": 1500}
        }
    }
    return cube


def update_cube_transform(cube, pymunk_body, radius, scale):
    """
    Update the Mitsuba cube's transformation based on Pymunk's x and y values treated as spherical angles.

    :param cube: The Mitsuba cube dictionary.
    :param pymunk_body: The Pymunk body providing x and y values.
    :param radius: The radius of the sphere on which the cube moves.
    """
    # Extract x and y values from Pymunk
    # Normalize x and y to appropriate angular ranges
    theta = ((pymunk_body.position[0] - 256) / 512.0) * 2 * np.pi  # Map x to [0, 2π]
    phi = ((pymunk_body.position[1] - 256) / 512.0) * np.pi        # Map y to [0, π]

    # Convert spherical coordinates to Cartesian
    x_pos = radius * np.sin(phi) * np.cos(theta)
    y_pos = radius * np.sin(phi) * np.sin(theta)
    z_pos = radius * np.cos(phi)

    # Update the cube's position
    position = mi.Vector3f(x_pos, y_pos, z_pos)

    # Apply a basic rotation for visual effect (optional)
    rotation_angle = pymunk_body.angle
    rotation = mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=np.degrees(rotation_angle))

    # Update the cube's transformation
    cube["to_world"] = (
        mi.ScalarTransform4f.translate(position) @ mi.ScalarTransform4f.scale(0.15) @ rotation
    )

def get_cube_center(cube):
    """
    Extract the center of the cube from its transformation matrix.
    :param cube: The Mitsuba cube dictionary.
    :return: A list of the cube's center coordinates in world space.
    """
    transform = cube["to_world"]
    # Get the translation component
    translation = transform.translation()
    return translation

def render_to_pygame(manager, vertex_extractor, vertex_buffer, width=512, height=512, fps=30):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mitsuba Render Viewer")
    clock = pygame.time.Clock()
    running = True

    # Initialize the cube
    initial_position = [0, 0, -1]
    cube = create_mitsuba_cube(initial_position, scale=1)

    # Add to the Mitsuba scene
    manager.scene["dynamic_cube"] = cube

    # Initialize WASDForceController
    wasd_controller = WASDForceController(vertex_extractor.body, force_magnitude=1000)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle WASD key events
            wasd_controller.handle_event(event)

        # Apply forces to the Pymunk body
        wasd_controller.apply_force()

        # Step the Pymunk simulation
        vertex_extractor.step_simulation(dt=1 / fps)

        # Extract transformations from Pymunk
        position, angle = vertex_extractor.get_pymunk_transform()
        vertex_buffer.apply_pymunk_transform(torch.tensor(position), torch.tensor(angle))
        update_cube_transform(manager.scene["dynamic_cube"], vertex_extractor.body, 3, 0.35)
        
        # Report the cube center
        cube_center = get_cube_center(manager.scene["dynamic_cube"])
        manager.update_camera_look_at(cube_center)
        print(f"Cube center in Mitsuba world coordinates: {cube_center}")

        # Render the updated scene
        rendered_image = normalize_rendered(manager.render())

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
    initial_vertices = [
        [.30, 0, 0], 
        [0, .30, 0], 
        [-.30, 0, 0], 
        [0, -.30, 0]
    ]  # Example sphere vertices
    vertex_buffer = VertexBufferManager(initial_vertices)

    vertex_extractor = PymunkVertexExtractor(
        circle_radius=.30, 
        circle_position=(256, 256), 
        num_segments=32
    )

    manager = SpheroidCameraManager(
        resolution=(100, 100),
        aovs=["radiance"],
    )
    manager.build_scene(integrator_type="direct")#, max_depth=3)

    render_to_pygame(manager, vertex_extractor, vertex_buffer, width=512, height=512, fps=10)
