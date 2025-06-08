import mitsuba as mi
import numpy as np
import pygame

mi.set_variant("scalar_rgb")
import io
import pymunk
import numpy as np
import torch
from fs.memoryfs import MemoryFS

class WASDForceController:
    def __init__(self, square_manager, force_magnitude=100):
        """
        Initialize the WASD force controller.
        :param square_manager: The PymunkSquareManager instance managing the squares.
        :param force_magnitude: Magnitude of the applied force for each key press.
        """
        self.square_manager = square_manager
        self.force_magnitude = force_magnitude
        self.force_vectors = [pymunk.Vec2d(0, 0) for _ in range(self.square_manager.num_squares)]

    def handle_event(self, event, camera_selector):
        """
        Process key press or release events to update the force vectors or select camera target.
        :param event: A pygame event.
        :param camera_selector: The CameraSelector instance to handle target selection.
        """
        if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
            increment = self.force_magnitude if event.type == pygame.KEYDOWN else -self.force_magnitude

            if event.key == pygame.K_w:  # Move up
                for i in range(self.square_manager.num_squares):
                    self.force_vectors[i] += pymunk.Vec2d(0, increment)
            elif event.key == pygame.K_s:  # Move down
                for i in range(self.square_manager.num_squares):
                    self.force_vectors[i] += pymunk.Vec2d(0, -increment)
            elif event.key == pygame.K_a:  # Move left
                for i in range(self.square_manager.num_squares):
                    self.force_vectors[i] += pymunk.Vec2d(-increment, 0)
            elif event.key == pygame.K_d:  # Move right
                for i in range(self.square_manager.num_squares):
                    self.force_vectors[i] += pymunk.Vec2d(increment, 0)
            elif pygame.K_1 <= event.key <= pygame.K_9:  # Number keys 1-9
                number = event.key - pygame.K_0  # Convert key to number
                camera_selector.select_target(number - 1)  # Zero-based index

    def apply_force(self):
        """
        Apply the accumulated forces to the bodies.
        """
        for i, force in enumerate(self.force_vectors):
            self.square_manager.bodies[i].apply_force_at_local_point(force)
            self.force_vectors[i] = pymunk.Vec2d(0, 0)  # Reset force after applying

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
        norm = np.linalg.norm(face_normal)
        if norm != 0:
            face_normal /= norm  # Normalize the face normal
        else:
            face_normal = np.array([0, 0, 1], dtype=np.float32)  # Default normal

        # Accumulate the normal for each vertex
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal

    # Normalize all vertex normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    normals /= norms
    return normals

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
    def __init__(self, resolution=(512, 512), aovs=None, camera_origin=(0,0,0)):
        """
        Initialize the Spheroid Camera Manager with an inner unit sphere camera and
        an outer 10x unit sphere with inverted normals.
        
        :param resolution: Resolution of the film (width, height).
        :param aovs: List of AOV variables to output (e.g., 'radiance', 'normals').
        :param camera_origin: The fixed position of the camera in world space.
        """
        self.camera_origin = np.array(camera_origin, dtype=np.float32)  # Fixed camera position
        self.resolution = resolution
        self.aovs = aovs or ["radiance"]  # Default AOV
        self.scene = None
        self.integrator_configurator = SceneIntegratorConfigurator()
        self.memfs = MemoryFS()  # Initialize in-memory filesystem

    def build_inner_camera(self, sensor_type="perspective", film_type="hdrfilm"):
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
                origin=self.camera_origin.tolist(),
                target=[0, 0, -1],  # Initial target; will be updated dynamically
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
            dynamic_mesh = self.vertex_buffer_to_mesh_dict(
                vertices.flatten().numpy(),
                indices.flatten().numpy(),
                normals.flatten().numpy()
            )

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

class PymunkSquareManager:
    def __init__(self, num_squares=4, square_size=30, initial_positions=None, space_gravity=(0.0, 0.0)):
        """
        Initialize the PymunkSquareManager with multiple squares.
        :param num_squares: Number of squares to initialize.
        :param square_size: Size (half-width) of each square.
        :param initial_positions: List of initial positions for each square [(x, y), ...].
        :param space_gravity: Gravity vector for the Pymunk space.
        """
        self.space = pymunk.Space()
        self.space.gravity = space_gravity
        self.num_squares = num_squares
        self.bodies = []
        self.shapes = []

        if initial_positions is None:
            # Arrange squares in a grid if no initial positions provided
            spacing = 100
            initial_positions = [
                (spacing * (i % 2), spacing * (i // 2)) for i in range(num_squares)
            ]

        for pos in initial_positions:
            body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            body.position = pos
            shape = pymunk.Poly.create_box(body, size=(square_size, square_size))
            shape.density = 1
            self.space.add(body, shape)
            self.bodies.append(body)
            self.shapes.append(shape)

    def step_simulation(self, dt=1 / 60.0):
        self.space.step(dt)
        for body in self.bodies:
            body.angle += 0.01
            # Optional: Add damping or other physics properties if needed

    def get_transforms(self):
        """
        Get the current position and angle of all Pymunk bodies.
        :return: List of tuples [(position, angle), ...] for each body.
        """
        transforms = []
        for body in self.bodies:
            transforms.append((body.position, body.angle))
        return transforms

class CameraSelector:
    def __init__(self, num_targets):
        """
        Initialize the CameraSelector to manage camera target selection.
        :param num_targets: Total number of possible targets.
        """
        self.num_targets = num_targets
        self.selected_index = 0  # Default to first target

    def select_target(self, index):
        """
        Select a target based on the given index.
        :param index: Zero-based index of the target to select.
        """
        if 0 <= index < self.num_targets:
            self.selected_index = index
            print(f"Camera target selected: {self.selected_index + 1}")
        else:
            print(f"Invalid target index: {index + 1}")

    def get_selected_index(self):
        """
        Get the currently selected target index.
        :return: Zero-based index of the selected target.
        """
        return self.selected_index

def create_mitsuba_cube(position, scale=1.0, rotation=None):
    """
    Create a Mitsuba cube with a given position, scale, and rotation.
    :param position: (x, y, z) position of the cube.
    :param scale: Uniform scale factor.
    :param rotation: Optional rotation (degrees) around the Z-axis.
    :return: Mitsuba-compatible cube dictionary.
    """
    # Start with the default transformation
    to_world = mi.ScalarTransform4f.translate(position) @ mi.ScalarTransform4f.scale(scale)

    # Add rotation if provided
    if rotation:
        # Assuming rotation is a single angle around Z-axis in degrees
        rot_z = mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=rotation)
        to_world = to_world @ rot_z

    # Create the cube primitive
    cube = {
        "type": "cube",
        "to_world": to_world,
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {"type": "rgb", "value": [1.0, 0.4, 0.65]},  # Color A
                "color1": {"type": "rgb", "value": [0.2, 0.0, 0.4]},   # Color B
                "to_uv": mi.ScalarTransform4f.scale([1, 1, 1]),        # Adjust scale of the checkerboard
            },
        },
    }
    return cube

def update_cube_transform(cube, pymunk_body, scale=1.0):
    """
    Update the Mitsuba cube's transformation based on Pymunk's position and angle.

    :param cube: The Mitsuba cube dictionary.
    :param pymunk_body: The Pymunk body providing position and angle.
    :param scale: Scale factor for the cube.
    """
    # Extract position and angle from Pymunk
    x, y = pymunk_body.position
    angle = pymunk_body.angle

    # Convert Pymunk's position to Mitsuba's coordinate system (assuming Z is up)
    position = mi.Vector3f(x / 100.0, y / 100.0, 0)  # Scaling down for visualization

    # Convert angle from radians to degrees
    rotation_angle = np.degrees(angle)

    # Update the cube's transformation
    cube["to_world"] = (
        mi.ScalarTransform4f.translate(position) @
        mi.ScalarTransform4f.scale(scale) @
        mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=rotation_angle)
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

def render_to_pygame(manager, square_manager, cubes, camera_selector, width=512, height=512, fps=30):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mitsuba Render Viewer")
    clock = pygame.time.Clock()
    running = True

    # Initialize WASDForceController
    wasd_controller = WASDForceController(square_manager, force_magnitude=1000)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle WASD and number key events
            wasd_controller.handle_event(event, camera_selector)

        # Apply forces to the Pymunk bodies
        wasd_controller.apply_force()

        # Step the Pymunk simulation
        square_manager.step_simulation(dt=1 / fps)

        # Extract transformations from Pymunk
        transforms = square_manager.get_transforms()

        # Update each cube's transformation
        for i, (position, angle) in enumerate(transforms):
            update_cube_transform(cubes[i], square_manager.bodies[i], scale=0.15)

        # Get the selected cube's center for camera targeting
        selected_index = camera_selector.get_selected_index()
        if 0 <= selected_index < len(cubes):
            cube_center = get_cube_center(cubes[selected_index])
            manager.update_camera_look_at(cube_center)
            print(f"Camera is targeting Cube {selected_index + 1} at {cube_center}")

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
    # Define the number of squares
    num_squares = 4

    # Define initial positions for squares
    initial_positions = [
        (200, 200),
        (300, 200),
        (200, 300),
        (300, 300)
    ]

    # Initialize the PymunkSquareManager
    square_manager = PymunkSquareManager(
        num_squares=num_squares,
        square_size=30,
        initial_positions=initial_positions,
        space_gravity=(0.0, 0.0)  # No gravity
    )

    # Initialize the CameraSelector
    camera_selector = CameraSelector(num_targets=num_squares)

    # Initialize the SpheroidCameraManager
    manager = SpheroidCameraManager(
        resolution=(256, 256),
        aovs=["radiance"],
        camera_origin=(0, 0, 10)  # Positioned along the Z-axis
    )
    manager.build_scene(integrator_type="path", max_depth=10)

    # Initialize Mitsuba cubes for each Pymunk square
    cubes = []
    for i in range(num_squares):
        # Initial position is scaled down for Mitsuba's coordinate system
        initial_pos = square_manager.bodies[i].position
        cube = create_mitsuba_cube(position=(initial_pos.x / 100.0, initial_pos.y / 100.0, 0), scale=0.15)
        cubes.append(cube)
        manager.scene[f"dynamic_cube_{i}"] = cube

    # Start the rendering loop
    render_to_pygame(manager, square_manager, cubes, camera_selector, width=512, height=512, fps=30)
