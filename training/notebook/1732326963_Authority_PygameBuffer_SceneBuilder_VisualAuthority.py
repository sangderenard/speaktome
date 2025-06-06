import mitsuba as mi
import numpy as np
import pygame
import time

mi.set_variant("scalar_rgb")

# Global World Dictionary
WORLD_CUBE = {
    "type": "cube",
    "to_world": mi.ScalarTransform4f.scale(25.0),  # Scale the cube to contain the scene
    "flip_normals": True,  # Flip normals to make the cube face inward
    "emitter": {
        "type": "area",
        "radiance": {
            "type": "rgb",
            "value": 0.8  # Light grey for the interior walls
        }
    }
}

# Authority Base Class
class Authority:
    def __init__(self, name, state):
        """
        Initialize the authority with a name and initial state.
        :param name: Unique identifier for the authority.
        :param state: Dictionary representing the current state.
        """
        self.name = name
        self.state = state

    def update_state(self, input_data):
        """
        Update the authority's state based on input data.
        :param input_data: Data from connected authorities.
        """
        pass  # To be implemented by subclasses

    def get_output(self):
        """
        Retrieve the current state as output data.
        :return: Dictionary representing the output state.
        """
        return self.state
def quaternion_to_transform4f(quaternion, translation):
    """
    Convert a quaternion and translation vector into a Mitsuba Transform4f matrix.
    
    :param quaternion: List or numpy array [x, y, z, w]
    :param translation: List or numpy array [tx, ty, tz]
    :return: Mitsuba Transform4f representing the combined rotation and translation.
    """
    x, y, z, w = quaternion
    tx, ty, tz = translation

    # Normalize the quaternion to avoid scaling issues
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Compute the rotation matrix
    rotation = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),   1 - 2 * (x**2 + y**2)]
    ])

    # Construct the 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]

    # Return as a Mitsuba Transform4f
    return mi.ScalarTransform4f(transform)


class SceneBuilder:
    def __init__(self, screen_width=800, screen_height=600):
        """
        Initialize the SceneBuilder with default parameters.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize positions and orientations
        self.cube_position = np.array([0, 0, 0])
        self.cube_orientation = np.array([0, 0, 0, 1])  # Quaternion (x, y, z, w)

        self.camera_position = np.array([3, 3, 3])
        self.camera_orientation = np.array([0, 0, 0, 1])  # Quaternion

        self.light_position = np.array([2, 2, 2])
        self.light_orientation = np.array([0, 0, 0, 1])  # Quaternion

        # Initialize scene configuration with the inverted world cube
        self.scene_config = {
            "type": "scene",
            "integrator": {
                "type": "path",
                "max_depth": 2
            },
            "emitter": self.create_spot_light(),
            "sensor": self.create_camera(),
            "world_cube": WORLD_CUBE,
            "dynamic_cube": None,
            "environment_light": {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": 0.2  # Light grey for subtle ambient light
                }
            }

        }

        # Add the dynamic cube to the scene
        self.add_dynamic_cube()
    def create_spot_light(self):
        """
        Create a spot light emitter with a proper to_world transformation.
        """
        to_world = quaternion_to_transform4f(self.light_orientation, self.light_position)
        return {
            "type": "spot",
            "to_world": to_world,  # Use the combined rotation and translation
            "cutoff_angle": 30,  # Degrees
            "intensity": {
                "type": "rgb",
                "value": (1.0, 1.0, 1.0)  # Normalized RGB for bright white light
            },
            #"falloff": 2.0  # How the light intensity decreases
        }


    def create_camera(self):
        """
        Create a perspective camera.
        """
        return {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=tuple(self.camera_position),
                target=tuple(self.cube_position),
                up=(0, 0, 1)  # Use a tuple
            ),
            "film": {
                "type": "hdrfilm",
                "width": self.screen_width,
                "height": self.screen_height,
                #"rfilter": {"type": "gaussian"}
            },
            "sampler": {
                "type": "independent",
                "sample_count": 10
            }
        }

    def add_dynamic_cube(self):
        """
        Add a dynamic cube object to the scene with a proper to_world transformation.
        """
        to_world = quaternion_to_transform4f(self.cube_orientation, self.cube_position)
        cube = {
            "type": "cube",
            "to_world": to_world,  # Proper transformation
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": (0.7, 0.2, 0.2)  # Normalized to range [0, 1]
                }
            }
        }
        self.scene_config["dynamic_cube"] = cube  # Add cube to the objects dictionary

    def update_dynamic_cube(self, position=None, orientation=None):
        """
        Update the dynamic cube's position and orientation.
        """
        if position is not None:
            self.cube_position = np.array(position)
        if orientation is not None:
            self.cube_orientation = np.array(orientation)

        to_world = quaternion_to_transform4f(self.cube_orientation, self.cube_position)
        self.scene_config["dynamic_cube"]["to_world"] = to_world

    def render_scene(self):
        """
        Render the current scene using Mitsuba and return the image as a NumPy array.
        """
        scene = mi.load_dict(self.scene_config)
        print(scene)
        image = mi.render(scene, spp=1)  # Single sample per pixel for simplicity
        image = np.clip(image, 0, 1) * 255
        return image.astype(np.uint8)


# PygameBuffer
class PygameBuffer:
    def __init__(self, screen):
        """
        Initialize the PygameBuffer with a Pygame screen.
        :param screen: Pygame display surface.
        """
        self.screen = screen

    def update(self, rendered_image):
        """
        Update the Pygame buffer with the rendered image.
        :param rendered_image: Image data as a NumPy array (H x W x 3).
        """
        # Convert the rendered image to a Pygame surface
        pygame_image = pygame.surfarray.make_surface(np.flipud(rendered_image.swapaxes(0, 1)))
        self.screen.blit(pygame_image, (0, 0))

    def display(self):
        """
        Update the Pygame display.
        """
        pygame.display.flip()

# VisualAuthority
class VisualAuthority(Authority):
    def __init__(self, name, state, scene_builder, pygame_buffer):
        """
        Initialize the VisualAuthority with a SceneBuilder and PygameBuffer.
        :param name: Unique identifier for the authority.
        :param state: Dictionary representing the current state.
        :param scene_builder: Instance of SceneBuilder.
        :param pygame_buffer: Instance of PygameBuffer.
        """
        super().__init__(name, state)
        self.scene_builder = scene_builder
        self.pygame_buffer = pygame_buffer

    def update_state(self, input_data):
        """
        Update the visual scene based on input parameters.
        """
        # No input changes for this minimal example
        rendered_image = self.scene_builder.render_scene()
        self.pygame_buffer.update(rendered_image)

# Main Execution
if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mitsuba Scene Renderer")

    # Initialize SceneBuilder
    scene_builder = SceneBuilder(screen_width=WIDTH, screen_height=HEIGHT)

    # Initialize PygameBuffer
    pygame_buffer = PygameBuffer(screen)

    # Initialize VisualAuthority
    visual_state = {"dt": 1.0}
    visual_authority = VisualAuthority("Visual", visual_state, scene_builder, pygame_buffer)

    # Rendering Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Render the scene
        visual_authority.update_state({})
        current_image = scene_builder.render_scene()
        print(current_image.mean())
        pygame_buffer.update(current_image)

        # Display the rendered image
        pygame_buffer.display()

        # Slow down to 1 frame per second
        time.sleep(1)

    pygame.quit()
