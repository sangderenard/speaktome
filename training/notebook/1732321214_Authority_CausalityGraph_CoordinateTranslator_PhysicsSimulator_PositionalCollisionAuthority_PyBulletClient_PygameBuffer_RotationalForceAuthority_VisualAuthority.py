#wholistic.py

import pygame
import pymunk
import pymunk.pygame_util
import pybullet as p
import pybullet_data
import mitsuba as mi
import numpy as np
import time
import math
from collections import defaultdict

# ===========================
# Constants and Configuration
# ===========================

WIDTH, HEIGHT = 800, 600
FPS = 60
SIM_FPS = 30
DT = 1.0 / SIM_FPS

# Shared Coordinate System Scaling
SCALE = 100  # Pixels per meter

# Gravity Settings
PYBULLET_GRAVITY = (0, 0, 0)  # No gravity in PyBullet
PYMUNK_GRAVITY = (0, 900)     # Flipped Y for Pymunk (pixels/sec²)

# ===========================
# Authority Base Class
# ===========================

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

# ===========================
# VisualAuthority Class
# ===========================

class VisualAuthority(Authority):
    def __init__(self, name, state, mitsuba_config, pygame_buffer):
        super().__init__(name, state)
        self.mitsuba_config = mitsuba_config  # Mitsuba XML-style Python dictionary
        self.pygame_buffer = pygame_buffer    # Pygame rendering buffer

    def update_state(self, input_data):
        """
        Update the visual scene based on input parameters.
        :param input_data: Dictionary containing positions of cube, light, and camera.
        """
        # Extract parameters
        cube_pos = input_data.get('cube_pos', self.mitsuba_config['objects']['cube']['position'])
        light_pos = input_data.get('light_pos', self.mitsuba_config['objects']['light']['position'])
        camera_pos = input_data.get('camera_pos', self.mitsuba_config['objects']['camera']['position'])

        # Update Mitsuba configuration
        self.mitsuba_config['objects']['cube']['position'] = cube_pos
        self.mitsuba_config['objects']['light']['position'] = light_pos
        self.mitsuba_config['objects']['camera']['position'] = camera_pos

        # Render the scene using Mitsuba and update Pygame buffer
        rendered_image = self.render_scene()
        self.pygame_buffer.update(rendered_image)

    def render_scene(self):
        """
        Render the scene using Mitsuba.
        :return: Rendered image data as a NumPy array.
        """
        # Load the scene from the configuration
        scene = mi.load_dict(self.mitsuba_config)
        
        # Render the scene
        rendered = mi.render(scene, spp=1)
        
        # Convert Mitsuba output to NumPy array
        rendered_image = np.clip(rendered, 0, 1) * 255
        rendered_image = rendered_image.astype(np.uint8)
        
        return rendered_image

# ===========================
# RotationalForceAuthority Class
# ===========================

class RotationalForceAuthority(Authority):
    def __init__(self, name, state, pybullet_client):
        super().__init__(name, state)
        self.pybullet_client = pybullet_client  # PyBullet simulation client

    def update_state(self, input_data):
        """
        Apply forces and update rotational state in PyBullet.
        :param input_data: Dictionary containing force and torque parameters.
        """
        force = input_data.get('force', [0, 0, 0])
        torque = input_data.get('torque', [0, 0, 0])

        # Apply force and torque to the cube
        self.pybullet_client.apply_force(force, torque)

        # Step the PyBullet simulation
        self.pybullet_client.step_simulation()

        # Update state based on PyBullet simulation
        position, orientation = self.pybullet_client.get_state()
        self.state['position'] = position
        self.state['orientation'] = orientation

# ===========================
# PositionalCollisionAuthority Class
# ===========================

class PositionalCollisionAuthority(Authority):
    def __init__(self, name, state, pymunk_space, player_body):
        super().__init__(name, state)
        self.pymunk_space = pymunk_space  # Pymunk simulation space
        self.player_body = player_body      # Pymunk body representing the player

    def update_state(self, input_data):
        """
        Update positions and handle collisions in Pymunk.
        :param input_data: Dictionary containing positional data and collision info.
        """
        # Apply input forces to the player body
        force = input_data.get('force', (0, 0))
        torque = input_data.get('torque', 0)
        self.player_body.apply_force_at_local_point(force, (0, 0))
        self.player_body.torque += torque

        # Step the Pymunk simulation
        self.pymunk_space.step(DT)

        # Update state based on Pymunk simulation
        self.state['position'] = self.player_body.position
        self.state['angle'] = self.player_body.angle

# ===========================
# CausalityGraph Class
# ===========================

class CausalityGraph:
    def __init__(self):
        """
        Initialize the causality graph with nodes and edges.
        Nodes represent authorities, and edges represent causal relationships with weights.
        """
        self.nodes = {}  # key: authority name, value: Authority instance
        self.edges = defaultdict(list)  # key: source authority, value: list of (target authority, weight)

    def add_authority(self, authority):
        """
        Add an authority to the graph.
        :param authority: Instance of Authority or its subclasses.
        """
        self.nodes[authority.name] = authority

    def add_causality(self, source, target, weight):
        """
        Add a causality edge between two authorities.
        :param source: Source authority name.
        :param target: Target authority name.
        :param weight: Weight of the edge, typically dt.
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target authority not found in the graph.")
        self.edges[source].append((target, weight))

    def propagate(self):
        """
        Propagate causality through the graph.
        Each authority processes its output and feeds it to connected authorities.
        """
        outputs = {}
        # Collect outputs from all authorities
        for name, authority in self.nodes.items():
            outputs[name] = authority.get_output()

        # Propagate outputs based on edges
        for source, connections in self.edges.items():
            for target, weight in connections:
                # For simplicity, weight is not used in this minimal implementation
                input_data = outputs[source]
                self.nodes[target].update_state(input_data)

# ===========================
# CoordinateTranslator Class
# ===========================

class CoordinateTranslator:
    def __init__(self):
        """
        Initialize the coordinate translator.
        For this minimal implementation, we'll assume all authorities share the same base coordinate system.
        """
        pass

    def translate(self, source_authority, target_authority, vectors):
        """
        Translate vectors from source authority's coordinate system to target authority's system.
        :param source_authority: Name of the source authority.
        :param target_authority: Name of the target authority.
        :param vectors: Dictionary of vectors to translate.
        :return: Translated vectors.
        """
        # Placeholder for actual transformation logic
        # Assuming identity transformation for minimal implementation
        return vectors

# ===========================
# PygameBuffer Class
# ===========================

class PygameBuffer:
    def __init__(self, screen):
        self.screen = screen

    def update(self, rendered_image):
        """
        Update the Pygame buffer with the rendered image.
        :param rendered_image: Image data as a NumPy array.
        """
        # Convert the rendered image to a Pygame surface
        pygame_image = pygame.surfarray.make_surface(rendered_image.swapaxes(0, 1))
        self.screen.blit(pygame_image, (0, 0))

    def display(self):
        """
        Update the Pygame display.
        """
        pygame.display.flip()

# ===========================
# PyBulletClient Class
# ===========================

class PyBulletClient:
    def __init__(self):
        """
        Initialize the PyBullet client in DIRECT mode.
        """
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*PYBULLET_GRAVITY, physicsClientId=self.client_id)
        self.cube_id = self.create_pybullet_cube()

    def create_pybullet_cube(self):
        """
        Create a cube in PyBullet.
        :return: Cube ID.
        """
        cube_size = 0.5  # Half extents
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client_id)
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3, physicsClientId=self.client_id)
        cube_mass = 1
        cube_start_pos = [0, 0, 0]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.createMultiBody(cube_mass, cube_collision, cube_visual, cube_start_pos, cube_start_orientation, physicsClientId=self.client_id)
        return cube_id

    def apply_force(self, force, torque):
        """
        Apply force and torque to the cube.
        :param force: Force vector [fx, fy, fz].
        :param torque: Torque vector [tx, ty, tz].
        """
        p.applyExternalForce(self.cube_id, -1, force, [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client_id)
        p.applyExternalTorque(self.cube_id, -1, torque, p.WORLD_FRAME, physicsClientId=self.client_id)

    def step_simulation(self):
        """
        Step the PyBullet simulation.
        """
        p.stepSimulation(physicsClientId=self.client_id)

    def get_state(self):
        """
        Get the current state of the cube.
        :return: Tuple of position and orientation.
        """
        pos, orn = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)
        return pos, orn

    def disconnect(self):
        """
        Disconnect the PyBullet client.
        """
        p.disconnect(self.client_id)

# ===========================
# PhysicsSimulator Class
# ===========================

class PhysicsSimulator:
    def __init__(self):
        """
        Initialize the physics simulator with all authorities and the causality graph.
        """
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Multi-Authority Physics Simulator")
        self.clock = pygame.time.Clock()
        self.pygame_buffer = PygameBuffer(self.screen)

        # Initialize Pymunk
        self.space = pymunk.Space()
        self.space.gravity = PYMUNK_GRAVITY
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Create static walls in Pymunk
        self.create_walls()

        # Create player cube in Pymunk
        self.player_body = self.create_player((WIDTH // 2, HEIGHT // 2))
        
        # Initialize PyBullet
        self.pybullet_client = PyBulletClient()

        # Initialize Mitsuba
        mi.set_variant("scalar_rgb")
        self.mitsuba_config = self.initialize_mitsuba_config()

        # Initialize Authorities
        visual_state = {"dt": DT}
        self.visual_authority = VisualAuthority("Visual", visual_state, self.mitsuba_config, self.pygame_buffer)

        rotational_state = {"dt": DT}
        self.rotational_authority = RotationalForceAuthority("RotationalForce", rotational_state, self.pybullet_client)

        positional_state = {"dt": DT}
        self.positional_authority = PositionalCollisionAuthority("PositionalCollision", positional_state, self.space, self.player_body)

        # Initialize Causality Graph
        self.causality_graph = CausalityGraph()
        self.causality_graph.add_authority(self.rotational_authority)
        self.causality_graph.add_authority(self.positional_authority)
        self.causality_graph.add_authority(self.visual_authority)

        # Define causality relationships
        self.causality_graph.add_causality("RotationalForce", "PositionalCollision", DT)
        self.causality_graph.add_causality("PositionalCollision", "Visual", DT)
        self.causality_graph.add_causality("Visual", "RotationalForce", DT)

        # Initialize Coordinate Translator
        self.coordinate_translator = CoordinateTranslator()

    def initialize_mitsuba_config(self):
        """
        Initialize the Mitsuba XML-style Python dictionary.
        :return: Mitsuba configuration dictionary.
        """
        return {
            "type": "scene",
            "objects": {
                "cube": {
                    "type": "obj",
                    "filename": "cube.obj",
                    "position": [0, 0, 0]
                },
                "light": {
                    "type": "point",
                    "position": [5, 5, 5],
                    "intensity": [1, 1, 1]
                },
                "camera": {
                    "type": "perspective",
                    "position": [0, -10, 5],
                    "look_at": [0, 0, 0],
                    "fov": 45
                }
            },
            "integrator": {
                "type": "path"
            }
        }

    def create_walls(self, thickness=10):
        """Create static walls in Pymunk."""
        static_body = self.space.static_body
        walls = [
            pymunk.Segment(static_body, (0, 0), (WIDTH, 0), thickness),               # Bottom
            pymunk.Segment(static_body, (0, HEIGHT), (WIDTH, HEIGHT), thickness),     # Top
            pymunk.Segment(static_body, (0, 0), (0, HEIGHT), thickness),              # Left
            pymunk.Segment(static_body, (WIDTH, 0), (WIDTH, HEIGHT), thickness)       # Right
        ]
        for wall in walls:
            wall.elasticity = 0.30
            wall.friction = 1.0
        self.space.add(*walls)

    def create_player(self, pos, size=50, mass=1):
        """Create the player cube in Pymunk."""
        moment = pymunk.moment_for_box(mass, (size, size))
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Poly.create_box(body, (size, size))
        shape.elasticity = 0.5
        shape.friction = 0.5
        self.space.add(body, shape)
        return body

    def handle_input(self):
        """
        Handle user input and return force and torque values.
        :return: Dictionary containing force and torque parameters.
        """
        keys = pygame.key.get_pressed()
        force = [0, 0, 0]  # Force for PyBullet
        torque = [0, 0, 0] # Torque for PyBullet

        # Translational forces for Pymunk
        if keys[pygame.K_w]:
            self.player_body.apply_force_at_local_point((0, -1000))
        if keys[pygame.K_s]:
            self.player_body.apply_force_at_local_point((0, 1000))
        if keys[pygame.K_a]:
            self.player_body.apply_force_at_local_point((-1000, 0))
        if keys[pygame.K_d]:
            self.player_body.apply_force_at_local_point((1000, 0))

        # Rotational torques for PyBullet
        if keys[pygame.K_q]:
            torque[2] -= 10  # Counter-clockwise
        if keys[pygame.K_e]:
            torque[2] += 10  # Clockwise

        return {'force': force, 'torque': torque}

    def synchronize_authorities(self):
        """
        Synchronize the state between PyBullet and Pymunk.
        """
        # Get state from PyBullet
        pyb_pos, pyb_orn = self.rotational_authority.get_output()['position'], self.rotational_authority.get_output()['orientation']
        
        # Convert PyBullet position to Pymunk coordinates
        pymunk_pos = pymunk.Vec2d(pyb_pos[0] * SCALE, HEIGHT - pyb_pos[1] * SCALE)
        self.player_body.position = pymunk_pos
        self.player_body.angle = -self.euler_from_quaternion(pyb_orn)[2]

    def euler_from_quaternion(self, quat):
        """
        Convert a quaternion into Euler angles.
        :param quat: Quaternion tuple (x, y, z, w).
        :return: Tuple of Euler angles (roll, pitch, yaw).
        """
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return (roll, pitch, yaw)

    def run_simulation(self):
        """
        Run the main simulation loop.
        """
        running = True
        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Handle input
            input_data = self.handle_input()

            # Update RotationalForceAuthority
            self.rotational_authority.update_state(input_data)

            # Synchronize PyBullet to Pymunk
            self.synchronize_authorities()

            # Step PositionalCollisionAuthority
            positional_input = {
                'force': (0, 0),
                'torque': 0
            }
            self.positional_authority.update_state(positional_input)

            # Update VisualAuthority with current positions
            visual_input = {
                'cube_pos': self.positional_authority.get_output()['position'],
                'light_pos': self.mitsuba_config['objects']['light']['position'],
                'camera_pos': self.mitsuba_config['objects']['camera']['position']
            }
            self.visual_authority.update_state(visual_input)

            # Render Pymunk objects
            self.space.debug_draw(self.draw_options)

            # Update Pygame display
            self.pygame_buffer.display()

            # Maintain FPS
            self.clock.tick(FPS)

        # Clean up
        self.pybullet_client.disconnect()
        pygame.quit()

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    simulator = PhysicsSimulator()
    simulator.run_simulation()
