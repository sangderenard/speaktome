import pygame
import pymunk
import pymunk.pygame_util
import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Constants
WIDTH, HEIGHT = 400, 400
FPS = 60
SIM_FPS = 30
DT = 1.0 / SIM_FPS
PYMUNK_GRAVITY = (0, -9)  # Pymunk uses pixels/sec²
PYBULLET_GRAVITY = (0, 0, 0)  # No gravity in PyBullet
PLAYER_FORCE = 1000

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pymunk, PyBullet, and Pygame Integration")
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Initialize Pymunk
space = pymunk.Space()
space.gravity = PYMUNK_GRAVITY

# Create static walls in Pymunk
def create_walls(space, width, height, thickness=10):
    static_body = space.static_body
    walls = [
        pymunk.Segment(static_body, (0, 0), (width, 0), thickness),        # Bottom
        pymunk.Segment(static_body, (0, height), (width, height), thickness),  # Top
        pymunk.Segment(static_body, (0, 0), (0, height), thickness),       # Left
        pymunk.Segment(static_body, (width, 0), (width, height), thickness)  # Right
    ]
    for wall in walls:
        wall.elasticity = .30  # Fully elastic
        wall.friction = 1.0    # High friction
    space.add(*walls)

create_walls(space, WIDTH, HEIGHT)
def calibrate_offset(pymunk_body, pybullet_id):
    """Calculate positional offset between Pymunk and PyBullet."""
    pymunk_pos = pymunk_body.position
    pybullet_pos, _ = p.getBasePositionAndOrientation(pybullet_id)
    return pymunk.Vec2d(pybullet_pos[0] - pymunk_pos.x, pybullet_pos[1] - pymunk_pos.y)
def update_pymunk_shape_from_projection(pybullet_id, pymunk_body, space):
    """Project PyBullet vertices into 2D, create a Pymunk shape, and assign it to the Pymunk body."""
    # Get projected vertices
    projected_vertices = get_projected_vertices(pybullet_id)  # List of (x, y) tuples

    # Flatten into a Pymunk-friendly format (relative to Pymunk body position)
    relative_vertices = [(x - WIDTH // 2, HEIGHT // 2 - y) for x, y in projected_vertices]

    # Remove existing shapes from the Pymunk body
    for shape in pymunk_body.shapes:
        space.remove(shape)

    # Create and add the new shape
    new_shape = pymunk.Poly(pymunk_body, relative_vertices)
    new_shape.elasticity = 0.5
    new_shape.friction = 0.5
    space.add(new_shape)

    # Update the moment of inertia
    pymunk_body.moment = pymunk.moment_for_poly(pymunk_body.mass, relative_vertices)

# Create the player cube in Pymunk
def create_player(space, pos, size=50, mass=1):
    moment = pymunk.moment_for_box(mass, (size, size))
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Poly.create_box(body, (size, size))
    shape.elasticity = 0.5
    shape.friction = 0.5
    space.add(body, shape)
    return body

player_body = create_player(space, (WIDTH // 2, HEIGHT // 2))

# Initialize PyBullet in headless mode
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*PYBULLET_GRAVITY)

# Create a cube in PyBullet
cube_size = 0.5  # Half extents
cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1,0,0,1])
cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3)
cube_mass = 1
cube_start_pos = [0, 0, 0]
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_id = p.createMultiBody(cube_mass, cube_collision, cube_visual, cube_start_pos, cube_start_orientation)

# Disable default PyBullet gravity
p.setGravity(0, 0, 0, physicsClientId=physicsClient)

# Calculate calibration offset once at the beginning
calibration_offset = calibrate_offset(player_body, cube_id)

def sync_pymunk_to_pybullet(pymunk_body, pybullet_id, offset):
    pos = pymunk_body.position + offset
    angle = pymunk_body.angle
    pybullet_pos = [pos.x, pos.y, 0]
    pybullet_orientation = p.getQuaternionFromEuler([0, 0, angle])
    p.resetBasePositionAndOrientation(pybullet_id, pybullet_pos, pybullet_orientation)
    vel = pymunk_body.velocity
    pybullet_vel = [vel.x, vel.y, 0]
    p.resetBaseVelocity(pybullet_id, pybullet_vel, [0, 0, 0])

def sync_pybullet_to_pymunk(pybullet_id, pymunk_body, offset):
    pos, orn = p.getBasePositionAndOrientation(pybullet_id)
    euler = p.getEulerFromQuaternion(orn)
    pymunk_body.position = pymunk.Vec2d(pos[0], pos[1]) - offset
    pymunk_body.angle = euler[2]  # Use Z-axis rotation for 2D
    linear_vel, angular_vel = p.getBaseVelocity(pybullet_id)
    pymunk_body.velocity = pymunk.Vec2d(linear_vel[0], linear_vel[1])
    pymunk_body.angular_velocity = angular_vel[2]  # Use Z-axis angular velocity


# Initial synchronization
sync_pymunk_to_pybullet(player_body, cube_id, calibration_offset)

# Functions for Rendering
def get_projected_vertices(cube_id):
    """Get the 2D projection of the PyBullet cube."""
    # Define the vertices of a cube centered at (0, 0, 0)
    half_extents = 0.5
    vertices = np.array([
        [-half_extents, -half_extents, -half_extents],
        [half_extents, -half_extents, -half_extents],
        [half_extents, half_extents, -half_extents],
        [-half_extents, half_extents, -half_extents],
        [-half_extents, -half_extents, half_extents],
        [half_extents, -half_extents, half_extents],
        [half_extents, half_extents, half_extents],
        [-half_extents, half_extents, half_extents],
    ])

    # Get the cube's position and orientation
    pos, orn = p.getBasePositionAndOrientation(cube_id)
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    
    # Transform vertices by rotation and translation
    transformed_vertices = np.dot(vertices, rotation_matrix.T) + np.array(pos)

    # Project 3D vertices to 2D (fixed slicing on the Z-plane)
    projection_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])  # Drops the Z-component
    projected_vertices = np.dot(transformed_vertices, projection_matrix.T)

    # Scale to screen coordinates
    screen_vertices = [
        (int(WIDTH // 2 + v[0] * 100), int(HEIGHT // 2 - v[1] * 100))
        for v in projected_vertices
    ]
    return screen_vertices

def render_cube(screen, vertices):
    """Render the 2D projection of the cube in Pygame."""
    # Define edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]

    # Draw edges
    for edge in edges:
        print(edge)
        pygame.draw.line(screen, (0, 0, 255), vertices[edge[0]], vertices[edge[1]], 2)

# Initialize authority flag
is_pymunk_turn = True

# Main loop variables
running = True
accumulator = 0.0
last_time = time.time()

while running:
    current_time = time.time()
    frame_time = current_time - last_time
    last_time = current_time
    accumulator += frame_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    force = pymunk.Vec2d(0, 0)  # Translational forces for Pymunk
    torque_pymunk = 0           # Rotational torque for Pymunk (Z-axis)
    torque_pybullet_top = 0     # Rotational torque for PyBullet (Z-axis, top spin)
    tilt_pybullet = 0           # Tilt torque for PyBullet (X-axis, bowling pin)

    # Translational forces (Pymunk)
    if keys[pygame.K_w]:
        force += (0, -PLAYER_FORCE)
    if keys[pygame.K_s]:
        force += (0, PLAYER_FORCE)
    if keys[pygame.K_a]:
        force += (-PLAYER_FORCE, 0)
    if keys[pygame.K_d]:
        force += (PLAYER_FORCE, 0)

    # Rotational forces (Pymunk)
    if keys[pygame.K_z]:  # Ferris wheel counterclockwise
        torque_pymunk -= 1000
    if keys[pygame.K_c]:  # Ferris wheel clockwise
        torque_pymunk += 1000

    # Rotational torques (PyBullet)
    if keys[pygame.K_q]:  # Top spin counterclockwise
        torque_pybullet_top -= 1
    if keys[pygame.K_e]:  # Top spin clockwise
        torque_pybullet_top += 1
    if keys[pygame.K_x]:  # Tilt away from screen
        tilt_pybullet -= 1
    if keys[pygame.K_2]:  # Tilt toward screen
        tilt_pybullet += 1

    # Apply translational forces and rotation to Pymunk
    player_body.apply_force_at_local_point(force, (0, 0))
    player_body.torque += torque_pymunk
    print(player_body.position)
    # Apply rotational torques to PyBullet
    # Ensure torque values are floats
    torque_vector = (float(tilt_pybullet), 0.0, float(torque_pybullet_top))
    p.applyExternalTorque(
        cube_id,
        -1,                          # linkIndex: -1 for base
        torque_vector,               # torqueObj
        p.WORLD_FRAME                # flags
    )

    while accumulator >= DT:
        if is_pymunk_turn:
            # Step Pymunk and synchronize to PyBullet
            space.step(DT)
            sync_pymunk_to_pybullet(player_body, cube_id, calibration_offset)
        else:
            # Step PyBullet and synchronize to Pymunk
            p.stepSimulation()
            sync_pybullet_to_pymunk(cube_id, player_body, calibration_offset)
            
            # Update Pymunk shape from PyBullet projection
            update_pymunk_shape_from_projection(cube_id, player_body, space)
        
        # Alternate authority
        is_pymunk_turn = not is_pymunk_turn

        # Decrement accumulator by fixed time step
        accumulator -= DT


    # Rendering
    screen.fill((25, 55, 25))

    # Render the 2D projection of the PyBullet cube
    projected_vertices = get_projected_vertices(cube_id)
    render_cube(screen, projected_vertices)
    


    # Update the display
    pygame.display.flip()

    # Maintain FPS
    clock.tick(FPS)

# Clean up
p.disconnect(physicsClient)
pygame.quit()
