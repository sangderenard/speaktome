from __future__ import annotations

try:
    import pygame
    import pymunk
    import pymunk.pygame_util
    import pybullet as p
    import pybullet_data
    import time
    import math
    import numpy as np
except Exception:
    print(
        "\n"
        "+-----------------------------------------------------------------------+\n"
        "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
        "| project and module you plan to use. Missing packages mean setup was |\n"
        "| skipped or incomplete.                                             |\n"
        "+-----------------------------------------------------------------------+\n"
    )
    raise
# --- END HEADER ---

# ===========================
# Constants and Configuration
# ===========================

WIDTH, HEIGHT = 400, 400
FPS = 60
SIM_FPS = 30
DT = 1.0 / SIM_FPS

# Shared Coordinate System Scaling
SCALE = 100  # Pixels per meter

# Gravity Settings
PYMUNK_GRAVITY = (0, 900)    # Flipped Y for Pymunk (pixels/sec²)
PYBULLET_GRAVITY = (0, 0, 0)  # No gravity in PyBullet

PLAYER_FORCE = 1000

# ===========================
# Initialization
# ===========================

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pymunk, PyBullet, and Pygame Integration")
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Initialize Pymunk
space = pymunk.Space()
space.gravity = PYMUNK_GRAVITY

# Initialize PyBullet in headless mode
physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(*PYBULLET_GRAVITY, physicsClientId=physics_client)

# ===========================
# Physics Setup Functions
# ===========================

def create_walls(space, width, height, thickness=10):
    """Create static walls in Pymunk."""
    static_body = space.static_body
    walls = [
        pymunk.Segment(static_body, (0, 0), (width, 0), thickness),               # Bottom
        pymunk.Segment(static_body, (0, height), (width, height), thickness),     # Top
        pymunk.Segment(static_body, (0, 0), (0, height), thickness),              # Left
        pymunk.Segment(static_body, (width, 0), (width, height), thickness)       # Right
    ]
    for wall in walls:
        wall.elasticity = 0.30
        wall.friction = 1.0
    space.add(*walls)

def create_player(space, pos, size=50, mass=1):
    """Create the player cube in Pymunk."""
    moment = pymunk.moment_for_box(mass, (size, size))
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Poly.create_box(body, (size, size))
    shape.elasticity = 0.5
    shape.friction = 0.5
    space.add(body, shape)
    return body

def create_pybullet_cube():
    """Create a cube in PyBullet."""
    cube_size = 0.5  # Half extents
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1])
    cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3)
    cube_mass = 1
    cube_start_pos = [0, 0, 0]
    cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = p.createMultiBody(cube_mass, cube_collision, cube_visual, cube_start_pos, cube_start_orientation)
    return cube_id

# ===========================
# Synchronization Functions
# ===========================

def calibrate_offset(pymunk_body, pybullet_id):
    """Calculate positional offset between Pymunk and PyBullet."""
    pymunk_pos = pymunk_body.position
    pybullet_pos, _ = p.getBasePositionAndOrientation(pybullet_id)
    return pymunk.Vec2d(pybullet_pos[0] - pymunk_pos.x, pymunk_pos.y - pybullet_pos[1])

def sync_pymunk_to_pybullet(pymunk_body, pybullet_id, offset):
    pos = pymunk_body.position + offset
    angle = -pymunk_body.angle  # Negate angle to reverse the flip
    
    # Adjust position for scale
    pybullet_pos = [pos.x / SCALE, (HEIGHT - pos.y) / SCALE, 0]
    pybullet_orientation = p.getQuaternionFromEuler([0, 0, angle])
    
    p.resetBasePositionAndOrientation(pybullet_id, pybullet_pos, pybullet_orientation, physicsClientId=physics_client)
    
    # Adjust velocity
    vel = pymunk_body.velocity
    pybullet_vel = [vel.x / SCALE, -vel.y / SCALE, 0]
    p.resetBaseVelocity(pybullet_id, pybullet_vel, [0, 0, 0], physicsClientId=physics_client)

def sync_pybullet_to_pymunk(pybullet_id, pymunk_body, offset):
    pos, orn = p.getBasePositionAndOrientation(pybullet_id, physicsClientId=physics_client)
    euler = p.getEulerFromQuaternion(orn)
    
    # Adjust position for flipped Y-axis and scale
    pymunk_body.position = pymunk.Vec2d(pos[0] * SCALE, HEIGHT - (pos[1] * SCALE)) - offset
    
    # Negate Z rotation to account for mirrored Y-axis
    pymunk_body.angle = -euler[2]
    
    # Adjust velocity
    linear_vel, angular_vel = p.getBaseVelocity(pybullet_id, physicsClientId=physics_client)
    pymunk_body.velocity = pymunk.Vec2d(linear_vel[0] * SCALE, -linear_vel[1] * SCALE)
    pymunk_body.angular_velocity = -angular_vel[2]


def update_pymunk_shape_from_projection(pybullet_id, pymunk_body, space):
    """Update Pymunk shape based on PyBullet projection."""
    projected_vertices = get_projected_vertices(pybullet_id)
    relative_vertices = [(x - WIDTH // 2, HEIGHT // 2 - y) for x, y in projected_vertices]
    
    for shape in pymunk_body.shapes:
        space.remove(shape)
    
    new_shape = pymunk.Poly(pymunk_body, relative_vertices)
    new_shape.elasticity = 0.5
    new_shape.friction = 0.5
    space.add(new_shape)
    
    pymunk_body.moment = pymunk.moment_for_poly(pymunk_body.mass, relative_vertices)

# ===========================
# Rendering Functions
# ===========================

def get_projected_vertices(cube_id):
    """Get the 2D projection of the PyBullet cube."""
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
    
    pos, orn = p.getBasePositionAndOrientation(cube_id, physicsClientId=physics_client)
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    transformed_vertices = np.dot(vertices, rotation_matrix.T) + np.array(pos)
    
    # Project to 2D (ignore Z-axis)
    projection_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    projected = np.dot(transformed_vertices, projection_matrix.T)
    
    # Scale to screen coordinates and flip Y
    screen_vertices = [
        (int(WIDTH // 2 + v[0] * SCALE), int(HEIGHT // 2 - v[1] * SCALE))
        for v in projected
    ]
    return screen_vertices

def render_cube(screen, vertices):
    """Render the 2D projection of the cube in Pygame."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    for edge in edges:
        pygame.draw.line(screen, (0, 0, 255), vertices[edge[0]], vertices[edge[1]], 2)

# ===========================
# Input Handling Functions
# ===========================

def handle_input():
    """Handle user input and return force and torque values."""
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
    
    return force, torque_pymunk, torque_pybullet_top, tilt_pybullet

def apply_forces(player_body, torque_pymunk, torque_pybullet_top, tilt_pybullet):
    """Apply forces and torques to Pymunk and PyBullet bodies."""
    # Apply translational forces and rotation to Pymunk
    player_body.apply_force_at_local_point(force, (0, 0))
    player_body.torque += torque_pymunk
    
    # Apply rotational torques to PyBullet
    torque_vector = (float(tilt_pybullet), float(torque_pymunk), float(torque_pybullet_top))
    p.applyExternalTorque(
        cube_id,
        -1,                          # linkIndex: -1 for base
        torque_vector,               # torqueObj
        p.WORLD_FRAME,               # flags
        physicsClientId=physics_client
    )

# ===========================
# Main Program Flow
# ===========================

# Create objects
create_walls(space, WIDTH, HEIGHT)
player_body = create_player(space, (WIDTH // 2, HEIGHT // 2))
cube_id = create_pybullet_cube()

# Calculate calibration offset
calibration_offset = calibrate_offset(player_body, cube_id)

# Initial synchronization
sync_pymunk_to_pybullet(player_body, cube_id, calibration_offset)

# Authority flag
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

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Input Handling
    force, torque_pymunk, torque_pybullet_top, tilt_pybullet = handle_input()
    
    # Apply Forces and Torques with Y-axis Flipping
    force_flipped = pymunk.Vec2d(force.x, force.y)  # Flip Y force
    player_body.apply_force_at_local_point(force_flipped, (0, 0))
    player_body.torque += torque_pymunk
    
    torque_vector = (float(tilt_pybullet), 0.0, float(torque_pybullet_top))
    p.applyExternalTorque(
        cube_id,
        -1,
        torque_vector,
        p.WORLD_FRAME,
        physicsClientId=physics_client
    )

    # Physics Simulation Steps
    while accumulator >= DT:
        if is_pymunk_turn:
            # Step Pymunk and synchronize to PyBullet
            space.step(DT)
            sync_pymunk_to_pybullet(player_body, cube_id, calibration_offset)
        else:
            # Step PyBullet and synchronize to Pymunk
            p.stepSimulation(physicsClientId=physics_client)
            sync_pybullet_to_pymunk(cube_id, player_body, calibration_offset)
            update_pymunk_shape_from_projection(cube_id, player_body, space)
        
        # Alternate authority
        is_pymunk_turn = not is_pymunk_turn

        # Decrement accumulator
        accumulator -= DT

    # Rendering
    screen.fill((25, 55, 25))
    
    # Render shared coordinate system (optional: visualize origin or axes)
    # pygame.draw.line(screen, (255, 255, 255), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 1)
    # pygame.draw.line(screen, (255, 255, 255), (0, HEIGHT//2), (WIDTH, HEIGHT//2), 1)
    
    # Render the 2D projection of the PyBullet cube
    projected_vertices = get_projected_vertices(cube_id)
    render_cube(screen, projected_vertices)
    
    # Optionally, render Pymunk objects
    space.debug_draw(draw_options)
    
    # Update the display
    pygame.display.flip()
    
    # Maintain FPS
    clock.tick(FPS)

# ===========================
# Clean Up
# ===========================

p.disconnect(physics_client)
pygame.quit()
