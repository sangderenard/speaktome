import torch
import pygame
import math
import numpy as np
from matplotlib import pyplot as plt
# Pygame setup
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600  # Expanded for three projections
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Rotating Points - Orthogonal Projections")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

# Font setup
pygame.font.init()
FONT = pygame.font.SysFont("Arial", 20)
count = 10
# Simulation Parameters
RADIUS_1 = torch.rand(count) * 200 + 20  # Radius of orbit for point 1
RADIUS_2 = torch.rand(count) * 200 + 20  # Radius of orbit for point 2
LENGTH_L = torch.rand(count) * 200 + RADIUS_1+RADIUS_2   # Fixed rod length
FREQ_1 = torch.rand(count) * 5 + .002    # Angular frequency of point 1
FREQ_2 = torch.rand(count) * 5 + .002    # Angular frequency of point 2
FPS = 60        # Frames per second

# Projection scale
SCALE = 1.0

distance_log = []

# Functions for 3D motion
def compute_orbit(radius, angle, center_z):
    """
    Compute the 3D position of a point orbiting in a circular path.
    :param radius: Radius of the orbit
    :param angle: Current angular position
    :param center_z: Z-axis offset (height of the plane)
    :return: (x, y, z) position of the orbiting point
    """
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    z = center_z
    return x, y, z

def enforce_fixed_length(p1, p2, length):
    """
    Adjust the Z positions of the two points to ensure the fixed rod length constraint is satisfied.
    :param p1: Position of the first point (x, y, z)
    :param p2: Position of the second point (x, y, z)
    :param length: Fixed length of the rod
    :return: Adjusted Z positions for the points
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    dx = x2 - x1
    dy = y2 - y1
    d_xy = torch.sqrt(dx**2 + dy**2)

    dz = torch.sqrt(length**2 - d_xy**2)
    new_z1 = -dz / 2
    new_z2 = dz / 2
    return new_z1, new_z2

def project_to_2d(x, y, z, center_x, center_y, projection):
    """
    Project a 3D point to a 2D plane.
    :param x: X-coordinate
    :param y: Y-coordinate
    :param z: Z-coordinate
    :param center_x: Center X of the screen
    :param center_y: Center Y of the screen
    :param projection: Projection type ('XZ', 'XY', 'YZ')
    :return: (screen_x, screen_y) position
    """
    if projection == 'XZ':  # Front view: XZ-plane
        screen_x = center_x + x * SCALE
        screen_y = center_y - z * SCALE
    elif projection == 'XY':  # Top view: XY-plane
        screen_x = center_x + x * SCALE
        screen_y = center_y - y * SCALE

    return torch.round(torch.stack([screen_x, screen_y])).to(torch.int32)

def draw_plane(center_x, center_y, radius, angle, color, projection):
    """
    Draw the path of the plane as a circle or a line depending on the projection.
    :param center_x: Center X of the plane projection
    :param center_y: Center Y of the plane projection
    :param radius: Radius of the circular orbit
    :param angle: Current angle of rotation
    :param color: Color of the plane
    :param projection: Type of projection
    """
    #print(f"(center_x - radius, center_y): ({center_x} - {radius}, {center_y}")
    if projection == 'XZ':  # Plane in XZ projection - as line
        pygame.draw.line(SCREEN, color, (center_x - radius.to(torch.int32), center_y.to(torch.int32)), (center_x + radius.to(torch.int32), center_y.to(torch.int32)), 1)
    elif projection == 'XY':  # Plane in XY projection - as circle
        pygame.draw.circle(SCREEN, color, (center_x, center_y), int(radius * SCALE), 1)
def compute_physics(p1, p2, p1_velocities, p2_velocities, delta_t=1/60, stiffness=3.0, damping=1.0):
    """
    Update positions and velocities while preserving angular momentum and enforcing rod constraint.
    """
    # Compute rod direction and distance
    direction = p2 - p1  # Vector from p1 to p2
    distance = torch.norm(direction, dim=0, keepdim=True) + 1e-8  # Avoid zero division
    unit_direction = direction / distance  # Normalize rod direction

    # Displacement to enforce fixed length
    target_length = LENGTH_L.unsqueeze(0)  # Shape: (1, count)
    displacement = (distance - target_length) * unit_direction / 2.0

    # Apply displacement to correct positions
    p1 += displacement
    p2 -= displacement

    # Project velocities onto the rod direction and remove parallel components
    p1_proj = torch.sum(p1_velocities * unit_direction, dim=0, keepdim=True) * unit_direction
    p2_proj = torch.sum(p2_velocities * unit_direction, dim=0, keepdim=True) * unit_direction

    # Update velocities: preserve perpendicular components only
    p1_velocities -= p1_proj
    p2_velocities -= p2_proj

    # Add restoring forces (Hooke's Law)
    force = -stiffness * displacement
    p1_velocities += force * delta_t
    p2_velocities -= force * delta_t

    # Apply damping
    p1_velocities *= damping
    p2_velocities *= damping

    # Update positions based on velocities
    p1 += p1_velocities * delta_t
    p2 += p2_velocities * delta_t

    return p1, p2, p1_velocities, p2_velocities



def main():
    clock = pygame.time.Clock()
    running = True
    refresh = False
    physics_mode = False
    previous_p1 = torch.tensor([0]*count)
    previous_p2 = torch.tensor([0]*count)
    angle_1 = 0  # Initial angle for point 1
    angle_2 = 0  # Initial angle for point 2

    # Z positions of the planes (initialized symmetrically)
    Z_1 = -50
    Z_2 = 50
    global RADIUS_1, RADIUS_2, LENGTH_L, FREQ_1, FREQ_2
    while running:
        if refresh:
            RADIUS_1 = torch.rand(count) * 200 + 20  # Radius of orbit for point 1
            RADIUS_2 = torch.rand(count) * 200 + 20  # Radius of orbit for point 2
            LENGTH_L = torch.rand(count) * 200 + RADIUS_1+RADIUS_2   # Fixed rod length
            FREQ_1 = torch.rand(count) * .5 + .002    # Angular frequency of point 1
            FREQ_2 = torch.rand(count) * .5 + .002    # Angular frequency of point 2
        SCREEN.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    physics_mode = not physics_mode

        # Update angles
        angle_1 += FREQ_1 * 2 * math.pi / FPS
        angle_2 -= FREQ_2 * 2 * math.pi / FPS
        if not physics_mode:
            # Compute 3D positions of the points
            p1 = compute_orbit(RADIUS_1, angle_1, Z_1)
            p2 = compute_orbit(RADIUS_2, angle_2, Z_2)
            x1s, y1s, z1s = p1
            x2s, y2s, z2s = p2
            # Enforce the fixed-length constraint
            Z_1, Z_2 = enforce_fixed_length(p1, p2, LENGTH_L)
            plane_distance = abs(Z_2 - Z_1)
            distance_log.append(plane_distance)

            p1 = torch.stack([x1s, y1s, Z_1])
            p2 = torch.stack([x2s, y2s, Z_2])
            p1_velocities = p1 - previous_p1
            p2_velocities = p2 - previous_p2
            previous_p1 = p1.clone()
            previous_p2 = p2.clone()
            # Compute real-time distance between planes
            plane_distance = abs(Z_2 - Z_1)
        else:
            p1, p2, p1_velocities, p2_velocities = compute_physics(previous_p1, previous_p2, p1_velocities, p2_velocities)
            #place the mean of the two points at 0, 0 while retaining velocity

        # Project points to all three views
        p1_xz = project_to_2d(*p1, WIDTH // 4, CENTER_Y, 'XZ')
        p2_xz = project_to_2d(*p2, WIDTH // 4, CENTER_Y, 'XZ')
        p1_xy = project_to_2d(*p1, WIDTH - WIDTH // 4, CENTER_Y, 'XY')
        p2_xy = project_to_2d(*p2, WIDTH - WIDTH // 4, CENTER_Y, 'XY')


        # Draw points and rods
        for i in range(count):
            # Draw the planes as circles or lines depending on projection
            draw_plane(WIDTH // 4, CENTER_Y + Z_2[i], RADIUS_1[i], angle_1[i], LIGHT_GRAY, 'XZ')
            draw_plane(WIDTH - WIDTH // 4, CENTER_Y, RADIUS_1[i], angle_1[i], LIGHT_GRAY, 'XY')
            draw_plane(WIDTH // 4, CENTER_Y + Z_1[i], RADIUS_2[i], angle_2[i], LIGHT_GRAY, 'XZ')
            draw_plane(WIDTH - WIDTH // 4, CENTER_Y, RADIUS_2[i], angle_2[i], LIGHT_GRAY, 'XY')

            pygame.draw.line(SCREEN, GREEN, tuple(p1_xz[...,i].to(torch.int32).tolist()), tuple(p2_xz[...,i].to(torch.int32).tolist()), 2)
            pygame.draw.circle(SCREEN, RED, tuple(p1_xz[...,i].to(torch.int32).tolist()), 5 + .01 * (p1[0,i]).item())
            pygame.draw.circle(SCREEN, BLUE, tuple(p2_xz[...,i].to(torch.int32).tolist()), 5 + .01 * (p2[0,i]).item())
            pygame.draw.line(SCREEN, GREEN, tuple(p1_xy[...,i].to(torch.int32).tolist()), tuple(p2_xy[...,i].to(torch.int32).tolist()), 2)
            pygame.draw.circle(SCREEN, RED, tuple(p1_xy[...,i].to(torch.int32).tolist()), 5 + .01 * (Z_1[i]).item())
            pygame.draw.circle(SCREEN, BLUE, tuple(p2_xy[...,i].to(torch.int32).tolist()), 5 + .01 * (Z_2[i]).item())

        # Render real-time distance between planes
        distance_text = FONT.render(f"Plane Distance: {plane_distance}", True, YELLOW)
        SCREEN.blit(distance_text, (10, 10))

        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    # Plot distance log
    plt.figure(figsize=(10, 6))
    plt.plot(distance_log)
    plt.title("Plane Distance Over Time")
    plt.xlabel("Frame Count")
    plt.ylabel("Distance Between Planes")
    plt.grid(True)
    plt.show()

    pygame.quit()
if __name__ == "__main__":
    main()
