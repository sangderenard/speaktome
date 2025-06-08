import torch
import logging
import numpy as np
import os
import pickle
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import torch
import numpy as np
import logging
import os
import pickle
import time
from triangulator import Triangulator
from compositegeometry import CompositeGeometry
from adaptivegraphnetwork import AdaptiveGraphNetwork
from youngman import YoungManAlgorithm
import threading
import sys
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")


def scalar_sphere(x, y, z, r=1.0):
    """Example scalar function for a sphere."""
    return torch.sin(x)**2 + torch.sin(y)**2 + torch.sin(z)**2 - r**2


def initialize_pygame(width=800, height=600):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Live Isosurface Visualization')


def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    
    # Set up projection matrix
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (800/600), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)


def dynamic_scalar_sphere(x, y, z, t, r=1.0):
    """Dynamic scalar function for a sphere with oscillating radius."""
    return torch.sin(x + t)**2 + torch.sin(y + t)**2 + torch.sin(z + t)**2 - r**2


def setup_lighting():
    """Set up basic lighting in the OpenGL scene."""
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)  # Enable Light 0

    # Set light position and properties
    light_position = [5.0, 5.0, 5.0, 1.0]  # x, y, z, w
    light_diffuse = [1.0, 1.0, 1.0, 1.0]  # RGB diffuse
    light_specular = [0.5, 0.5, 0.5, 1.0]  # RGB specular
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

    # Set material properties
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])  # High specular reflection
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)  # Shininess exponent


def handle_events(events, geometries, scalar_functions, processor, current_geometry, current_scalar_function, current_mode, num_modes, rotating, network_active):
    """
    Handle all events, including mode switching, geometry switching, scalar function switching, and network activation.
    """
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                # Trigger network activation
                return "activate_network", current_geometry, current_scalar_function, current_mode, rotating  # Ensure 5 values are returned
            if not network_active:
                if pygame.K_1 <= event.key <= pygame.K_9:
                    # Switch scalar function
                    idx = event.key - pygame.K_1
                    if idx < len(scalar_functions):
                        logging.info(f"Switching to Scalar Function {idx + 1}")
                        current_scalar_function = idx
                elif pygame.K_a <= event.key <= pygame.K_z:
                    # Switch geometry
                    idx = event.key - pygame.K_a
                    if idx < len(geometries):
                        geometry = geometries[idx]
                        logging.info(f"Switching to Geometry: {geometry}")
                        processor.switch_geometry(geometry)
                        current_geometry = geometry
                    if event.key == pygame.K_r:
                        rotating = not rotating
                elif event.key == pygame.K_UP:
                    # Increment rendering mode
                    current_mode = (current_mode + 1) % num_modes
                    logging.info(f"Switched to rendering mode {current_mode}")
                elif event.key == pygame.K_DOWN:
                    # Decrement rendering mode
                    current_mode = (current_mode - 1) % num_modes
                    logging.info(f"Switched to rendering mode {current_mode}")
                elif event.key == pygame.K_ESCAPE:
                    # Exit the program
                    pygame.quit()
                    sys.exit()

    # Default return to ensure 5 values are always returned
    return None, current_geometry, current_scalar_function, current_mode, rotating


def render_opengl_data(evaluation_result, rotation_angle, mode, triangulator=None, rotating=False, intersection_sample=None, font=None, network_stats=None, fps=0):
    """
    Render isosurface using the specified rendering mode and overlay information.
    """
    vertices = evaluation_result["vertices"].detach().cpu().numpy()
    indices = evaluation_result["indices"].detach().cpu().numpy()
    if not rotating:
        rotation_angle = 0

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5.0)
    glRotatef(rotation_angle, 1, 1, 0)

    if mode == 0:  # Point rendering
        glBegin(GL_POINTS)
        for vertex in vertices:
            glVertex3fv(vertex)
        glEnd()
    elif mode == 1:  # Triangle rendering (raw output)
        glBegin(GL_TRIANGLES)
        for tri in indices:
            for vertex_idx in tri:
                vertex = vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
    elif mode == 2:  # Delaunay triangulation
        triangulated_data = triangulator.apply(vertices, decimation_factor=1)
        triangulated_indices = triangulated_data["indices"]

        glBegin(GL_TRIANGLES)
        for tri in triangulated_indices:
            for vertex_idx in tri:
                vertex = vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
    elif mode == 3:  # Decimated Delaunay triangulation
        triangulated_data = triangulator.apply(vertices, decimation_factor=2)
        triangulated_indices = triangulated_data["indices"]

        glBegin(GL_TRIANGLES)
        for tri in triangulated_indices:
            for vertex_idx in tri:
                vertex = vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
    elif mode == 4:  # High-quality rendering (not yet implemented)
        print("High-quality rendering is a placeholder.")
        # Placeholder: Add high-quality rendering logic here.
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Render Overlays
    if intersection_sample is not None and font is not None:
        # Switch to orthographic projection for 2D overlays
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 800, 0, 600)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Disable lighting and depth testing for overlay
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # Draw the intersection sample in the lower right corner
        if 'vertices' in intersection_sample and 'centroid' in intersection_sample:
            sample_vertices = intersection_sample['vertices'].detach().cpu().numpy()
            sample_centroid = intersection_sample['centroid'].detach().cpu().numpy()

            # Normalize and position the sample
            scale = 50  # Scale factor for visibility
            offset_x = 700  # Position in the lower right
            offset_y = 50

            glBegin(GL_LINES)
            glColor3f(1.0, 1.0, 0.0)  # Yellow lines
            for tri in intersection_sample['indices']:
                for vertex_idx in tri:
                    vertex = sample_vertices[vertex_idx]
                    glVertex2f(offset_x + vertex[0]*scale, offset_y + vertex[1]*scale)
            glEnd()

            # Draw centroid
            glColor3f(1.0, 0.0, 0.0)  # Red point
            glPointSize(5)
            glBegin(GL_POINTS)
            glVertex2f(offset_x + sample_centroid[0]*scale, offset_y + sample_centroid[1]*scale)
            glEnd()

        # Render network statistics and FPS in the bottom left corner
        if network_stats is not None and font is not None:
            text_surface = font.render(f"Loss: {network_stats['loss']:.4f} | FPS: {fps}", True, (255, 255, 255))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(10, 10)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Restore OpenGL states
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    pygame.display.flip()


def train_network(network, optimizer, criterion, graph_data, target_offsets, target_edges, device, precision, num_epochs=100):
    """
    Train the AdaptiveGraphNetwork to match target_offsets and target_edges.
    """
    network.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predicted_offsets, predicted_connectivity = network(graph_data)

        # Loss for vertex positions
        #loss_vertices = criterion(predicted_offsets, target_offsets)

        # Loss for connectivity
        #loss_connectivity = criterion(predicted_connectivity, target_edges)

        # Total loss
        #loss = loss_vertices + loss_connectivity

        #loss.backward()
        #optimizer.step()

        #if epoch % 10 == 0:
        #    logging.info(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

    network.eval()
    return #loss.item()


def main():
    # Initialize Pygame and OpenGL
    initialize_pygame()
    setup_opengl()
    setup_lighting()

    # Initialize font for overlays
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 18)

    # Available geometries and scalar functions
    geometries = ["tetrahedron", "cube", "icosahedron", "octahedron", "square"]
    scalar_functions = [
        lambda x, y, z, t: x**2 + y**2 + z**2 - (1.0 + 0.5 * torch.sin(t)),  # Breathing sphere
        lambda x, y, z, t: torch.sin(x + y + z + t),                        # Sine wave propagation
        lambda x, y, z, t: torch.cos(x * y * z + t),                        # Oscillating cosine wave
        lambda x, y, z, t: x**2 - y**2 + z**2 - (0.5 + 0.3 * torch.cos(t)), # Ellipsoid oscillation
        lambda x, y, z, t: torch.sqrt(x**2 + y**2 + z**2) - (1.0 + 0.2 * torch.sin(2 * t)),  # Breathing shell
        lambda x, y, z, t: torch.sin(x + t) * torch.cos(y + t) * torch.sin(z + t)  # Ripple sphere
    ]

    # Initialize YoungManAlgorithm and Triangulator
    density = 10
    domain_bounds = [
        (-2.5, 2.5),  # x range
        (-2.5, 2.5),  # y range
        (-2.5, 2.5),  # z range
    ]
    isovalue = 0.0
    processor = YoungManAlgorithm(geometry=geometries[0], density=density, jitter_enabled=False, micro_jitter=False)
    triangulator = Triangulator()
    rotating = False

    # Default geometry, scalar function, and rendering mode
    current_geometry = geometries[0]
    current_scalar_function = 0
    current_mode = 0

    rotation_angle = 0
    clock = pygame.time.Clock()
    current_time = torch.tensor([0.0], device="cuda")

    # Network-related variables
    network = None
    optimizer = None
    criterion = None
    network_active = False
    network_stats = None
    intersection_sample = None

    # FPS calculation
    fps = 0
    fps_timer = time.time()
    frame_count = 0

    while True:
        # Handle events for mode switching and network activation
        events = pygame.event.get()
        result, current_geometry, current_scalar_function, current_mode, rotating = handle_events(
            events, geometries, scalar_functions, processor, current_geometry, current_scalar_function, current_mode, 5, rotating, network_active
        )

        if result == "activate_network" and not network_active:
            logging.info("Activating AdaptiveGraphNetwork...")
            network = AdaptiveGraphNetwork(input_dim=7, hidden_dim=128, device="cuda", precision=torch.float64)
            network = network.to("cuda", dtype=torch.float64)
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            network_active = True

            # Create graph data from current evaluation
            evaluation_result = processor.evaluate(
                lambda x, y, z: scalar_functions[current_scalar_function](x, y, z, t=current_time),
                domain_bounds,
                isovalue=isovalue,
                gradient_normals=False,
                compute_vertex_normals=False,
                centroid_refinement=False,
                deduplicate=True,
                oversampling_grid=(1,1,1),
                oversampling_spot=1,
                jitter_strength=(0, 0, 0)
            )

            intersection_tensor = evaluation_result.get("intersection_tensor", None)
            if intersection_tensor is None:
                logging.error("No intersection tensor found. Cannot train network.")
                network_active = False
            else:
                graph_data = AdaptiveGraphNetwork.create_graph(
                    intersection_tensor=intersection_tensor,
                    offsets=processor.vertex_offsets,
                    edge_index=processor.edge_pairs.T,
                    isosurface_func=lambda x, y, z: scalar_functions[current_scalar_function](x, y, z, t=current_time),
                    device="cuda"
                )

                # Clone current geometry as target
                target_offsets = processor.vertex_offsets.clone().detach()
                target_edges = torch.ones_like(processor.edge_pairs, dtype=torch.float64, device="cuda")  # Assuming target connectivity is all ones

                # Start training in a separate thread
                def train_network_thread():
                    global network_stats
                    loss = train_network(network, optimizer, criterion, graph_data, target_offsets, target_edges, "cuda", torch.float64, num_epochs=100)
                    network_stats = {"loss": loss}
                    logging.info("Network training completed.")
                    global network_active
                    network_active = False

                training_thread = threading.Thread(target=train_network_thread)
                training_thread.start()

        # Update current time for dynamic scalar fields
        current_time += 0.01  # Time progression

        # Get the active scalar function
        scalar_function = scalar_functions[current_scalar_function]

        # Evaluate the scalar field for the current geometry
        evaluation_result = processor.evaluate(
            lambda x, y, z: scalar_function(x, y, z, t=current_time),
            domain_bounds,
            isovalue=isovalue,
            gradient_normals=False,
            compute_vertex_normals=False,
            centroid_refinement=False,
            deduplicate=True,
            oversampling_grid=(1,1,1),
            oversampling_spot=1,
            jitter_strength=(0, 0, 0)  # (torch.pi, torch.pi/2, 0)
        )

        # If network is active, perform training steps
        if network_active and network is not None:
            # Create graph data from current evaluation
            intersection_tensor = evaluation_result.get("intersection_tensor", None)
            if intersection_tensor is not None:
                graph_data = AdaptiveGraphNetwork.create_graph(
                    intersection_tensor=intersection_tensor,
                    offsets=processor.vertex_offsets,
                    edge_index=processor.edge_pairs.T,
                    isosurface_func=lambda x, y, z: scalar_functions[current_scalar_function](x, y, z, t=current_time),
                    device="cuda"
                )

                # Perform a single training step
                network.train()
                optimizer.zero_grad()
                predicted_offsets, predicted_connectivity = network(graph_data)

                # Update geometry with network's predictions
                updated_offsets, updated_edges = network.refresh_geometry(graph_data, processor.vertex_offsets, processor.edge_pairs.T)
                processor.vertex_offsets = updated_offsets
                processor.edge_pairs = updated_edges

                # Evaluate the scalar field for the updated geometry
                evaluation_result = processor.evaluate(
                    lambda x, y, z: scalar_function(x, y, z, t=current_time),
                    domain_bounds,
                    isovalue=isovalue,
                    gradient_normals=False,
                    compute_vertex_normals=False,
                    centroid_refinement=False,
                    deduplicate=True,
                    oversampling_grid=(1,1,1),
                    oversampling_spot=1,
                    jitter_strength=(0, 0, 0)
                )

                # Get the intersection points
                intersection_tensor = evaluation_result.get("intersection_tensor", None)

                if intersection_tensor is not None:
                    # Flatten and filter valid intersection points
                    intersection_points = intersection_tensor.view(-1, 3)
                    valid_mask = ~torch.isnan(intersection_points[:, 0])
                    valid_points = intersection_points[valid_mask]

                    # Compute scalar function values at intersection points
                    scalar_values = scalar_function(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], t=current_time)

                    # Compute loss as MSE between scalar_values and isovalue
                    loss = torch.mean((scalar_values - isovalue) ** 2)
                else:
                    # If no intersection points, set loss to zero
                    loss = torch.tensor(0.0, device="cuda", requires_grad=True)

                # Backpropagate and update the network
                loss.backward()
                optimizer.step()

                network_stats = {"loss": loss.item()}
            else:
                # Handle the case when no intersection tensor is available
                logging.warning("No intersection tensor found. Skipping training step.")


        # Render the updated data using the current mode
        # Collect intersection sample for overlay (e.g., first tile's data)
        if 'intersection_tensor' in evaluation_result:
            first_tile_tensor = evaluation_result['intersection_tensor'][0]
            # Extract vertices and centroid for the sample
            sample_vertices = first_tile_tensor[~torch.isnan(first_tile_tensor[:, 0])]
            if len(sample_vertices) >= 3:
                sample_centroid = sample_vertices.mean(dim=0)
                intersection_sample = {
                    "vertices": sample_vertices[:3],  # Take first three for a triangle
                    "indices": torch.tensor([[0, 1, 2]], device="cuda"),
                    "centroid": sample_centroid
                }
            else:
                intersection_sample = None
        else:
            intersection_sample = None

        render_opengl_data(
            evaluation_result=evaluation_result,
            rotation_angle=rotation_angle,
            mode=current_mode,
            triangulator=triangulator,
            rotating=rotating,
            intersection_sample=intersection_sample,
            font=font,
            network_stats=network_stats,
            fps=int(fps)
        )

        # Update rotation angle
        rotation_angle += 1
        if rotation_angle >= 360:
            rotation_angle -= 360

        # FPS calculation
        frame_count += 1
        current_time_fps = time.time()
        if current_time_fps - fps_timer >= 1.0:
            fps = frame_count / (current_time_fps - fps_timer)
            frame_count = 0
            fps_timer = current_time_fps

        # Cap the frame rate
        clock.tick(60)


if __name__ == "__main__":
    main()
