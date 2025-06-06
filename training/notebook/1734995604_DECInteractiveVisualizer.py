import pygame
import torch
import math
import numpy as np
import json
from compositegeometry import CompositeGeometry, MetaNetwork, SPACE
from hud import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
# Update SPACE for visualization preferences
SPACE["simplicial_complex"]["vertices"]["configurable_options"]["position_lock"] = True
SPACE["simplicial_complex"]["vertices"]["spontaneous_generation"] = False

class DECInteractiveVisualizer:
    def __init__(self, metanetwork, screen_size=(800, 800), scale_factor=0.5):
        pygame.init()
        self.screen_size = screen_size
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        self.screen = pygame.display.set_mode(screen_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Interactive Geometry Editor with OpenGL")
        self.fixed_origin = True  # Default to dynamic centering
        # Rendering mode flags
        self.render_points = True
        self.render_lines = True
        self.paused = True

        self.clock = pygame.time.Clock()

        # MetaNetwork data
        self.metanetwork = metanetwork
        self.vertices = metanetwork.data.pos.cpu()
        self.active_layer = "default"
        self.edges = self.filter_edges_by_layer()

        self.scale_factor = scale_factor
        self.rotation_speed = 0.05
        self.angle_x, self.angle_y = 0, 0

        self.selected_vertices = []
        self.normalize_geometry()
        
        self.font = pygame.font.SysFont(None, 24)
        self.camera_offset = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.pan_speed = 0.01  # Adjust pan speed
        self.zoom_factor = 1.0

        # OpenGL setup
        #glEnable(GL_DEPTH_TEST)
        self.shader = self.create_shader()

        self.vao, self.vbo = self.create_buffers(self.vertices) if self.vertices is not None else (None, None)
    def compute_rotation_matrix(self):
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)

        # Rotation around X-axis
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, cos_x, -sin_x, 0],
            [0, sin_x, cos_x, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Rotation around Y-axis
        rotation_y = np.array([
            [cos_y, 0, sin_y, 0],
            [0, 1, 0, 0],
            [-sin_y, 0, cos_y, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Combine rotations
        return np.matmul(rotation_y, rotation_x)

    def normalize_geometry(self):
        """Normalize vertex positions for consistent scaling."""
        self.xy_min = self.vertices.min(dim=0).values
        self.xy_max = self.vertices.max(dim=0).values
        self.xy_range = self.xy_max - self.xy_min
    def create_shader(self):
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;

        uniform mat4 rotation;
        uniform mat4 projection;  // Add a projection matrix

        void main() {
            gl_Position = vec4(position, 1.0);
        }

        """

        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.2, 0.8, 0.08); // White color for now
        }
        """

        return compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER),
                            compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    def render_points_opengl(self):
        """Render vertices as points."""
        if self.paused:
            glPointSize(10.0)  # Set point size
        else:
            import random
            glPointSize(5.0*random.random())
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.vertices.shape[0])
        glBindVertexArray(0)
    def render_lines_opengl(self):
        """Render edges as lines."""
        if not self.edges or len(self.edges) == 0:
            return  # No edges to render

        edge_indices = np.array(self.edges, dtype=np.uint32).flatten()

        if not hasattr(self, 'ebo') or self.ebo is None:
            # Create EBO only once
            self.ebo = glGenBuffers(1)

        # Bind VAO first!
        glBindVertexArray(self.vao)

        # Bind and populate EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edge_indices.nbytes, edge_indices, GL_STATIC_DRAW)

        try:
            glDrawElements(GL_LINES, len(edge_indices), GL_UNSIGNED_INT, None)
        except OpenGL.error.GLError as e:
            print(f"OpenGL Error during line rendering: {e}")
        finally:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glBindVertexArray(0)  # Unbind VAO


    def render_opengl(self):
        """Render vertices and edges using OpenGL."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)#_MINUS_SRC_ALPHA)

        glUseProgram(self.shader)

        # Compute and pass the rotation matrix
        rotation_matrix = self.compute_rotation_matrix()
        rotation_location = glGetUniformLocation(self.shader, "rotation")
        glUniformMatrix4fv(rotation_location, 1, GL_FALSE, rotation_matrix)

        # Create and pass the projection matrix
        aspect_ratio = self.screen_size[0] / self.screen_size[1]
        ortho_scale = 1.0
        projection_matrix = np.array([
            [1.0 / (aspect_ratio * ortho_scale), 0, 0, 0],
            [0, 1.0 / ortho_scale, 0, 0],
            [0, 0, 2.0 / (10.0 - 0.1), (10.0 + 0.1) / (10.0 - 0.1)],
            [0, 0, 0, 1.0],
        ], dtype=np.float32)
        projection_location = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection_matrix)

        # Render points and lines based on flags
        if self.render_points:
            self.render_points_opengl()
        if self.render_lines:
            self.render_lines_opengl()

        pygame.display.flip()

    def update_buffers(self):
        """Update VBO with current vertex positions (after camera transforms)."""
        rotated_vertices = self.rotate_vertices()
        screen_vertices = self.map_to_ndc(rotated_vertices)

        # Prepare 2D positions for OpenGL rendering (adding z=0 for compatibility)
        vertex_data = torch.cat((screen_vertices, torch.ones(screen_vertices.size(0), 1)), dim=1)
        vertices = vertex_data.contiguous().numpy().astype('float32')
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


    def create_buffers(self, vertices):
        vertices = vertices.detach().cpu().numpy()
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao, vbo

    def rotate_vertices(self):
        """Rotate vertices in spherical angles."""
        rotated = []
        #self.vertices = self.metanetwork.data.pos.cpu()
        
        with self.metanetwork.data_queue_lock:
            if not self.metanetwork.data_queue.empty():
                self.previous_data = self.metanetwork.data_queue.get()
            
                self.vertices = self.previous_data#x[...,self.metanetwork.field_index("position")].cpu()
                
        for vertex in self.vertices:
            x, y, z = vertex.tolist()
            temp_y = y * math.cos(self.angle_x) - z * math.sin(self.angle_x)
            temp_z = y * math.sin(self.angle_x) + z * math.cos(self.angle_x)
            y, z = temp_y, temp_z
            temp_x = x * math.cos(self.angle_y) - z * math.sin(self.angle_y)
            temp_z = x * math.sin(self.angle_y) + z * math.cos(self.angle_y)
            x, z = temp_x, temp_z
            rotated.append([x, y, z])
        return torch.tensor(rotated)

    def map_to_screen(self, xy):
        self.normalize_geometry()
        scale = self.scale_factor * self.zoom_factor
        screen_x = ((xy[:, 0] - self.xy_min[0]) / self.xy_range[0] + self.camera_offset[0]) * self.screen_size[0] * scale
        screen_y = ((xy[:, 1] - self.xy_min[1]) / self.xy_range[1] + self.camera_offset[1]) * self.screen_size[1] * scale
        
        return torch.stack((screen_x, screen_y), dim=1)
    def map_to_ndc(self, xy):
        """Map vertex positions to Normalized Device Coordinates (NDC)."""
        self.normalize_geometry()  # Ensure geometry bounds are updated
        scale = self.scale_factor * self.zoom_factor

        if self.fixed_origin:
            # Keep (0, 0) fixed at the center
            max_range = 1#max(self.xy_range.max(), 1e-6)  # Avoid division by zero
            ndc_x = xy[:, 0] / max_range * 2 * scale
            ndc_y = xy[:, 1] / max_range * 2 * scale
        else:
            # Dynamically center based on vertex bounds
            ndc_x = ((xy[:, 0] - self.xy_min[0]) / self.xy_range[0] - 0.5 + self.camera_offset[0]) * 2 * scale
            ndc_y = ((xy[:, 1] - self.xy_min[1]) / self.xy_range[1] - 0.5 + self.camera_offset[1]) * 2 * scale

        return torch.stack((ndc_x, ndc_y), dim=1)



    def get_clicked_vertex(self, screen_vertices, mouse_pos):
        """Return the closest vertex to the mouse click."""
        for idx, (x, y) in enumerate(screen_vertices):
            if (x - mouse_pos[0]) ** 2 + (y - mouse_pos[1]) ** 2 < 15 ** 2:  # Radius tolerance
                return idx
        return None

    def save_geometry(self):
        """Save the current geometry (vertices and edges) to a JSON file."""
        data = {"vertices": self.vertices.tolist(), "edges": self.edges}
        with open("saved_geometry.json", "w") as f:
            json.dump(data, f)
        print("Geometry saved to 'saved_geometry.json'.")
    def filter_edges_by_layer(self):
        """Filter edges to display based on the active layer."""
        active_layer_index = self.metanetwork.edge_types[self.active_layer][0]
        mask = self.metanetwork.edge_type_map == active_layer_index
        return self.metanetwork.data.edge_index[:, mask].t().tolist()
    def draw(self):
        """Main visualization loop."""
        running = True
        dragging = False
        move_vertex_idx = None
        previous_mouse_pos = None
        self.move_history = []
        update_colors = 0
        colors = None

        while running:
            start_time = time.time()
            #self.screen.fill((0, 0, 0))
            mouse_clicked = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Mouse click detection
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                    mouse_clicked = True
                    previous_mouse_pos = pygame.mouse.get_pos()
                    dragging = True
                    vertex_idx = self.get_clicked_vertex(self.map_to_screen(self.rotate_vertices()[:, :2]), previous_mouse_pos)
                    print(vertex_idx)
                    if vertex_idx is not None:
                        move_vertex_idx = vertex_idx
                    else:
                        move_vertex_idx = None  # No vertex selected; global move

                    self.move_history.append(self.vertices.clone())  # Save for undo
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up
                        self.zoom_factor *= 1.1
                    if event.button == 5:  # Scroll down
                        self.zoom_factor /= 1.1

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Mouse release
                    dragging = False
                    move_vertex_idx = None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save geometry
                        self.save_geometry()
                    if event.key == pygame.K_u and self.move_history:  # Undo
                        self.vertices = self.move_history.pop()
                        self.metanetwork.vertices = self.vertices
                        self.metanetwork.data.pos = self.vertices

            # Handle key-based rotation
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]: self.paused = not self.paused
            if keys[pygame.K_UP]: self.angle_x -= self.rotation_speed
            if keys[pygame.K_DOWN]: self.angle_x += self.rotation_speed
            if keys[pygame.K_LEFT]: self.angle_y -= self.rotation_speed
            if keys[pygame.K_RIGHT]: self.angle_y += self.rotation_speed
            if keys[pygame.K_w]: self.camera_offset[1] -= self.pan_speed * self.zoom_factor
            if keys[pygame.K_s]: self.camera_offset[1] += self.pan_speed * self.zoom_factor
            if keys[pygame.K_a]: self.camera_offset[0] -= self.pan_speed * self.zoom_factor
            if keys[pygame.K_d]: self.camera_offset[0] += self.pan_speed * self.zoom_factor
            # Rotate and map vertices
            rotated_vertices = self.rotate_vertices()
            screen_vertices = self.map_to_screen(rotated_vertices[:, :2])

            # Handle dragging
            if dragging and previous_mouse_pos:
                current_mouse_pos = pygame.mouse.get_pos()
                dx = (current_mouse_pos[0] - previous_mouse_pos[0]) / self.screen_size[0] * self.xy_range[0]
                dy = (current_mouse_pos[1] - previous_mouse_pos[1]) / self.screen_size[1] * self.xy_range[1]
                previous_mouse_pos = current_mouse_pos

                # Convert 2D motion back to 3D motion using the inverse rotation
                movement_3d = self.project_back_to_3d(dx, dy)

                if move_vertex_idx is not None:
                    # Move only the selected vertex
                    self.vertices[move_vertex_idx] += movement_3d
                else:
                    # Move all vertices
                    self.vertices += movement_3d

                # Update MetaNetwork state
                #self.metanetwork.vertices = self.vertices
                #self.metanetwork.data.pos = self.vertices

            # Handle mouse clicks for edge editing
            if mouse_clicked:
                mouse_pos = pygame.mouse.get_pos()
                vertex_idx = self.get_clicked_vertex(screen_vertices, mouse_pos)
                if vertex_idx is not None:
                    if len(self.selected_vertices) == 1 and self.selected_vertices[-1] == vertex_idx:
                        self.selected_vertices = []
                    elif len(self.selected_vertices) == 0 or self.selected_vertices[-1] != vertex_idx:
                        self.selected_vertices.append(vertex_idx)
                if len(self.selected_vertices) == 2:
                    edge = torch.tensor([tuple(sorted(self.selected_vertices))]).t()
                    if self.metanetwork.edge_exists(edge):  # Remove edge
                        self.metanetwork.remove_edge(edge)
                    else:  # Add edge
                        self.metanetwork.add_edge(edge.t())
                    self.edges = self.filter_edges_by_layer()
                    print(self.edges)
                    self.selected_vertices = []

            # Draw edges for the active layer
            active_edges = self.filter_edges_by_layer()
            headless = True
            edgeless = True
            pointless = True
            if not headless:
                if not edgeless:
                    for edge in active_edges:
                        start, end = edge
                        pygame.draw.line(self.screen, (200, 200, 255),
                                        screen_vertices[start].tolist(),
                                        screen_vertices[end].tolist(), 2)
                if not pointless:
                    # Draw vertices
                    # Normalize the "acceleration" field
                    if update_colors % 60 == 0 or colors is None:
                        normalized_field = self.normalize_field("acceleration")  # Shape: (N, 3)
                        intensities = (normalized_field * 255).to(torch.uint8)   # Scale to [0, 255] and ensure uint8
                        colors = torch.stack([intensities[:, 0], intensities[:, 1], intensities[:, 2]], dim=1)

                    # Draw vertices
                    for i, (x, y) in enumerate(screen_vertices):
                        if i in self.selected_vertices:
                            color = (255, 0, 0)  # Highlight selected vertices in red
                        else:
                            color = colors[i].tolist()  # Extract RGB triplet for vertex
                        pygame.draw.circle(self.screen, color, (int(x), int(y)), 2)
            # Update buffers with current camera transforms
            self.update_buffers()

            # Render with OpenGL
            self.render_opengl()

            
            # Display instructions
            #text_surface = self.font.render(
            #    "Left-click: Edit/Move | Drag: Move all | U: Undo | Arrow Keys: Rotate | S: Save",
            #    True, (255, 255, 255))
            #self.screen.blit(text_surface, (10, 10))

            #pygame.display.flip()
            #if not self.paused:
            #    #self.metanetwork.forward()
            self.clock.tick(60-(time.time() - start_time))

        pygame.quit()
    def normalize_field(self, field_name):
        field_index = self.metanetwork.field_index(field_name)
        field_values = self.metanetwork.data.x[:, field_index].squeeze()

        min_val, max_val = field_values.min(), field_values.max()
        normalized = (field_values - min_val) / (max_val - min_val + 1e-10)  # Avoid div-by-zero
        return normalized.clamp(0.0, 1.0)  # Clip between 0 and 1

    def project_back_to_3d(self, dx, dy):
        """Convert 2D screen displacement back to 3D space based on rotation."""
        # Create a 2D movement vector
        movement_2d = torch.tensor([dx, dy, 0], dtype=torch.float32)

        # Construct rotation matrices for current angles
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        cos_y, sin_y = math.cos(-self.angle_y), math.sin(-self.angle_y)

        # Inverse rotation matrix
        rotation_x_inv = torch.tensor([
            [1, 0, 0],
            [0, cos_x, sin_x],
            [0, -sin_x, cos_x]
        ], dtype=torch.float32)

        rotation_y_inv = torch.tensor([
            [cos_y, 0, -sin_y],
            [0, 1, 0],
            [sin_y, 0, cos_y]
        ], dtype=torch.float32)

        # Apply the inverse rotations to the movement
        movement_3d = torch.matmul(rotation_x_inv, movement_2d)
        movement_3d = torch.matmul(rotation_y_inv, movement_3d)

        return movement_3d
# Example Usage
if __name__ == "__main__":
    # Initialize empty MetaNetwork
    metanetwork = MetaNetwork()

    # Instantiate multiple shapes into the MetaNetwork
    #metanetwork.instantiate_shape(geometry_type="tetrahedron", position_offset=torch.tensor([0.0, 0.0, 0.0]))
    #metanetwork.instantiate_shape(geometry_type="cube", position_offset=torch.tensor([0.0, 0.0, 0.0]))
    #metanetwork.instantiate_shape(geometry_type="octahedron", position_offset=torch.tensor([0.0, 0.0, 0.0]))
    #metanetwork.instantiate_shape(geometry_type="icosahedron", position_offset=torch.tensor([4.0, 4.0, 0.0]))
    

    # Available shapes
    shape_types = ["sphere"]#"triangle", "square", "tetrahedron", "cube", "octahedron"]

    # Stress test configuration
    num_shapes = 1  # Number of shapes to generate
    jitter_strength = 1e-3  # Strength of the jitter to avoid singularities
    spacing = 0.1  # Base spacing between shapes
    import random
    # Generate shapes with jittered offsets
    for i in range(num_shapes):
        # Select a random shape
        shape_type = random.choice(shape_types)
        
        # Compute a base position offset
        base_offset = torch.tensor([i * spacing, 0.0, 0.0], dtype=torch.float32)
        
        # Add small random jitter to avoid zero positions
        jitter = (torch.rand(3) - 0.5) * jitter_strength  # Random jitter in [-jitter_strength, jitter_strength]
        position_offset = base_offset + jitter
        
        # Instantiate shape
        print(f"Instantiating {shape_type} at {position_offset}")
        metanetwork.instantiate_shape(geometry_type=shape_type, position_offset=position_offset)

    # Launch the visualizer
    editor = DECInteractiveVisualizer(metanetwork)
    metanetwork.forward_runner.start()
    editor.draw()
