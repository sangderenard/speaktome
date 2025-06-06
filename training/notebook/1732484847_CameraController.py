# renderer/camera_controller.py

import numpy as np
import pygame
import quaternion  # Ensure you have numpy-quaternion installed

class CameraController:
    def __init__(self, config, engine):
        self.engine = engine
        self.config = config
        self.camera_pos = np.array([2.0, 2.0, 1.0])  # Initial camera position
        self.forward = np.array([0.0, 0.0, -1.0])  # Initial forward direction
        self.up = np.array([0.0, 1.0, 0.0])  # Up direction
        self.right = np.array([1.0, 0.0, 0.0])  # Right direction for strafing
        self.speed = 0.1  # Movement speed
        self.mouse_sensitivity = 1e-2  # Sensitivity for looking around
        self.projection_mode = 0  # 0: Perspective, 1: Isometric, 2: Flat
        self.active_vertices = None  # To store the current active vertices for near/far calculation
        self.near = 1.0  # Default near plane
        self.far = 100.0  # Default far plane
        self.orientation = np.quaternion(1, 0, 0, 0)  # Quaternion representing the initial orientation
        self.yaw = 0
        self.pitch = 0
        self.light_lock = 0

    def update(self, delta_radius=0.0, delta_theta=0.0, delta_phi=0.0):
        # Unused now, as we're enabling free camera movement.
        pass
    def get_light_position(self):
        """Calculate light position offset from the camera's position to simulate a handheld flashlight."""
        # Define the offsets (adjust these values to match one forearm length)
        forward_distance = -0.5  # Forward by 0.5 units
        right_distance = 5.0    # Right by 0.3 units
        down_distance = -1.00     # Down by 0.2 units

        # Calculate the offset vectors
        forward_offset = self.forward * forward_distance
        right_offset = self.right * right_distance
        down_offset = -self.up * down_distance  # Negative to move down

        # Compute the new light position
        return self.camera_pos + forward_offset + right_offset + down_offset

    def process_keyboard_input(self, keys):
        """Process keyboard input for free movement."""
        if keys[pygame.K_w]:  # Move forward
            self.camera_pos += self.forward * self.speed
        if keys[pygame.K_s]:  # Move backward
            self.camera_pos -= self.forward * self.speed
        if keys[pygame.K_a]:  # Move left (strafe)
            self.camera_pos -= self.right * self.speed
        if keys[pygame.K_d]:  # Move right (strafe)
            self.camera_pos += self.right * self.speed
        if keys[pygame.K_q]:  # Move up
            self.camera_pos += self.up * self.speed
        if keys[pygame.K_z]:  # Move down
            self.camera_pos -= self.up * self.speed

        # Buffer selection logic
        if keys[pygame.K_0]:
            self.engine.set_selected_output((0,0))
        elif keys[pygame.K_1]:
            self.engine.set_selected_output((0,1))
        elif keys[pygame.K_2]:
            self.engine.set_selected_output((0,2))
        elif keys[pygame.K_3]:
            self.engine.set_selected_output((1,0))
        elif keys[pygame.K_4]:
            self.engine.set_selected_output((1,1))
        elif keys[pygame.K_5]:
            self.engine.set_selected_output((1,2))
        elif keys[pygame.K_6]:
            self.engine.set_selected_output((1,3))





        if keys[pygame.K_SPACE] and self.light_lock <= 0:
            self.light_lock = 100
            # Add a new unidirectional light at the current camera position, aiming in the forward direction
            camera_pos = self.get_camera_position()
            forward_direction = self.forward
            light_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # White light
            self.engine.lighting.add_light(camera_pos, forward_direction, light_color)
            print(f"Added light at position: {camera_pos}, direction: {forward_direction}")
        elif self.light_lock > 0:
            self.light_lock -= 1

        if verbose:
            print(f"Camera position after input: {self.camera_pos}")

    def process_mouse_movement(self, mouse_dx, mouse_dy):
        """Process mouse movement for looking around using quaternions."""
        yaw = -mouse_dx * self.mouse_sensitivity
        pitch = -mouse_dy * self.mouse_sensitivity

        # Create quaternions for pitch and yaw
        yaw_rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
        pitch_rotation = np.quaternion(np.cos(pitch / 2), np.sin(pitch / 2), 0, 0)

        # Update the orientation
        self.orientation = yaw_rotation * self.orientation * pitch_rotation
        self.orientation = self.orientation.normalized()

        # Update camera direction vectors based on orientation
        self.update_camera_vectors()
        if verbose:
            print(f"Orientation after mouse movement: {self.orientation}")
            print(f"Forward vector after mouse movement: {self.forward}")
            print(f"Up vector after mouse movement: {self.up}")
            print(f"Right vector after mouse movement: {self.right}")

    def update_camera_vectors(self):
        """Update the camera's forward, right, and up vectors based on the quaternion orientation."""
        # Rotate the initial forward vector
        forward_quat = self.orientation * np.quaternion(0, 0, 0, -1) * self.orientation.conjugate()
        self.forward = np.array([forward_quat.x, forward_quat.y, forward_quat.z])

        # Rotate the initial up vector
        up_quat = self.orientation * np.quaternion(0, 0, 1, 0) * self.orientation.conjugate()
        self.up = np.array([up_quat.x, up_quat.y, up_quat.z])
        # Calculate right vector by cross product of forward and up
        self.right = np.cross(self.forward, self.up)

        if verbose:
            print(f"Updated forward vector: {self.forward}")
            print(f"Updated up vector: {self.up}")
            print(f"Updated right vector: {self.right}")

    def get_camera_position(self):
        return self.camera_pos

    def get_view_matrix(self):
        """Return the lookAt matrix based on the current camera position and direction."""
        look_at_point = self.camera_pos + self.forward
        view_matrix = self.lookAt(self.camera_pos, look_at_point, self.up)
        if verbose:
            print(f"View matrix: {view_matrix}")
        return view_matrix

    def update_active_vertices(self, vertices):
        """Update the active vertices for near/far plane calculation."""
        self.active_vertices = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        self.calculate_near_far()

    def calculate_near_far(self, margin=1.2):
        """Automatically calculate the near and far planes based on active vertices."""
        if self.active_vertices is not None:
            camera_pos = np.array(self.get_camera_position())
            distances = np.linalg.norm(self.active_vertices - camera_pos, axis=1)
            self.near = max(np.min(distances) * 0.5, 0.1)  # Minimum near plane to avoid clipping
            self.far = np.max(distances) * margin  # Apply margin to the far plane
        if verbose:
            print(f"Near plane: {self.near}, Far plane: {self.far}")

    def calculate_data_extents(self, vertices, padding_factor=0.1):
        """Calculate the extents for orthographic camera projection."""
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()

        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)

        left, right = min_vals[0], max_vals[0]
        bottom, top = min_vals[1], max_vals[1]
        near, far = min_vals[2], max_vals[2]

        # Add a bit of padding around the bounding box
        left -= (right - left) * padding_factor
        right += (right - left) * padding_factor
        bottom -= (top - bottom) * padding_factor
        top += (top - bottom) * padding_factor
        near = max(near - (far - near) * padding_factor, 0.1)  # Ensure near > 0.1
        far += (far - near) * padding_factor

        if verbose:
            print(f"Data extents - Left: {left}, Right: {right}, Bottom: {bottom}, Top: {top}, Near: {near}, Far: {far}")
        return left, right, bottom, top, near, far

    def apply_camera(self, engine):
        camera_pos = self.get_camera_position()
        forward = self.forward
        up = self.up
        if self.projection_mode == 0:
            engine.set_perspective_camera(camera_pos, forward, up, self.near, self.far, self.get_view_matrix())
        elif self.projection_mode == 1:
            left, right, bottom, top, near, far = self.calculate_data_extents(self.active_vertices)
            engine.set_orthographic_camera(left, right, bottom, top, near, far, camera_pos, forward, up, self.get_view_matrix())
        elif self.projection_mode == 2:
            engine.set_flat_camera(-10, 10, -10, 10, 1, 100, camera_pos, forward, up, self.get_view_matrix())
        if verbose:
            print(f"Camera applied with position: {camera_pos}, forward: {forward}, up: {up}")

    def toggle_projection(self):
        self.projection_mode = (self.projection_mode + 1) % 3
        if verbose:
            print(f"Projection mode toggled to: {self.projection_mode}")

    def lookAt(self, eye, target, up):
        f = (target - eye)
        f /= np.linalg.norm(f)
        up_norm = up / np.linalg.norm(up)
        s = np.cross(f, up_norm)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        m = np.identity(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m.T  # Note the transpose