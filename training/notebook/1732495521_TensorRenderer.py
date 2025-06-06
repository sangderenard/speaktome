import logging
import threading
import queue
import torch
import pygame
from OpenGL.GL import *
from .renderer_config import RendererConfig
from .render_engine import RenderEngine
from .camera_controller import CameraController
from .test_pattern_generator import TestPatternGenerator

class TensorRenderer:
    def __init__(self, config=None, max_queue_size=1, width=800, height=600):
        logging.info("Initializing TensorRenderer.")
        self.config = config or RendererConfig()
        self.engine = RenderEngine(self.config)
        self.camera_controller = CameraController(self.config, self.engine)
        self.engine.set_controller(self.camera_controller)
        self.active_vertices = None
        self.data = None
        self.colors = None
        self.buffer_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_thread = False
        self.static_phase_buffers = {}
        self.cache_lock = threading.Lock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.debug(f"TensorRenderer device set to {self.device}.")
        self.test_pattern_generator = None
        self.width = width
        self.height = height
        self.overlay_texture = None
        self.init_overlay_buffer()

    def init_overlay_buffer(self):
        """
        Initialize a blank overlay texture for HUD rendering or fallback imagery.
        """
        logging.info("Initializing overlay buffer.")
        self.overlay_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.overlay_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        logging.debug("Overlay buffer initialized and bound.")

    def mix_and_render(self, layout, overlay_data=None, overlay_instructions=None):
        """
        Mix vertex and texture buffers, then render with optional overlay.

        Parameters:
        - layout: List of dictionaries specifying vertex and texture buffers and mixing rules.
        - overlay_data: Optional 2D tensor or numpy array for overlay texture (RGBA format).
        - overlay_instructions: Dictionary with control flags for overlay (e.g., alpha blending).
        """
        logging.info("Mixing buffers and rendering.")
        combined_vertices, combined_indices, combined_normals, combined_colors = self._mix_buffers(layout)
        logging.debug(f"Combined vertices: {len(combined_vertices)}, indices: {len(combined_indices)}.")

        # Render mixed data
        self.render(combined_vertices, combined_indices, combined_normals, combined_colors)

        # Render overlay if provided
        if overlay_data is not None:
            logging.info("Rendering overlay.")
            self.update_overlay_buffer(overlay_data)
            self.render_overlay(overlay_instructions)

    def _mix_buffers(self, layout):
        """
        Combine vertex, index, normal, and color data from layout instructions.
        Returns combined buffers for rendering.
        """
        logging.info("Mixing buffers from layout instructions.")
        combined_vertices = []
        combined_indices = []
        combined_normals = []
        combined_colors = []

        for instruction in layout:
            vertices = instruction.get("vertices")
            indices = instruction.get("indices")
            normals = instruction.get("normals", None)
            colors = instruction.get("colors", None)

            # Apply transformations if specified
            transform = instruction.get("transform", None)
            if transform is not None:
                vertices = (vertices @ transform[:3, :3].T) + transform[:3, 3]
                logging.debug(f"Applied transformation: {transform}.")

            # Adjust indices for vertex offset
            if indices is not None:
                offset = sum(len(v) for v in combined_vertices)
                indices = indices + offset

            combined_vertices.append(vertices)
            if indices is not None:
                combined_indices.append(indices)
            if normals is not None:
                combined_normals.append(normals)
            if colors is not None:
                combined_colors.append(colors)

        # Concatenate buffers
        final_vertices = np.concatenate(combined_vertices, axis=0).astype(np.float32)
        final_indices = np.concatenate(combined_indices, axis=0).astype(np.uint32) if combined_indices else None
        final_normals = np.concatenate(combined_normals, axis=0).astype(np.float32) if combined_normals else None
        final_colors = np.concatenate(combined_colors, axis=0).astype(np.float32) if combined_colors else None

        logging.debug("Buffers mixed successfully.")
        return final_vertices, final_indices, final_normals, final_colors

    def update_overlay_buffer(self, overlay_data):
        """
        Update the overlay buffer with new 2D image data (RGBA format).
        """
        logging.info("Updating overlay buffer.")
        glBindTexture(GL_TEXTURE_2D, self.overlay_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, overlay_data)
        glBindTexture(GL_TEXTURE_2D, 0)
        logging.debug("Overlay buffer updated.")

    def render_overlay(self, instructions):
        """
        Render the overlay buffer with optional alpha blending.

        Parameters:
        - instructions: Dictionary specifying overlay options:
          - 'alpha': Alpha blending value (0.0 - 1.0).
        """
        logging.info("Rendering overlay with alpha blending.")
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        alpha = instructions.get("alpha", 1.0)
        logging.debug(f"Alpha blending value: {alpha}.")

        glUseProgram(self.overlay_shader)
        glUniform1f(glGetUniformLocation(self.overlay_shader, "alpha"), alpha)
        glBindTexture(GL_TEXTURE_2D, self.overlay_texture)

        # Render fullscreen quad for overlay
        self._render_fullscreen_quad()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_BLEND)
        logging.info("Overlay rendering complete.")

    def initialize_test_pattern_generator(self):
        # Initialize the test pattern generator
        if self.test_pattern_generator is None:
            logging.info("Initializing test pattern generator.")
            self.test_pattern_generator = TestPatternGenerator(self.device)

    def render_still(self, vertices, indices, normals, color_buffers, phase_index):
        """Main rendering function to display a single phase frame."""
        logging.info("Rendering still frame.")
        self.active_vertices = vertices.clone().detach()
        self.active_indices = indices.clone().detach()
        self.active_normals = normals.clone().detach()
        self.camera_controller.update_active_vertices(self.active_vertices)
        self.camera_controller.apply_camera(self.engine)

        # Call render with the necessary buffers and phase slice
        self.engine.render(self.active_vertices, self.active_indices, self.active_normals, color_buffers, phase_index)
        logging.debug("Still frame rendered successfully.")

    # Remaining methods retain the same structure but now include logging for each significant operation
