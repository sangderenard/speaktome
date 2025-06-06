# boxplot_generator.py

from OpenGL.GL import *
import torch
import numpy as np
import logging
from collections import deque

class BoxPlotGenerator:
    """
    Generates and renders box plots from 1D tensor inputs using OpenGL.
    Maintains a local circular queue to store recent data.
    """

    def __init__(self, queue_size=100, window_width=800, window_height=600):
        self.queue_size = queue_size
        self.data_queue = deque(maxlen=queue_size)
        self.window_width = window_width
        self.window_height = window_height

        # Initialize OpenGL buffers for box plot components
        self.vbo_lines = None
        self.vbo_rect = None

        logging.info("BoxPlotGenerator initialized.")

    def add_data(self, tensor):
        """
        Adds a new 1D tensor to the circular queue.
        """
        if not isinstance(tensor, torch.Tensor):
            logging.error("Input data must be a PyTorch tensor.")
            return
        if tensor.ndim != 1:
            logging.error("Input tensor must be 1-dimensional.")
            return
        self.data_queue.append(tensor.cpu().numpy())
        logging.info(f"Added data to queue. Queue size: {len(self.data_queue)}")

    def compute_statistics(self):
        """
        Computes box plot statistics from the current queue data.
        Returns a dictionary with min, Q1, median, Q3, and max.
        """
        if not self.data_queue:
            logging.warning("Data queue is empty.")
            return None

        # Concatenate all data in the queue
        all_data = np.concatenate(self.data_queue)
        all_data_sorted = np.sort(all_data)

        stats = {
            'min': np.min(all_data_sorted),
            'q1': np.percentile(all_data_sorted, 25),
            'median': np.median(all_data_sorted),
            'q3': np.percentile(all_data_sorted, 75),
            'max': np.max(all_data_sorted)
        }

        logging.info(f"Computed statistics: {stats}")
        return stats

    def _normalize(self, value, min_val, max_val):
        """
        Normalizes a value to the OpenGL coordinate system (-1 to 1).
        """
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def render(self):
        """
        Renders the box plot using OpenGL.
        """
        stats = self.compute_statistics()
        if stats is None:
            return

        # Normalize statistics for OpenGL coordinates
        min_val = stats['min']
        max_val = stats['max']

        q1 = self._normalize(stats['q1'], min_val, max_val)
        median = self._normalize(stats['median'], min_val, max_val)
        q3 = self._normalize(stats['q3'], min_val, max_val)
        gl_min = self._normalize(stats['min'], min_val, max_val)
        gl_max = self._normalize(stats['max'], min_val, max_val)

        # Clear previous buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw box (from Q1 to Q3)
        box_vertices = np.array([
            -0.5, q1,
             0.5, q1,
             0.5, q3,
            -0.5, q3,
            -0.5, q1
        ], dtype=np.float32)

        if self.vbo_rect:
            glDeleteBuffers(1, [self.vbo_rect])

        self.vbo_rect = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_rect)
        glBufferData(GL_ARRAY_BUFFER, box_vertices.nbytes, box_vertices, GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glColor3f(0.0, 0.5, 1.0)  # Blue box
        glDrawArrays(GL_LINE_STRIP, 0, len(box_vertices)//2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Draw median line
        median_vertices = np.array([
            -0.5, median,
             0.5, median
        ], dtype=np.float32)

        if self.vbo_lines:
            glDeleteBuffers(1, [self.vbo_lines])

        self.vbo_lines = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glBufferData(GL_ARRAY_BUFFER, median_vertices.nbytes, median_vertices, GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glColor3f(1.0, 0.0, 0.0)  # Red median
        glDrawArrays(GL_LINES, 0, len(median_vertices)//2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Draw whiskers
        whisker_vertices = np.array([
            0.0, gl_min,
            0.0, q1,
            0.0, q3,
            0.0, gl_max
        ], dtype=np.float32)

        self.vbo_lines = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glBufferData(GL_ARRAY_BUFFER, whisker_vertices.nbytes, whisker_vertices, GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glColor3f(0.0, 1.0, 0.0)  # Green whiskers
        glDrawArrays(GL_LINES, 0, len(whisker_vertices)//2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Swap buffers if using double buffering
        # This part depends on your OpenGL context setup
        # For example, if using GLUT:
        # glutSwapBuffers()

    def cleanup(self):
        """
        Cleans up OpenGL resources.
        """
        if self.vbo_rect:
            glDeleteBuffers(1, [self.vbo_rect])
        if self.vbo_lines:
            glDeleteBuffers(1, [self.vbo_lines])
        logging.info("BoxPlotGenerator resources cleaned up.")
