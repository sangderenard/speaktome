# dvibus.py
from OpenGL.GL import *
import torch
import logging
from queue import Queue
from Renderer.boxplot_generator import BoxPlotGenerator

class DVIBus:
    """
    A simple bus class for bidirectional data transfer between devices.
    """

    def __init__(self, device_a, device_b, queue_size=100):
        self.device_a = device_a
        self.device_b = device_b
        self.buffer_a_to_b = []
        self.buffer_b_to_a = []
        self.input_queue_a = Queue()
        self.input_queue_b = Queue()
        self.box_plot_a = BoxPlotGenerator(queue_size=queue_size)
        self.box_plot_b = BoxPlotGenerator(queue_size=queue_size)
        logging.info(f"DVIBus initialized between '{device_a.object_id}' and '{device_b.object_id}'.")

    def update(self):
        """
        Transfers data between devices and updates box plots.
        """
        # Transfer data from device A to device B
        while self.buffer_a_to_b:
            data = self.buffer_a_to_b.pop(0)
            self.device_b.receive_data(data)
            logging.info(f"Data transferred from '{self.device_a.object_id}' to '{self.device_b.object_id}': {data}")
            self.box_plot_b.add_data(data)

        # Transfer data from device B to device A
        while self.buffer_b_to_a:
            data = self.buffer_b_to_a.pop(0)
            self.device_a.receive_data(data)
            logging.info(f"Data transferred from '{self.device_b.object_id}' to '{self.device_a.object_id}': {data}")
            self.box_plot_a.add_data(data)

    def send_data(self, sender, data):
        """
        Translates PyTorch tensors into OpenGL buffers and queues them for rendering.
        Also updates the box plot with the new data.
        """
        if isinstance(data, torch.Tensor):
            if data.ndim == 0:  # Scalar tensor
                data = data.unsqueeze(0)  #
            if data.ndim == 2:  # Assume vertex data (e.g., [N, 3] or [N, 2])
                buffer_data = self._create_vertex_buffer(data)
                queue_data = {"vertices": buffer_data}
            elif data.ndim == 3:  # Assume texture data (e.g., [H, W, C])
                texture_id = self._create_texture(data)
                queue_data = {"textures": texture_id}
            elif data.ndim == 1:
                queue_data = data
                # Update the box plot
                if sender == self.device_a:
                    self.box_plot_b.add_data(data)
                elif sender == self.device_b:
                    self.box_plot_a.add_data(data)
            else:
                logging.warning(f"Unsupported tensor dimensionality: {data.ndim}. {data}")
                return

            # Send the processed data to the appropriate device
            if sender == self.device_a:
                self.device_b.input_queue.put(queue_data)
            elif sender == self.device_b:
                self.device_a.input_queue.put(queue_data)
        else:
            super().send_data(sender, data)  # Handle non-tensor data

    def render_box_plots(self):
        """
        Renders box plots for both devices.
        """
        logging.info("Rendering box plots for Device A and Device B.")
        self.box_plot_a.render()
        self.box_plot_b.render()

    def cleanup(self):
        """
        Cleans up resources for box plots.
        """
        self.box_plot_a.cleanup()
        self.box_plot_b.cleanup()
        logging.info("DVIBus cleanup completed.")


    def _create_vertex_buffer(self, tensor):
        """
        Creates an OpenGL Vertex Buffer Object (VBO) from a PyTorch tensor.
        """
        tensor_np = tensor.cpu().numpy().astype(np.float32)  # Convert to NumPy
        vbo_id = glGenBuffers(1)  # Generate a new VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBufferData(GL_ARRAY_BUFFER, tensor_np.nbytes, tensor_np, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)  # Unbind for safety
        return vbo_id

    def _create_texture(self, tensor):
        """
        Creates an OpenGL texture from a PyTorch tensor.
        """
        tensor_np = tensor.permute(2, 0, 1).cpu().numpy().astype(np.uint8)  # Convert to NumPy and reorder channels
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tensor_np.shape[2], tensor_np.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tensor_np)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)  # Unbind for safety
        return texture_id
