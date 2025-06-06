# computer_screen.py

import logging
from Primitives.device import Device
import pygame
import threading
import queue
import numpy as np
from Renderer.tensor_renderer import TensorRenderer

class ComputerScreen(Device):
    def __init__(self, object_id, position, orientation, thread_manager,
                 heat_generation=30.0, energy_consumption=80.0,
                 width=800, height=600, title="Computer Screen", dvibus=None):
        super().__init__(object_id, position, orientation, thread_manager,
                         heat_generation, energy_consumption)
        self.width = width
        self.height = height
        self.title = title
        self.thread_manager = thread_manager
        self.window_texture = None  # Texture from the Window/MitsubaShell
        # Initialize Pygame and set up the screen
        self._init_pygame()
        self.tensor_renderer = TensorRenderer(width=width, height=height)
        

        # Device communication bus
        self.dvibus = dvibus

        # Flags for controlling the display thread
        self.running = True

        # Buffers
        self.managed_buffer = {"vertices": [], "textures": []}

        # Input and data queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Lock for synchronizing access to the managed buffer
        self.buffer_lock = threading.Lock()
        self.input_lock = threading.Lock()

        # Thread for the display and input loop
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)

        logging.info(f"ComputerScreen '{self.object_id}' initialized.")
        
        # Send diagnostic mode data
        self._start_diagnostic_mode()
        
        # Start the rendering thread
        self.display_thread.start()

    def _start_diagnostic_mode(self):
        """
        Sends diagnostic mode buffers to initialize the screen with a basic gradient and overlay.
        """
        vertices = np.array([
            [-0.5, -0.5, 0.0],  # Bottom-left
            [0.5, -0.5, 0.0],   # Bottom-right
            [-0.5, 0.5, 0.0],   # Top-left
            [0.5, 0.5, 0.0]     # Top-right
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,  # First triangle
            2, 1, 3   # Second triangle
        ], dtype=np.uint32)

        colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red (Bottom-left)
            [0.0, 1.0, 0.0, 1.0],  # Green (Bottom-right)
            [0.0, 0.0, 1.0, 1.0],  # Blue (Top-left)
            [1.0, 1.0, 0.0, 1.0]   # Yellow (Top-right)
        ], dtype=np.float32)

        overlay_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        overlay_data[250:350, 350:450, :3] = [255, 255, 255]  # White square
        overlay_data[250:350, 350:450, 3] = 255  # Fully opaque

        overlay_instructions = {"alpha": 1.0}  # No blending

        # Deliver diagnostic buffers to the input queue
        self.deliver({
            "layout": [
                {"vertices": vertices, "indices": indices, "colors": colors}
            ],
            "overlay_data": overlay_data,
            "overlay_instructions": overlay_instructions
        })
    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()


    def _display_loop(self):
        """
        Main rendering loop. Handles rendering data from `input_queue` and passes user events to `output_queue`.
        """
        
        while self.running:
            # Check for rendering data from the input queue
            try:
                with self.input_lock:
                    render_data = self.input_queue.get_nowait()
                    
                    if isinstance(render_data, dict):
                        layout = render_data.get("layout", [])
                        overlay_data = render_data.get("overlay_data", None)
                        overlay_instructions = render_data.get("overlay_instructions", None)

                        # Pass the data to the TensorRenderer for processing
                        self.tensor_renderer.mix_and_render(layout, overlay_data, overlay_instructions)
            except queue.Empty:
                pass  # No data to process

            # Collect user events (provided externally) from the rendering engine
            while not self.output_queue.empty():
                user_event = self.output_queue.get()
                logging.debug(f"ComputerScreen '{self.object_id}' processed event: {user_event}")

            # Limit frame rate
            self.clock.tick(60)

        pygame.quit()

    def update_window_texture(self):
        """
        Retrieves the latest rendering from the MitsubaShell and updates the window texture.
        """
        mitsuba_render = self.thread_manager.mitsuba_shell.get_render_output()
        if mitsuba_render:
            # Convert the Mitsuba render output to a Pygame surface
            self.window_texture = pygame.surfarray.make_surface(mitsuba_render)
            logging.debug("ComputerScreen updated window texture from MitsubaShell.")

    def _get_blend_mode(self):
        """
        Determines the current blend mode for processing data.
        Can be extended to read from configuration or external input.
        """
        # Example: Static configuration for now
        return "latest"  # Change to "stack" or "blit" as needed

    def _process_latest_data(self):
        """
        Extracts the newest data from the managed buffer to create a layout plan.
        """
        vertices = self.managed_buffer.get("vertices", [])
        layout = [{"vertices": vertices, "indices": [], "colors": []}]  # Example layout
        logging.debug(f"Processing latest data into layout: {layout}")
        return layout

    def _process_additive_data(self):
        """
        Combines all delivered data into a stacked layout plan.
        """
        vertices = self.managed_buffer.get("vertices", [])
        stacked_layout = [{"vertices": v, "indices": [], "colors": []} for v in vertices]
        logging.debug(f"Processing additive stacked data into layout: {stacked_layout}")
        return stacked_layout

    def _process_blit_data(self):
        """
        Blends new data with existing layout in a blit-like manner.
        """
        # Example: Combine data into a single composite layout
        vertices = self.managed_buffer.get("vertices", [])
        textures = self.managed_buffer.get("textures", [])
        composite_layout = [{"vertices": vertices, "textures": textures, "colors": []}]
        logging.debug(f"Processing blit data into layout: {composite_layout}")
        return composite_layout
    def update(self, dt):
        """
        Updates the state of the ComputerScreen.
        Regulates data from the managed buffer, processes it into layout plans,
        and prepares it for rendering based on the blend mode.
        """
        super().update(dt)  # Call parent update logic if applicable
        print("before the lock")
        with self.buffer_lock:
            print("after the lock")
            # Process data into layout plans based on blend mode
            blend_mode = self._get_blend_mode()  # Determine the blend mode
            if blend_mode == "latest":
                layout = self._process_latest_data()
            elif blend_mode == "stack":
                layout = self._process_additive_data()
            elif blend_mode == "blit":
                layout = self._process_blit_data()
            else:
                raise ValueError(f"Unknown blend mode: {blend_mode}")

            # Add the layout to the input queue for rendering
            with self.input_lock:
                self.input_queue.put({
                    "layout": layout,
                    "overlay_data": self.managed_buffer.get("textures", None),
                    "overlay_instructions": {"blend_mode": blend_mode}
                })

            logging.debug(f"ComputerScreen '{self.object_id}' prepared layout with blend mode '{blend_mode}'.")


    def deliver(self, data):
        """
        Accepts a dictionary of data to deliver to the buffer.
        This method can be called by the main simulation loop or other devices.
        """
        with self.buffer_lock:
            if "vertices" in data:
                self.managed_buffer["vertices"] = data["vertices"]
            if "textures" in data:
                self.managed_buffer["textures"] = data["textures"]
            logging.debug(f"ComputerScreen '{self.object_id}' updated buffers with new data.")

    def get_input_events(self):
        """
        Retrieves all input events captured by the display thread.
        """
        events = []
        while not self.input_queue.empty():
            events.append(self.input_queue.get())
        return events

    def stop(self):
        """
        Stops the display thread.
        """
        self.running = False
        self.display_thread.join()
        logging.info(f"ComputerScreen '{self.object_id}' stopped.")
        # Clean up Pygame
        pygame.quit()

    def send_data(self, data):
        """
        Sends data via DVIBus to another device.
        """
        if self.dvibus:
            self.dvibus.send_data(self, data)

    def receive_data(self, data):
        """
        Receives data from another device via DVIBus.
        """
        logging.info(f"ComputerScreen '{self.object_id}' received data: {data}")
        # Update the display buffers with the received data
        self.deliver(data)

# Rest of the methods remain the same or are adjusted accordingly
