import pygame
import numpy as np
import threading
import sys
import time
import select
import numpy as np
from queue import Queue
import uuid
from OpenGL.GL import *
from OpenGL.GLU import *
from hotrodprintingpress import GrandPrintingPress
pygame.font.init()
class Console:
    def __init__(self, dt, local_state, width=800, height=600, font_size=72):
        """
        Initialize the Console plugin.

        Args:
            dt (float): Time step for initialization (not stored).
            local_state (dict): Hyperlocal state for initialization context (not stored).
            width (int): Console width in pixels.
            height (int): Console height in pixels.
            font_size (int): Font size for rendering.
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.font_path = 'consola.ttf'
        self.printing_press = GrandPrintingPress(width, height)
        
        
        self.vb_out_buffer = []
 
        self.user_input_queue = Queue()  # Queue for user inputs
        self.program_instruction_queue = Queue()  # Queue for programmatic instructions

        self.running = True
        self.menu_tree = {}  # Decision-tree dictionary for text adventure
        self.current_menu = None  # Current state in the menu tree
        self.text_output_buffer = []  # Buffer for displaying messages
        self.current_input = ""  # Current user input
        self.last_blink_time = time.time()
        self.cursor_visible = True  # For blinking cursor

        self.initialize_menu_tree(local_state)

        # Start stdin reader thread
        self.stdin_thread = threading.Thread(target=self._stdin_reader_thread, daemon=True)
        self.stdin_thread.start()

    def initialize_menu_tree(self, local_state):
        """
        Initialize the menu tree with dynamic plugin list based on local_state.

        Args:
            local_state (dict): Hyperlocal state for menu initialization.
        """
        plugins = local_state.get("plugins", [])
        start_menu_options = {}
        for idx, plugin_name in enumerate(plugins, start=1):
            start_menu_options[str(idx)] = plugin_name
        start_menu_options[str(len(plugins) + 1)] = "exit"

        self.menu_tree = {"start": start_menu_options, "exit": {}}
        self.current_menu = "start"
        self._write_message("Welcome to the Console!")
        self._write_message("Available plugins:")
        for option, plugin_name in start_menu_options.items():
            self._write_message(f"{option}. {plugin_name}" if plugin_name != "exit" else f"{option}. Exit")

    def process_user_input(self, user_input, local_state):
        """
        Process user inputs and update the local state.

        Args:
            user_input (str): Input from the user.
            local_state (dict): Hyperlocal state for this input processing context.
        """
        user_input = user_input.strip()
        if self.current_menu and user_input in self.menu_tree.get(self.current_menu, {}):
            next_action = self.menu_tree[self.current_menu][user_input]
            if next_action == "exit":
                self.running = False
                self._write_message("Exiting...")
            elif next_action in local_state.get("plugins", []):
                self.program_instruction_queue.put({"action": "start_plugin", "plugin_name": next_action})
                self._write_message(f"Starting plugin: {next_action}")
                self.current_menu = "start"
            else:
                self.current_menu = next_action
                self._write_message(f"Moved to menu: {next_action}")
        else:
            self._write_message("Invalid choice. Try again.")


    def _render_text_to_texture(self, text, max_width, max_height, press: GrandPrintingPress, font_path: str, font_size: int):
        """
        Render text to a GPU texture using PyTorch tensors for efficient GPU integration.

        Args:
            text (str): The text to render.
            max_width (int): Maximum width of the rendering area.
            max_height (int): Maximum height of the rendering area.
            press (GrandPrintingPress): Instance of the GrandPrintingPress class for rendering.
            font_path (str): Path to the font file.
            font_size (int): Font size for rendering.

        Returns:
            int: OpenGL texture ID for the rendered text.
        """
        # Ensure the font is loaded into the GrandPrintingPress
        press.load_font(font_path, font_size)

        # Render text as a tensor
        rendered_tensor = press.print_text(text, font_path, font_size).to("cuda")  # Move tensor to GPU

        # Resize tensor if necessary to fit within the specified dimensions
        if rendered_tensor.shape[0] > max_height or rendered_tensor.shape[1] > max_width:
            rendered_tensor = torch.nn.functional.interpolate(
                rendered_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(max_height, max_width),
                mode="bilinear",
                align_corners=False
            ).squeeze().byte()

        # Add alpha channel if it doesn't already exist
        if rendered_tensor.dim() == 2:  # Grayscale tensor
            rendered_tensor = rendered_tensor.unsqueeze(0).repeat(4, 1, 1)  # Convert to RGBA with placeholder alpha
            rendered_tensor[3, :, :] = 255  # Set alpha to opaque

        # Ensure tensor is correctly formatted for OpenGL
        texture_data = rendered_tensor.permute(1, 2, 0).contiguous()  # Change to HWC format
        texture_data_np = texture_data.cpu().numpy()  # Transfer back to CPU for OpenGL compatibility

        # Generate OpenGL texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            texture_data_np.shape[1], texture_data_np.shape[0],
            0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data_np
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return texture


    def _render_to_vb(self):
        """
        Render text output and menu to textures, then package metadata.
        """
        max_width = self.width
        max_height = self.height // 2

        # Render scrollable text output (left side)
        output_texture = self._render_text_to_texture(
            "\n".join(self.text_output_buffer),
            max_width // 2,
            max_height,
            self.printing_press,
            self.font_path,
            self.font_size,
        )

        # Render menu (right side)
        menu_text = "\n".join(f"{key}: {action}" for key, action in self.menu_tree.get(self.current_menu, {}).items())
        menu_texture = self._render_text_to_texture(
            menu_text,
            max_width // 2,
            max_height,
            self.printing_press,
            self.font_path,
            self.font_size,
        )

        # Package metadata
        metadata = [
            {
                "buffer": output_texture,
                "buffer_type": "TEXTURE",
                "location": (0.0, 0.5),
                "size": (0.5, 0.5),
                "intended_time": -1,
                "duration": -1,
            },
            {
                "buffer": menu_texture,
                "buffer_type": "TEXTURE",
                "location": (0.5, 0.5),
                "size": (0.5, 0.5),
                "intended_time": -1,
                "duration": -1,
            },
        ]
        return {"0,0,0": metadata}

    def _format_authorized_message(self, message):
        """
        Format a message with a UUID and timestamp for authentication.

        Args:
            message (str): The message to format.

        Returns:
            str: The formatted message.
        """
        msg_uuid = str(uuid.uuid4())
        timestamp = time.time()
        return f"{msg_uuid} {timestamp} {message}"

    def authorized_print(self, message):
        """
        Print a message directly with authorization.

        Args:
            message (str): The message to print.
        """
        formatted_message = self._format_authorized_message(message)
        print(formatted_message)
    def start(self, dt, local_state):
        """
        Start the console plugin.

        Args:
            dt (float): Time step for initialization.
            local_state (dict): Initialization state context.
        """
        self.running = True  # Mark the console as running
        self._write_message("Console started.")

    def update(self, dt, local_state):
        """
        Update the console plugin.

        Args:
            dt (float): Time step for the update.
            local_state (dict): Current hyperlocal state for the update.

        Returns:
            dict: Instructions and updated state for VBO rendering.
        """
        # Process user input from the queue
        while not self.user_input_queue.empty():
            user_input = self.user_input_queue.get()
            self.process_user_input(user_input, local_state)

        # Capture new stdout from the stream interceptor
        txt_data = local_state.get('txt_in', [])
        if txt_data:
            self.text_output_buffer.extend(txt_data)
            # Clear txt_in after processing
            local_state['txt_in'] = []

        # Update the blinking cursor
        if time.time() - self.last_blink_time > 0.5:  # Blink every 500ms
            self.cursor_visible = not self.cursor_visible
            self.last_blink_time = time.time()

        vb = self._render_to_vb()
        self.vb_out_buffer.append(vb)
        # Generate VBO representation and push to local_state
        local_state["video_buffer_in"].append(self.vb_out_buffer)
        self.vb_out_buffer = []

        # Return any instructions
        instructions = []
        while not self.program_instruction_queue.empty():
            instruction = self.program_instruction_queue.get()
            instructions.append(instruction)
        return {"instructions": instructions}

    def handle_input_queue(self, user_input):
        """
        Push user input to the processing queue.

        Args:
            user_input (str): Input from the user.
        """
        self.user_input_queue.put(user_input)

    def stop(self, dt, local_state):
        """
        Stop the console plugin.

        Args:
            dt (float): Time step for shutdown.
            local_state (dict): State context for shutdown.
        """
        self.running = False
        self._write_message("Console stopped.")

    def _stdin_reader_thread(self):
        """
        Thread to read stdin input without blocking the main thread.
        """
        while self.running:
            try:
                
                user_input = sys.stdin.readline().strip()
                if user_input:
                    self.handle_input_queue(user_input)
            except Exception as e:
                # Handle exceptions if needed
                pass

    def _write_message(self, message):
        """
        Write a message to both the text output buffer and stdout.

        Args:
            message (str): The message to write.
        """
        # Append to text output buffer
        self.text_output_buffer.append(message)
        # Write to stdout (will be intercepted by StreamInterceptor)
        
        self.authorized_print(message)



from baseplugin import BasePlugin
import torch

class Plugin(BasePlugin):
    def __init__(self, dt, local_state, stream_interceptor=None):
        self.dt_draw = 70.0
        self.battery_level = 1000.0
        self.instance = Console(dt, local_state)

        super().__init__(dt, local_state, stream_interceptor=None, autostart=True, needs_power=True, has_battery=True, battery_level=self.battery_level, dt_draw=self.dt_draw)
        
    def _start_plugin(self, dt, local_state):
        self.instance.start(dt, local_state)

    def _update_plugin(self, dt, local_state):
        return self.instance.update(dt, local_state)

    def _stop_plugin(self, dt, local_state):
        self.instance.stop(dt, local_state)

    def get_class_name(self):
        return self.instance.__class__.__name__
