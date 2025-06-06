import threading
import time
from queue import Queue
import pygame
import logging
import os

class Console:
    def __init__(self):
        """
        Initialize the Console plugin.
        """
        self.user_input_queue = Queue()  # Queue for user inputs
        self.program_instruction_queue = Queue()  # Queue for programmatic instructions
        self.local_hash = {"user_input": [], "action_history": [], "mouse_events": [], "keyboard_events": [], "txt_data": []}
        self.console_thread = None
        self.running = True
        self.menu_tree = {}  # Decision-tree dictionary for text adventure
        self.current_menu = None  # Current state in the menu tree
        self.text_output_buffer = []  # Buffer for displaying messages
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def start(self):
        """
        Start the console thread for user interaction.
        """
        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()
        logging.info("Console plugin started.")

    def stop(self):
        """
        Stop the console thread and clean up.
        """
        self.running = False
        if self.console_thread:
            self.console_thread.join()
        logging.info("Console plugin stopped.")

    def console_loop(self):
        """
        Run the console input loop in its own thread.
        """
        while self.running:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear console for scrolling effect
            with self.lock:
                # Print buffered output
                for line in self.text_output_buffer[-20:]:  # Display last 20 lines
                    print(line)

            print("\n>>> ", end="", flush=True)  # Input prompt
            user_input = input()  # User input
            with self.lock:
                self.user_input_queue.put(user_input)

    def process_user_input(self):
        """
        Process user inputs from the queue and update the local hash.
        """
        while not self.user_input_queue.empty():
            user_input = self.user_input_queue.get()
            self.local_hash["user_input"].append({"timestamp": time.time(), "input": user_input})
            self.parse_menu_input(user_input)

    def parse_menu_input(self, user_input):
        """
        Parse user input based on the current menu tree and update programmatic instructions.
        """
        if self.current_menu and user_input in self.menu_tree.get(self.current_menu, {}):
            next_menu = self.menu_tree[self.current_menu][user_input]
            if callable(next_menu):
                next_menu = next_menu(self.local_hash)  # Resolve menu dynamically based on local state
            self.current_menu = next_menu
            self.text_output_buffer.append(f"Moved to menu: {next_menu}")
        elif self.current_menu:
            self.text_output_buffer.append("Invalid choice. Try again.")
        else:
            self.text_output_buffer.append("No active menu. Type 'start' to begin.")

        # Submit programmatic instructions (if any) associated with the current menu
        if isinstance(self.menu_tree.get(self.current_menu, {}), dict) and "instructions" in self.menu_tree[self.current_menu]:
            self.program_instruction_queue.put(self.menu_tree[self.current_menu]["instructions"])

    def initialize_menu_tree(self):
        """
        Initialize a dynamic menu tree with inventory awareness.
        """
        self.menu_tree = {
            "start": {
                "1": "explore",
                "2": "check_inventory",
                "3": "exit",
            },
            "explore": {
                "1": self.generate_dynamic_menu,
                "2": "back",
            },
            "check_inventory": self.check_inventory,
            "exit": {},
        }
        self.current_menu = "start"

    def generate_dynamic_menu(self, local_hash):
        """
        Generate a dynamic menu based on the local hash and field state.
        Args:
            local_hash (dict): The current local hash of the game.
        Returns:
            dict: A dynamically created menu based on local state.
        """
        density_map = local_hash.get("density_map", {})
        force_map = local_hash.get("force_map", {})
        nearby_objects = self.scan_local_area(density_map, force_map)

        dynamic_menu = {"back": "explore"}
        for obj_id, obj_data in nearby_objects.items():
            dynamic_menu[f"inspect_{obj_id}"] = f"Inspect {obj_data['name']}"

        self.text_output_buffer.append("Dynamic menu generated based on local field state.")
        return dynamic_menu

    def scan_local_area(self, density_map, force_map):
        """
        Analyze the local density and force maps for nearby objects or opportunities.
        Args:
            density_map (dict): The density map from the local hash.
            force_map (dict): The force map from the local hash.
        Returns:
            dict: A dictionary of nearby objects with their properties.
        """
        nearby_objects = {}
        for obj_id, density in density_map.items():
            if density > 0.5:  # Example threshold for significant presence
                nearby_objects[obj_id] = {
                    "name": f"Object_{obj_id}",
                    "density": density,
                    "force": force_map.get(obj_id, 0),
                }
        return nearby_objects

    def check_inventory(self, local_hash):
        """
        Generate a menu to display the current inventory based on the local hash.
        Args:
            local_hash (dict): The current local hash of the game.
        Returns:
            dict: Inventory menu.
        """
        inventory = local_hash.get("inventory", [])
        self.text_output_buffer.append("Inventory:")
        for item in inventory:
            self.text_output_buffer.append(f"- {item}")
        return "start"  # Return to the main menu

    def update(self, dt, local_state):
        """
        Update the console plugin with the current dt and process local state.
        
        Args:
            dt (float): Time interval for the current update cycle.
        """
        self.process_user_input()

        # Generate a programmatic instruction queue to return
        instructions = []
        while not self.program_instruction_queue.empty():
            instructions.append(self.program_instruction_queue.get())
        # Retrieve new text data from the local hash
        txt_data = local_state.get("txt_in", [])
        with self.lock:
            self.text_output_buffer.extend(txt_data)
            local_state["txt_in"].clear()

        return {"instructions": instructions, "text_output": self.text_output_buffer}

    def handle_pygame_input(self, events):
        """
        Handle pygame input events (keyboard and mouse) and update the local hash.

        Args:
            events (list): List of pygame events.
        """
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
                self.local_hash["mouse_events"].append({"timestamp": time.time(), "event": event})
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                self.local_hash["keyboard_events"].append({"timestamp": time.time(), "event": event})

    def render_text_output(self):
        """
        Render text output from the buffer to the console (can be extended for VBO rendering).
        """
        while self.text_output_buffer:
            message = self.text_output_buffer.pop(0)
            print(message)

from baseplugin import BasePlugin
import torch
class Plugin(BasePlugin):
    def __init__(self, dt, local_state):
        self.dt_draw = 1.0
        self.battery_level = 1.0
        super().__init__(dt, local_state, autostart=True, needs_power=True, stdout_redirect=False, has_battery=True, battery_level=self.battery_level, dt_draw=self.dt_draw)
        
        transients = super()._get_initial_power_transients()
        self.instance = Console()
        transients = torch.cat([self.instance._get_initial_power_transients(), transients])
        super()._instantaneous_power_queue.put(transients)
        has_mains = super()._has_mains_power()
        if not has_mains:
            self.battery_level -= torch.sum(transients) + self.dt_draw * dt
        
    def start(self, dt, local_state):
        return self.instance.start()

    def update(self, dt, local_state):
        draw = dt * self.dt_draw
        draw += torch.sum(self.instance._instantaneous_power_queue.get())
        self.battery_level -= draw
        if self.battery_level <= 0:
            self.stop(dt, local_state)
        return self.instance.update(dt, local_state)

    def stop(self, dt, local_state):
        return self.instance.stop()

    def get_class_name(self):
        return self.instance.__class__.__name__
