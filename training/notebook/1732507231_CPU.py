import logging
from Primitives.device import Device
import torch
import numpy as np
from Devices.computer_screen import ComputerScreen
from Networking.pillbug_network import PillbugNetwork

class CPU(Device):
    def __init__(self, object_id, position, orientation, thread_manager,
                 heat_generation=50.0, energy_consumption=100.0, dvibus=None):
        super().__init__(object_id, position, orientation, thread_manager,
                         heat_generation, energy_consumption)
        self.dvibus = dvibus
        self.network = PillbugNetwork(
            num_nodes=3,
            num_features=3,
            num_subfeatures=1,
            temperature=0.0,
            radiation_coefficient=0.1
        )
        logging.info(f"CPU '{self.object_id}' initialized.")

    def start(self):
        """
        Starts the ComputerScreen and initializes the network.
        """
        self.screen.run()
        logging.info("ComputerScreen started.")

    def stop(self):
        """
        Stops the ComputerScreen.
        """
        self.screen.stop()
        logging.info("ComputerScreen stopped.")

    def deliver_to_screen(self, data):
        """
        Delivers data to the ComputerScreen.
        
        Args:
            data (dict): Data to deliver to the screen buffers.
        """
        self.screen.deliver(data)
        logging.info(f"Delivered data to screen: {data}")

    def examine_screen(self, params=None):
        """
        Examines the contents of the screen's managed buffer.
        
        Args:
            params (dict, optional): Parameters to filter the examined data.
        
        Returns:
            dict: The contents of the screen's managed buffer.
        """
        buffer_contents = self.screen.examine(params)
        logging.info(f"Examined screen buffer: {buffer_contents}")
        return buffer_contents

    def clear_screen_buffers(self, buffer_type=None, fill_data=None):
        """
        Clears the screen buffers and optionally fills them with data.
        
        Args:
            buffer_type (str, optional): The buffer to clear ("vertices", "textures", or None for all).
            fill_data (Any, optional): Data to fill the cleared buffers with.
        """
        self.screen.clear(buffer_type, fill_data)
        logging.info(f"Cleared screen buffers: {buffer_type or 'all'}")

    def update(self, dt):
        """
        Executes one clock cycle.
        """
        # Update the network
        super().update(dt)
        state_change = self.network.process_iteration(dt)
        logging.info(f"CPU '{self.object_id}' ran one cycle. State change: {state_change}")
        # Additional CPU-specific logic can be added here
        self.send_data(state_change)
        return state_change
    # In CPU class
    def send_data(self, data, rendering_config=None):
        # Send data via DVIBus
        if self.dvibus:
            if rendering_config is None:
                self.dvibus.send_data(self, data)
            else:
                self.dvibus.send_data(self, rendering_config.pipeline(data))

    def receive_data(self, data):
        # Handle received data
        logging.info(f"CPU '{self.object_id}' received data: {data}")


# In cpu.py

def generate_output_data(self):
    """
    Generates data to be sent to other devices.
    """
    # For example, produce some data based on the current state
    data = {
        "vertices": [np.array([100, 100]), np.array([200, 200])]
    }
    return data
