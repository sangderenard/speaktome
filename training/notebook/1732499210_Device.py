# device.py

from .physicalobject import PhysicalObject
import logging

class Device(PhysicalObject):
    def __init__(self, object_id, position, orientation, thread_manager,
                 heat_generation=0.0, energy_consumption=0.0):
        super().__init__(object_id, position, orientation, thread_manager)
        self.heat_generation = heat_generation  # Joules per second
        self.energy_consumption = energy_consumption  # Joules per second
        self.temperature = 293.15  # Initial temperature in Kelvin (20°C)
        self.power_state = True  # Device is powered on by default
        logging.info(f"Device '{self.object_id}' initialized with heat generation {self.heat_generation} J/s "
                     f"and energy consumption {self.energy_consumption} J/s.")

    def update(self, dt):
        """
        Updates the device's state over the given time step.
        """
        super().update(dt)
        if self.power_state:
            # Simple proportional temperature change
            self.temperature += self.heat_generation * dt
            # Energy consumption tracking
            # For simplicity, we can just log it for now
            logging.info(f"Device '{self.object_id}' consumed {self.energy_consumption * dt} Joules of energy.")

