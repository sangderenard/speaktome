from abc import ABC, abstractmethod
import threading
import sys
from queue import Queue

class BasePlugin(ABC):
    def __init__(self, dt, local_state, stream_interceptor=None, autostart=False, needs_power=True, has_battery=False, battery_level=1.0, dt_draw=0.0):
        """
        Base class for plugins with shared functionality.

        Args:
            dt (float): Update interval.
            local_state (dict): Shared state hash.
            stream_interceptor: A stream interceptor for stdout redirection.
            autostart (bool): Whether the plugin starts automatically.
            needs_power (bool): If the plugin requires power to run.
            has_battery (bool): If the plugin has a battery.
            battery_level (float): Initial battery level.
            dt_draw (float): Power draw per update cycle.
        """
        self.autostart = autostart
        self.needs_power = needs_power
        self.has_battery = has_battery
        self.battery_level = battery_level
        self.dt_draw = dt_draw
        self.running = False
        self.lock = threading.Lock()
        self.stream_interceptor = stream_interceptor  # Optional stream redirection
        self._instantaneous_power_queue = Queue()

        if self.stream_interceptor:
            self._redirect_stdout()

        if self.autostart:
            self.start(dt=dt, local_state=local_state)

    def _redirect_stdout(self):
        """
        Redirect stdout to the provided stream interceptor.
        """
        if self.stream_interceptor:
            sys.stdout = self.stream_interceptor

    def start(self, dt, local_state):
        """
        Start the plugin, considering the power state.
        """
        with self.lock:
            if self.needs_power and not self._check_power_state():
                return
            if not self.running:
                self.running = True
                self._start_plugin(dt, local_state)

    @abstractmethod
    def _start_plugin(self, dt, local_state):
        """
        Plugin-specific start logic to be implemented by subclasses.
        """
        pass

    def update(self, dt, local_state):
        """
        Default update behavior; starts the plugin if autostart is True.
        """
        if self.autostart and not self.running:
            self.start(dt=dt, local_state=local_state)
        return self._update_plugin(dt, local_state)

    @abstractmethod
    def _update_plugin(self, dt, local_state):
        """
        Plugin-specific update logic to be implemented by subclasses.
        """
        pass

    def stop(self, dt, local_state):
        """
        Stop the plugin.
        """
        with self.lock:
            if self.running:
                self.running = False
                self._stop_plugin(dt, local_state)

    @abstractmethod
    def _stop_plugin(self, dt, local_state):
        """
        Plugin-specific stop logic to be implemented by subclasses.
        """
        pass

    def _check_power_state(self):
        """
        Check if the plugin should start based on power state.
        """
        # Placeholder for actual power state logic; for now, assume power is available.
        return True

    def get_class_name(self):
        """
        Return the class name.
        """
        return self.__class__.__name__
