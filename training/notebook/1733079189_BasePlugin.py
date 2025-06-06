from abc import ABC, abstractmethod
import threading
import sys

class BasePlugin(ABC):
    def __init__(self, dt, local_state, autostart=False, needs_power=True, stdout_redirect=True, has_battery=False, battery_level=1.0, dt_draw=0.0):
        self.autostart = autostart
        self.needs_power = needs_power
        self.stdout_redirect = stdout_redirect
        self.running = False
        self.lock = threading.Lock()
        self.stdout = sys.stdout  # Preserve the original stdout
        self.has_battery = has_battery
        self.dt_draw = dt_draw
        
        self._instantaneous_power_queue = Queue()
        if self.stdout_redirect:
            self._redirect_stdout()
        if self.autostart:
            self.start()

    def _redirect_stdout(self):
        """
        Redirect stdout to the provided stream interceptor.
        """
        if self.stream_interceptor:
            sys.stdout = self.stream_interceptor

    def start(self, **kwargs):
        """
        Start the plugin, considering the power state.
        """
        with self.lock:
            if self.needs_power and not self._check_power_state():
                return
            if not self.running:
                self.running = True
                self._start_plugin(**kwargs)
    
    @abstractmethod
    def _start_plugin(self, dt, local_state):
        """
        Plugin-specific start logic.
        """
        pass

    def update(self, dt, local_state):
        """
        Default update behavior; starts the plugin if autostart is True.
        """
        if self.autostart and not self.running:
            self.start()
        # Plugin-specific update logic
        return self._update_plugin(dt, local_state)

    @abstractmethod
    def _update_plugin(self, dt, local_state):
        """
        Plugin-specific update logic.
        """
        pass

    def stop(self):
        """
        Stop the plugin.
        """
        with self.lock:
            if self.running:
                self.running = False
                self._stop_plugin()

    @abstractmethod
    def _stop_plugin(self):
        """
        Plugin-specific stop logic.
        """
        pass

    def _check_power_state(self):
        """
        Check if the plugin should start based on power state.
        Placeholder for actual power state logic.
        """
        # For now, we return True as a placeholder
        return True

    def get_class_name(self):
        """
        Return the class name.
        """
        return self.__class__.__name__
