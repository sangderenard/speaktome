import logging
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *
import importlib.util
import time

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")


class GameWindowManager:
    def __init__(self, width=800, height=600, plugin_paths=None):
        """
        Initialize the game window manager with multiple plugins.

        Args:
            width (int): Width of the game window.
            height (int): Height of the game window.
            plugin_paths (list): List of plugin paths to load.
        """
        self.width = width
        self.height = height
        self.plugin_paths = plugin_paths or []
        self.plugins = {}
        self.local_hash = {"fft_in": None, "fft_out": None, "vbo_in": None}
        self.running = True
        self.clock = pygame.time.Clock()
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Update every 100ms

        # Initialize Pygame
        self._init_pygame()

        # Initialize OpenGL
        self._init_opengl()

        # Load plugins
        self._load_plugins()

    def _init_pygame(self):
        """
        Initialize the Pygame window.
        """
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Game Window Manager")
        logging.info("Pygame initialized.")

    def _init_opengl(self):
        """
        Initialize OpenGL settings.
        """
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5.0)
        logging.info("OpenGL initialized.")

    def _load_plugins(self):
        """
        Dynamically load plugins from the specified paths.
        """
        for path in self.plugin_paths:
            spec = importlib.util.spec_from_file_location("plugin", path)
            if not spec:
                logging.error(f"Failed to find plugin at {path}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Identify the class name as the plugin name
            plugin_name = getattr(module, "__name__", f"Plugin_{len(self.plugins)}")

            # Assume a `main` class exists
            if hasattr(module, plugin_name):
                plugin_class = getattr(module, plugin_name)
                self.plugins[plugin_name] = plugin_class()
                logging.info(f"Plugin '{plugin_name}' loaded and initialized.")
            else:
                logging.error(f"Plugin at {path} does not define a recognizable class.")

    def update_plugins(self):
        """
        Update all plugins and manage local state hashing.
        """
        for name, plugin in self.plugins.items():
            if hasattr(plugin, "update"):
                local_state = self.local_hash.get("fft_in", None)
                result = plugin.update(self.update_interval, local_state)

                # Handle plugin output and update local state
                if isinstance(result, dict):
                    self.local_hash.update(result)
                    logging.debug(f"Plugin '{name}' updated local hash: {result}")

    def render(self):
        """
        Render the current scene with data from the local VBO hash.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        vbo_data = self.local_hash.get("vbo_in", None)
        if vbo_data:
            # Example rendering logic (expand as needed for structured VBOs)
            frequencies = vbo_data.get("frequencies", [])
            magnitudes = vbo_data.get("magnitudes", [])
            glBegin(GL_POINTS)
            for freq, mag in zip(frequencies, magnitudes):
                glColor3f(mag, 0.0, 0.0)  # Intensity represented by red
                glVertex3f(freq, mag, 0.0)  # Simplified point rendering
            glEnd()
        pygame.display.flip()

    def handle_events(self):
        """
        Handle Pygame events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logging.info("Quitting game...")

    def run(self):
        """
        Main game loop.
        """
        logging.info("Starting game loop.")
        try:
            while self.running:
                self.handle_events()

                # Update plugins at fixed intervals
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    self.update_plugins()
                    self.last_update_time = current_time

                # Render the scene
                self.render()

                # Cap the frame rate
                self.clock.tick(60)
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Cleanup resources and stop all plugins.
        """
        for name, plugin in self.plugins.items():
            if hasattr(plugin, "stop"):
                plugin.stop()
                logging.info(f"Plugin '{name}' stopped.")
        pygame.quit()
        logging.info("Cleaned up resources and exited.")


if __name__ == "__main__":
    # Define plugin paths relative to the script's directory
    plugin_paths = ["boombox.py", "fftvisualizer.py"]
    manager = GameWindowManager(plugin_paths=plugin_paths)
    manager.run()
