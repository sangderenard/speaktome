import logging
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *
import importlib.util
import time
from streaminterceptor import StreamInterceptor
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

class GameWindowManager:
    def __init__(self, width=800, height=600, plugin_paths=None):
        
        self.width = width
        self.height = height
        self.plugin_paths = plugin_paths or []
        self.plugins = {}
        self.local_hash = {"fft_in": None, "fft_out": None, "fft_wave_in": None, "vbo_in": None, "txt_in": []}
        self._redirect_stdout()
        self.running = True
        self.clock = pygame.time.Clock()
        self.last_update_time = time.time()
        self.update_interval = 0.1
        self._init_pygame()
        self._init_opengl()
        self._load_plugins()
        

    def _redirect_stdout(self):
        """
        Redirect stdout to a StreamInterceptor instance.
        """
        sys.stdout = StreamInterceptor(sys.stdout, self.local_hash, key="txt_in")
        logging.info("StreamInterceptor initialized and stdout redirected.")

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Game Window Manager")
        logging.info("Pygame initialized.")

    def _init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5.0)
        logging.info("OpenGL initialized.")

    def _load_plugins(self):
        for path in self.plugin_paths:
            spec = importlib.util.spec_from_file_location("plugin", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugin = module.Plugin()
            self.plugins[plugin.get_class_name()] = plugin
            logging.info(f"Loaded plugin: {plugin.get_class_name()}")

    def update_plugins(self):
        for name, plugin in self.plugins.items():
            result = plugin.update(self.update_interval, self.local_hash)
            if result:
                self.local_hash.update(result)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        vbo_data = self.local_hash.get("vbo_in", None)
        if vbo_data:
            frequencies = vbo_data.get("frequencies", [])
            magnitudes = vbo_data.get("magnitudes", [])
            glBegin(GL_POINTS)
            for freq, mag in zip(frequencies, magnitudes):
                glColor3f(mag, 0.0, 0.0)
                glVertex3f(freq, mag, 0.0)
            glEnd()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logging.info("Quitting game...")

    def run(self):
        logging.info("Starting game loop.")
        try:
            while self.running:
                self.handle_events()
                if time.time() - self.last_update_time >= self.update_interval:
                    self.update_plugins()
                    self.last_update_time = time.time()
                self.render()
                self.clock.tick(60)
        finally:
            self.cleanup()

    def cleanup(self):
        for plugin in self.plugins.values():
            plugin.stop()
        pygame.quit()
        logging.info("Cleaned up resources and exited.")


if __name__ == "__main__":
    plugin_paths = ["boombox.py", "console.py"]
    manager = GameWindowManager(plugin_paths=plugin_paths)
    manager.run()
