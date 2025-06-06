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
        self.local_hash = {"fft_in": None, "fft_out": None, "fft_wave_in": None, "video_buffer_in": [], "txt_in": []}
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
        self.stream_interceptor = StreamInterceptor(sys.stdout, self.local_hash, key="txt_in")
        sys.stdout = self.stream_interceptor
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
        plugin_names = []
        for path in self.plugin_paths:
            spec = importlib.util.spec_from_file_location("plugin", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugin = module.Plugin(dt=self.update_interval, local_state=self.local_hash, stream_interceptor=self.stream_interceptor)
            plugin_name = plugin.get_class_name()
            self.plugins[plugin_name] = plugin
            plugin_names.append(plugin_name)
            logging.info(f"Loaded plugin: {plugin_name}")
        # Store the list of plugin names in local_hash
        self.local_hash['plugins'] = plugin_names

    def update_plugins(self):
        # Check for plugin actions
        plugin_actions = self.local_hash.get('plugin_actions', [])
        for action in plugin_actions:
            if action['action'] == 'start':
                plugin_name = action['plugin_name']
                if plugin_name in self.plugins:
                    plugin = self.plugins[plugin_name]
                    plugin.start(dt=self.update_interval, local_state=self.local_hash)
                    logging.info(f"Started plugin: {plugin_name}")
        # Clear the plugin actions after processing
        self.local_hash['plugin_actions'] = []
        # Update plugins
        for name, plugin in self.plugins.items():
            result = plugin.update(self.update_interval, self.local_hash)
            if result:
                self.local_hash.update(result)

    def render(self):
        """
        Render buffers based on the hierarchical data structure in `video_buffer_in`.

        Structure of `video_buffer_in`:
        [
            [  # Array from one source
                {"0,0,0": [buffer_metadata1, buffer_metadata2]},  # Spatial hash to metadata
                {"1,0,0": [buffer_metadata3]},                    # Another spatial hash
            ],
            [  # Array from another source
                {"0,0,0": [buffer_metadata4]},                    # Spatial hash reused
            ]
        ]

        Each buffer_metadata includes:
        - 'buffer': OpenGL buffer handle.
        - 'buffer_type': Type of buffer ('VBO', 'TEXTURE', etc.).
        - 'intended_time': Time when the buffer should appear (-1 for immediate).
        - 'duration': Duration the buffer should appear (-1 for indefinite).
        - 'location': (x, y) normalized screen coordinates (0 to 1).
        - 'size': (width, height) normalized dimensions.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        video_buffer_in = self.local_hash.get("video_buffer_in", [])
        current_time = time.time()

        for source_package in video_buffer_in:
            for spatial_data in source_package:
                for spatial_hash, buffer_list in spatial_data.items():
                    for buffer_data in buffer_list:
                        # Extract metadata
                        buffer_handle = buffer_data.get("buffer")
                        buffer_type = buffer_data.get("buffer_type", "UNKNOWN")
                        intended_time = buffer_data.get("intended_time", -1)
                        duration = buffer_data.get("duration", -1)
                        location = buffer_data.get("location", (0.0, 0.0))
                        size = buffer_data.get("size", (1.0, 1.0))

                        # Timing checks
                        if intended_time != -1 and current_time < intended_time:
                            continue
                        if duration != -1 and intended_time != -1 and current_time > intended_time + duration:
                            continue

                        # Render based on buffer type
                        if buffer_handle is not None:
                            if buffer_type == "VBO":
                                self._render_vbo(buffer_handle, location, size)
                            elif buffer_type == "TEXTURE":
                                self._render_texture(buffer_handle, location, size)
                            else:
                                logging.warning(f"Unknown buffer type: {buffer_type}")

        pygame.display.flip()


    def _render_vbo(self, vbo, location, size):
        """
        Render a VBO (Vertex Buffer Object).

        Args:
            vbo (int): OpenGL buffer handle for vertex data.
            location (tuple): (x, y) normalized screen coordinates.
            size (tuple): (width, height) normalized dimensions.
        """
        glPushMatrix()
        glTranslatef(location[0] * 2 - 1, location[1] * 2 - 1, 0)  # Map to OpenGL normalized coordinates
        glScalef(size[0], size[1], 1.0)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, None)  # Assuming 2D vertex data
        glDrawArrays(GL_POINTS, 0, 4)  # Render as points (or adjust as needed)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glPopMatrix()


    def _render_texture(self, texture, location, size):
        """
        Render a texture.

        Args:
            texture (int): OpenGL texture handle.
            location (tuple): (x, y) normalized screen coordinates.
            size (tuple): (width, height) normalized dimensions.
        """
        glPushMatrix()
        glTranslatef(location[0] * 2 - 1, location[1] * 2 - 1, 0)  # Map to OpenGL normalized coordinates
        glScalef(size[0], size[1], 1.0)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

        glPopMatrix()


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
            plugin.stop(dt=self.update_interval, local_state=self.local_hash)
        pygame.quit()
        logging.info("Cleaned up resources and exited.")


if __name__ == "__main__":
    plugin_paths = ["boombox.py", "console.py"]
    manager = GameWindowManager(plugin_paths=plugin_paths)
    manager.run()
