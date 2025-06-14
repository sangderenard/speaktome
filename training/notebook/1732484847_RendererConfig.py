# renderer/renderer_config.py

from __future__ import annotations

try:
    import pygame
    from OpenGL.GL import *  # noqa: F401,F403
except Exception:
    from AGENTS.tools.headers.header_utils import ENV_SETUP_BOX
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---

class RendererConfig:
    def __init__(self, width=800, height=600, camera_pos=(2.0, 2.0, 1.0), lighting=True):
        self.width = width
        self.height = height
        self.camera_pos = camera_pos
        self.lighting = lighting
        self.setup_gl()

    def setup_gl(self):
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 32)
        glEnable(GL_DEPTH_TEST)
        #glEnable(GL_CULL_FACE)
        if self.lighting:
            glEnable(GL_LIGHTING)
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.01, 0.01, 0.01, 1.0])  # Minimal ambient lighting

            #glEnable(GL_LIGHT0)
            glEnable(GL_LIGHT1)
            # Main Light (bright diagonal light at 45° overhead)
            glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.0, 1.0, 0.0, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.0, 1.0, 0.0, 1.0])

            # Secondary Light (forward light, narrow and quick fall-off)
            glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.0, 0.0, 1.0, 1.0])  # Higher intensity
            glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 1.0, 1.0])
            #glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 15.0)  # Narrow beam (15° cutoff)
            #glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 50.0)  # Quick fall-off
            glEnable(GL_COLOR_MATERIAL)

        glEnable(GL_COLOR_MATERIAL)
        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)