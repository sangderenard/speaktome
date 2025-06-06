import numpy as np
import pymunk
from OpenGL.GL import *
from OpenGL.GLU import *


class CricketSimulation:
    def __init__(self, width, height, gravity, color, pymunk_width, pymunk_height):
        """
        Initialize the simulation environment.
        :param width: Screen width in pixels
        :param height: Screen height in pixels
        :param gravity: Gravity magnitude
        :param color: Rendering color (R, G, B, A)
        :param pymunk_width: Width of the Pymunk simulation space
        :param pymunk_height: Height of the Pymunk simulation space
        """
        self.width = width
        self.height = height
        self.gravity = gravity
        self.color = color
        self.pymunk_width = pymunk_width
        self.pymunk_height = pymunk_height

        # Define Pymunk simulation bounds
        self.pymunk_bounds = (
            (-pymunk_width / 2 + width / 2, pymunk_width / 2 + width / 2),
            (-pymunk_height / 2 + height / 2, pymunk_height / 2 + height / 2),
        )

        # Initialize Pymunk Space
        self.space = pymunk.Space()
        self.space.gravity = (0, -self.gravity)

        # Add walls and dynamic body
        self.create_walls()
        self.body = self.create_dynamic_body()

    def pymunk_to_opengl(self, pos):
        """
        Convert Pymunk coordinates to OpenGL screen coordinates.
        :param pos: (x, y) position in Pymunk coordinates
        :return: (x, y) position in OpenGL coordinates
        """
        (left, right), (bottom, top) = self.pymunk_bounds
        norm_x = (pos[0] - left) / (right - left)
        norm_y = (pos[1] - bottom) / (top - bottom)
        ogl_x = norm_x * self.width
        ogl_y = (1 - norm_y) * self.height  # Invert Y-axis for OpenGL
        return ogl_x, ogl_y

    def create_walls(self):
        """
        Create rigid, smooth walls around the simulation.
        """
        thickness = 10
        (left, right), (bottom, top) = self.pymunk_bounds
        walls = [
            pymunk.Segment(self.space.static_body, (left, bottom), (right, bottom), thickness),
            pymunk.Segment(self.space.static_body, (right, bottom), (right, top), thickness),
            pymunk.Segment(self.space.static_body, (right, top), (left, top), thickness),
            pymunk.Segment(self.space.static_body, (left, top), (left, bottom), thickness),
        ]
        for wall in walls:
            wall.elasticity = 1.0  # Perfectly smooth and rigid
            wall.friction = 0.0
        self.space.add(*walls)

    def create_dynamic_body(self):
        """
        Create a single dynamic body in the simulation.
        :return: Pymunk body object
        """
        mass = 1
        radius = 0.1
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)

        # Center the dynamic body in Pymunk space
        body.position = (
            (self.pymunk_bounds[0][0] + self.pymunk_bounds[0][1]) / 2,
            (self.pymunk_bounds[1][0] + self.pymunk_bounds[1][1]) / 2,
        )
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 1.0
        shape.friction = 0.0
        self.space.add(body, shape)
        return body

    def update(self, dt):
        """
        Update the physics simulation.
        :param dt: Time step (delta time)
        """
        self.space.step(dt)

    def draw(self, buffer):
        """
        Render the simulation state onto the provided OpenGL buffer.
        :param buffer: OpenGL buffer to render to
        """
        glBindFramebuffer(GL_FRAMEBUFFER, buffer)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glColor4f(*self.color)

        # Render walls
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                p1 = self.pymunk_to_opengl(shape.a)
                p2 = self.pymunk_to_opengl(shape.b)
                glBegin(GL_LINES)
                glVertex2f(*p1)
                glVertex2f(*p2)
                glEnd()

        # Render dynamic body
        pos = self.pymunk_to_opengl(self.body.position)
        glBegin(GL_QUADS)
        glVertex2f(pos[0] - 5, pos[1] - 5)
        glVertex2f(pos[0] + 5, pos[1] - 5)
        glVertex2f(pos[0] + 5, pos[1] + 5)
        glVertex2f(pos[0] - 5, pos[1] + 5)
        glEnd()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)


import pygame
from OpenGL.GL import *

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 800), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Cricket Simulation Test")

# Initialize OpenGL
glViewport(0, 0, 800, 800)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0, 800, 800, 0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glClearColor(1, 1, 1, 1)

# Instantiate the CricketSimulation
cricket = CricketSimulation(
    width=800,
    height=800,
    gravity=500,
    color=(0, 0, 1, 0.5),  # Blue with alpha
    pymunk_width=40,
    pymunk_height=40,
)

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the simulation
    cricket.update(1 / 60)

    # Render to the screen
    cricket.draw(0)  # Default framebuffer
    pygame.display.flip()

    clock.tick(60)

pygame.quit()
