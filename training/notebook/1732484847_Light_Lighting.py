# renderer/lighting.py

import numpy as np
from OpenGL.GL import *

MAX_LIGHTS = 10

class Light:
    def __init__(self, position, direction, color):
        self.position = np.array(position, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)


class Lighting:
    def __init__(self, max_lights=MAX_LIGHTS):
        self.max_lights = max_lights
        self.lights = []
        self.num_lights = 0

    def add_light(self, position, direction, color):
        if self.num_lights < self.max_lights:
            self.lights.append(Light(position, direction, color))
            self.num_lights += 1
        else:
            print("Max number of lights reached.")

    def update_light(self, index, position=None, direction=None, color=None):
        if 0 <= index < self.num_lights:
            if position is not None:
                self.lights[index].position = np.array(position, dtype=np.float32)
            if direction is not None:
                self.lights[index].direction = np.array(direction, dtype=np.float32)
            if color is not None:
                self.lights[index].color = np.array(color, dtype=np.float32)

    def set_light_uniforms(self, shader):
        # Prepare arrays
        positions = np.zeros((self.max_lights, 3), dtype=np.float32)
        directions = np.zeros((self.max_lights, 3), dtype=np.float32)
        colors = np.zeros((self.max_lights, 4), dtype=np.float32)

        for i, light in enumerate(self.lights):
            positions[i] = light.position
            directions[i] = light.direction
            colors[i] = light.color

        # Get uniform locations
        numLights_loc = glGetUniformLocation(shader, 'numLights')
        positions_loc = glGetUniformLocation(shader, 'lights[0].position')
        directions_loc = glGetUniformLocation(shader, 'lights[0].direction')
        colors_loc = glGetUniformLocation(shader, 'lights[0].color')

        # Set uniforms
        glUniform1i(numLights_loc, self.num_lights)

        for i in range(self.num_lights):
            glUniform3f(glGetUniformLocation(shader, f'lights[{i}].position'), *positions[i])
            glUniform3f(glGetUniformLocation(shader, f'lights[{i}].direction'), *directions[i])
            glUniform4f(glGetUniformLocation(shader, f'lights[{i}].color'), *colors[i])

    