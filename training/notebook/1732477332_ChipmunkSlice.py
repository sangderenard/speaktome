# chipmunk_slice.py

import pymunk
import logging

# ========================================
# ChipmunkSlice Class (using cricket.py settings with gravity (0,0))
# ========================================

class ChipmunkSlice:
    def __init__(self):
        # Use the environment settings from cricket.py, with gravity (0,0)
        self.width = 800
        self.height = 800
        self.gravity = 0  # Gravity magnitude set to 0
        self.color = (0, 0, 1, 0.5)  # Blue with alpha
        self.pymunk_width = 40
        self.pymunk_height = 40

        # Define Pymunk simulation bounds
        self.pymunk_bounds = (
            (-self.pymunk_width / 2 + self.width / 2, self.pymunk_width / 2 + self.width / 2),
            (-self.pymunk_height / 2 + self.height / 2, self.pymunk_height / 2 + self.height / 2),
        )

        # Initialize Pymunk Space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity

        # Add walls and dynamic body
        self.create_walls()
        self.body = self.create_dynamic_body()

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

    def simulate(self, state_dict, dt):
        """
        Simulate the ChipmunkSlice for a given state and time step.

        Args:
            state_dict (dict): Dictionary containing state variables.
            dt (float): Time step.

        Returns:
            dict: Projected state after dt time.
        """
        # Map the provided position and velocity to the Pymunk body
        pos_x, pos_y = (tuple(state_dict.get('position', (0, 0))) + (0, 0))[:2]

        vel_x, vel_y = (tuple(state_dict.get('velocity', (0, 0))) + (0, 0))[:2]

        # Convert to Pymunk coordinates
        body_position = pymunk.Vec2d(pos_x, pos_y)
        body_velocity = pymunk.Vec2d(vel_x, vel_y)

        self.body.position = body_position
        self.body.velocity = body_velocity
        self.body.mass = state_dict['mass']

        # Simulate for dt
        self.space.step(dt)

        # Get projected state
        projected_state = {
            'position': [self.body.position.x, self.body.position.y],
            'velocity': [self.body.velocity.x, self.body.velocity.y],
        }
        return projected_state