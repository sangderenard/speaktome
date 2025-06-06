import numpy as np
from enum import Enum
from typing import Dict, Any, Optional


class CoordinateSystemType(Enum):
    RECTANGULAR = 'rectangular'
    SPHERICAL = 'spherical'
    CYLINDRICAL = 'cylindrical'
    # Add more as needed


class ControlDomain(Enum):
    CONTINUOUS = 'continuous'
    STEPPER = 'stepper'
    # Add more as needed


class AxisMeaning(Enum):
    POSITION = 'position'
    CHANGE = 'change'
    HOVER = 'hover'
    AZIMUTH = 'azimuth'
    ELEVATION = 'elevation'
    RADIUS = 'radius'
    TRANSLATION = 'translation'
    # Add more as needed


class ControlProfile:
    """
    Represents a control profile for either input or output, defining the axes and their properties.
    """

    def __init__(self, axes: Dict[str, Dict[str, Any]]):
        """
        Initializes the ControlProfile.

        :param axes: A dictionary where keys are axis names and values are dictionaries with 'domain' and 'meaning'.
        """
        self.axes = {}
        for axis_name, properties in axes.items():
            domain = properties.get('domain')
            meaning = properties.get('meaning')
            if isinstance(domain, str):
                domain = ControlDomain(domain)
            if isinstance(meaning, str):
                meaning = AxisMeaning(meaning)
            self.axes[axis_name] = {
                'domain': domain,
                'meaning': meaning
            }

    def get_axis_properties(self, axis_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the properties of a given axis.

        :param axis_name: Name of the axis.
        :return: Dictionary of properties or None if axis does not exist.
        """
        return self.axes.get(axis_name)


class AutoProjector:
    """
    The `AutoProjector` class is a simulation tool designed to represent the dynamics of a telescoping mechanism 
    with bidirectional control and joint manipulation. The class models the interactions and transformations 
    between two sides of the system (Side A and Side B), simulating realistic mass properties, coordinate systems, 
    and motion constraints. 

    This class abstracts away universal reference frames, focusing instead on the relative transformations between 
    the components it models. The system emphasizes flexibility, enabling simulation of machines with varying degrees 
    of constraints and freedom.

    Key Features and Components:
    ----------------------------------
    1. **Bidirectional Input/Output Relationships**:
       - The input joint and output joint are symmetric in terms of their control and output capacity.
       - Either joint can drive the other, with transformations computed relative to their respective reference frames.
       - Metadata side channels allow storage of auxiliary data about transformations or control processes.

    2. **Coordinate Systems**:
       - The input and output systems can use different coordinate representations (e.g., rectangular, spherical).
       - These are tracked in `self.coordinate_systems` for flexible interconversion or transformations.

    3. **Mass and Inertia Properties**:
       - Each side of the mechanism has an independent mass property (`mass_side_a`, `mass_side_b`).
       - The `calculate_moment_of_inertia` function computes the combined rotational inertia for the system, useful for 
         simulating rotational dynamics.

    4. **Telescoping Mechanism**:
       - The system represents a telescoping boom with adjustable length (`min_length`, `max_length`, `current_length`).
       - The boom can dynamically adjust its length within defined bounds, simulating expansion or contraction.

    5. **Control Profiles**:
       - Input and output control profiles define the nature of interaction along different axes:
         - Input profiles (`input_translation_profile`) define axes for position, change, and hover effects.
         - Output profiles (`output_translation_profile`) define spherical orientation (azimuth, elevation) and translocation (radius).
       - Each axis has attributes specifying its domain (e.g., continuous, stepper) and meaning (e.g., position, change).

    6. **Axis Definitions**:
       - **Position**: Specifies relative normalized positions (e.g., move the output joint to match the input's relative extent).
       - **Change**: Simulates dynamic actuation with instantaneous acceleration but time-dependent displacement.
       - **Hover**: Smoothly maintains relative orientation while applying minor drift for stabilization.

    7. **Machine Simulation**:
       - By setting specific constants (e.g., infinite mass on Side A or zero mass on Side B), the system can model specialized
         behaviors like a warp (infinite inertia) or weft (negligible mass).
       - Side A typically represents the "shell" or base, while Side B represents the "piston" or actuating part.

    8. **Flexibility in Design**:
       - The system supports varying degrees of constraints by dynamically adjusting inputs, outputs, and mass properties.
       - Coordinate system transformations allow seamless simulation of machines operating in disparate reference frames.

    Examples of Use Cases:
    ----------------------
    - Robotics: Simulating a robotic arm or telescoping actuator with fine-grained control.
    - Mechanical Engineering: Analyzing dynamics of telescoping components with varying mass and inertia.
    - Physics Education: Demonstrating principles of relative motion, torque, and coordinate system transformations.
    """

    def __init__(self,
                 mass_side_a: float = 1.0,
                 mass_side_b: float = 1.0,
                 min_length: float = 1.0,
                 max_length: float = 10.0,
                 current_length: float = 5.0,
                 input_profile_axes: Dict[str, Dict[str, Any]] = None,
                 output_profile_axes: Dict[str, Dict[str, Any]] = None,
                 coordinate_systems: Dict[str, CoordinateSystemType] = None):
        """
        Initializes the AutoProjector with mass properties, telescoping mechanism parameters, control profiles,
        and coordinate systems.

        :param mass_side_a: Mass of Side A (shell/base).
        :param mass_side_b: Mass of Side B (piston/actuating part).
        :param min_length: Minimum length of the telescoping boom.
        :param max_length: Maximum length of the telescoping boom.
        :param current_length: Current length of the telescoping boom.
        :param input_profile_axes: Dictionary defining the input control profile axes.
        :param output_profile_axes: Dictionary defining the output control profile axes.
        :param coordinate_systems: Dictionary defining the coordinate systems for input and output.
        """
        # Mass properties
        self.mass_side_a = mass_side_a
        self.mass_side_b = mass_side_b

        # Telescoping mechanism
        self.min_length = min_length
        self.max_length = max_length
        self.current_length = current_length
        self.length_velocity = 0.0  # For dynamic adjustments

        # Control profiles
        if input_profile_axes is None:
            input_profile_axes = {
                'position': {'domain': 'continuous', 'meaning': 'position'},
                'change': {'domain': 'continuous', 'meaning': 'change'},
                'hover': {'domain': 'continuous', 'meaning': 'hover'}
            }
        if output_profile_axes is None:
            output_profile_axes = {
                'azimuth': {'domain': 'continuous', 'meaning': 'azimuth'},
                'elevation': {'domain': 'continuous', 'meaning': 'elevation'},
                'radius': {'domain': 'continuous', 'meaning': 'radius'}
            }

        self.input_translation_profile = ControlProfile(input_profile_axes)
        self.output_translation_profile = ControlProfile(output_profile_axes)

        # Coordinate systems
        if coordinate_systems is None:
            coordinate_systems = {
                'input': CoordinateSystemType.RECTANGULAR,
                'output': CoordinateSystemType.SPHERICAL
            }
        self.coordinate_systems = coordinate_systems

        # Metadata side channels
        self.metadata_input = {}
        self.metadata_output = {}

        # Initialize transformation matrices or parameters
        self.input_transformation = np.identity(4)  # 4x4 transformation matrix
        self.output_transformation = np.identity(4)

    def calculate_moment_of_inertia(self) -> float:
        """
        Calculates the combined moment of inertia for the system assuming cylindrical symmetry.

        :return: Combined moment of inertia.
        """
        # For simplicity, assuming both masses are point masses at the end of the telescoping boom
        # I = m * r^2 for each mass
        # Total I = I_a + I_b
        radius_a = self.current_length / 2  # Arbitrary choice
        radius_b = self.current_length / 2
        inertia_a = self.mass_side_a * (radius_a ** 2)
        inertia_b = self.mass_side_b * (radius_b ** 2)
        total_inertia = inertia_a + inertia_b
        self.metadata_input['moment_of_inertia'] = total_inertia
        return total_inertia

    def adjust_length(self, delta_length: float):
        """
        Adjusts the current length of the telescoping boom by delta_length within the defined bounds.

        :param delta_length: Change in length to apply.
        """
        new_length = self.current_length + delta_length
        new_length = max(self.min_length, min(self.max_length, new_length))
        self.length_velocity = delta_length  # Simplistic model: velocity proportional to delta
        self.current_length = new_length
        self.metadata_output['length_adjustment'] = self.current_length

    def set_control_input(self, axis: str, value: float):
        """
        Sets the control input for a specific axis.

        :param axis: Name of the axis.
        :param value: Value to set.
        """
        if axis in self.input_translation_profile.axes:
            # Apply control based on axis meaning
            meaning = self.input_translation_profile.get_axis_properties(axis)['meaning']
            if meaning == AxisMeaning.POSITION:
                self.apply_position_control(value)
            elif meaning == AxisMeaning.CHANGE:
                self.apply_change_control(value)
            elif meaning == AxisMeaning.HOVER:
                self.apply_hover_control(value)
            # Add more controls as needed
            self.metadata_input['control_input_' + axis] = value
        else:
            raise ValueError(f"Input axis '{axis}' not defined in the control profile.")

    def set_control_output(self, axis: str, value: float):
        """
        Sets the control output for a specific axis.

        :param axis: Name of the axis.
        :param value: Value to set.
        """
        if axis in self.output_translation_profile.axes:
            # Apply control based on axis meaning
            meaning = self.output_translation_profile.get_axis_properties(axis)['meaning']
            if meaning in [AxisMeaning.AZIMUTH, AxisMeaning.ELEVATION, AxisMeaning.RADIUS]:
                self.apply_spherical_control(axis, value)
            elif meaning == AxisMeaning.TRANSLATION:
                self.apply_translation_control(value)
            # Add more controls as needed
            self.metadata_output['control_output_' + axis] = value
        else:
            raise ValueError(f"Output axis '{axis}' not defined in the control profile.")

    def apply_position_control(self, value: float):
        """
        Applies position control to adjust the telescoping boom to a specific position.

        :param value: Normalized position value (0.0 to 1.0).
        """
        target_length = self.min_length + value * (self.max_length - self.min_length)
        delta = target_length - self.current_length
        self.adjust_length(delta)

    def apply_change_control(self, value: float):
        """
        Applies change control to dynamically actuate the telescoping boom.

        :param value: Change value representing acceleration or force.
        """
        # Simplistic model: change directly affects length velocity
        self.length_velocity += value
        delta = self.length_velocity * 0.1  # Assuming a time step
        self.adjust_length(delta)

    def apply_hover_control(self, value: float):
        """
        Applies hover control to maintain relative orientation with minor drift.

        :param value: Drift value to apply for stabilization.
        """
        # Simplistic model: Apply a small adjustment to current length
        delta = value * 0.01  # Minor adjustment
        self.adjust_length(delta)

    def apply_spherical_control(self, axis: str, value: float):
        """
        Applies spherical coordinate control.

        :param axis: 'azimuth', 'elevation', or 'radius'.
        :param value: Value to set for the axis.
        """
        if axis == 'azimuth':
            self.output_transformation = self.rotate_about_z(value)
        elif axis == 'elevation':
            self.output_transformation = self.rotate_about_y(value)
        elif axis == 'radius':
            self.adjust_length(value - self.current_length)
        else:
            raise ValueError(f"Spherical control axis '{axis}' is not recognized.")

    def apply_translation_control(self, value: float):
        """
        Applies translation control to move the output joint.

        :param value: Translation value to apply.
        """
        translation_vector = np.array([value, 0, 0, 1])  # Simplistic along x-axis
        self.output_transformation[:3, 3] += translation_vector[:3]
        self.metadata_output['translation'] = self.output_transformation[:3, 3]

    def rotate_about_z(self, angle_rad: float) -> np.ndarray:
        """
        Creates a rotation matrix for rotating about the Z-axis.

        :param angle_rad: Rotation angle in radians.
        :return: 4x4 rotation matrix.
        """
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0,     0,      1, 0],
            [0,     0,      0, 1]
        ])
        return rotation @ self.output_transformation

    def rotate_about_y(self, angle_rad: float) -> np.ndarray:
        """
        Creates a rotation matrix for rotating about the Y-axis.

        :param angle_rad: Rotation angle in radians.
        :return: 4x4 rotation matrix.
        """
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation = np.array([
            [cos_a, 0, sin_a, 0],
            [0,     1, 0,     0],
            [-sin_a,0, cos_a, 0],
            [0,     0, 0,     1]
        ])
        return rotation @ self.output_transformation

    def transform_coordinate(self, point: np.ndarray, from_system: str, to_system: str) -> np.ndarray:
        """
        Transforms a point from one coordinate system to another.

        :param point: 3D point as a NumPy array.
        :param from_system: Source coordinate system ('input' or 'output').
        :param to_system: Target coordinate system ('input' or 'output').
        :return: Transformed 3D point as a NumPy array.
        """
        if from_system not in self.coordinate_systems or to_system not in self.coordinate_systems:
            raise ValueError("Invalid coordinate system specified.")

        from_type = self.coordinate_systems[from_system]
        to_type = self.coordinate_systems[to_system]

        # Convert from source to rectangular if not already
        if from_type == CoordinateSystemType.RECTANGULAR:
            rect_point = point
        elif from_type == CoordinateSystemType.SPHERICAL:
            r, theta, phi = point
            rect_point = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
        elif from_type == CoordinateSystemType.CYLINDRICAL:
            r, theta, z = point
            rect_point = np.array([
                r * np.cos(theta),
                r * np.sin(theta),
                z
            ])
        else:
            raise NotImplementedError(f"Transformation from {from_type} is not implemented.")

        # Convert from rectangular to target coordinate system
        if to_type == CoordinateSystemType.RECTANGULAR:
            return rect_point
        elif to_type == CoordinateSystemType.SPHERICAL:
            x, y, z = rect_point
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r) if r != 0 else 0.0
            phi = np.arctan2(y, x)
            return np.array([r, theta, phi])
        elif to_type == CoordinateSystemType.CYLINDRICAL:
            x, y, z = rect_point
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            return np.array([r, theta, z])
        else:
            raise NotImplementedError(f"Transformation to {to_type} is not implemented.")

    def simulate_step(self, time_step: float = 0.1):
        """
        Simulates a single time step, updating the system's state based on current controls and dynamics.

        :param time_step: Duration of the simulation step.
        """
        # Update length based on velocity
        delta_length = self.length_velocity * time_step
        self.adjust_length(delta_length)

        # Apply damping or other dynamic effects if necessary
        damping_factor = 0.95
        self.length_velocity *= damping_factor

        # Update transformations if needed
        # For example, apply rotation based on current state
        # This is a placeholder for more complex dynamics

        # Recalculate moment of inertia
        self.calculate_moment_of_inertia()

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the current state of the AutoProjector.

        :return: Dictionary containing state information.
        """
        state = {
            'mass_side_a': self.mass_side_a,
            'mass_side_b': self.mass_side_b,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'current_length': self.current_length,
            'length_velocity': self.length_velocity,
            'moment_of_inertia': self.calculate_moment_of_inertia(),
            'input_transformation': self.input_transformation,
            'output_transformation': self.output_transformation,
            'metadata_input': self.metadata_input,
            'metadata_output': self.metadata_output
        }
        return state

    def set_mass_side_a(self, mass: float):
        """
        Sets the mass of Side A.

        :param mass: Mass value to set.
        """
        self.mass_side_a = mass
        self.metadata_input['mass_side_a'] = mass

    def set_mass_side_b(self, mass: float):
        """
        Sets the mass of Side B.

        :param mass: Mass value to set.
        """
        self.mass_side_b = mass
        self.metadata_input['mass_side_b'] = mass

    def set_coordinate_system(self, side: str, system_type: CoordinateSystemType):
        """
        Sets the coordinate system for a given side.

        :param side: 'input' or 'output'.
        :param system_type: CoordinateSystemType to set.
        """
        if side not in self.coordinate_systems:
            raise ValueError("Side must be 'input' or 'output'.")
        self.coordinate_systems[side] = system_type
        self.metadata_input['coordinate_system_' + side] = system_type.value


import pygame
import numpy as np

# Constants for the Pygame window
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def project_to_2d(point, focal_length=500):
    """
    Projects a 3D point to 2D for display in Pygame.
    """
    x, y, z = point
    factor = focal_length / (focal_length + z)
    px = int(WINDOW_WIDTH // 2 + x * factor)
    py = int(WINDOW_HEIGHT // 2 - y * factor)
    return px, py

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("AutoProjector 3D Visualization")
    clock = pygame.time.Clock()

    # Initialize the AutoProjector
    auto_proj = AutoProjector()

    # Camera settings
    focal_length = 500

    # Simulation parameters
    running = True
    time_step = 0.1

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    auto_proj.set_control_input('position', 0.9)  # Move closer to max length
                elif event.key == pygame.K_DOWN:
                    auto_proj.set_control_input('position', 0.1)  # Move closer to min length
                elif event.key == pygame.K_LEFT:
                    auto_proj.set_control_output('azimuth', -np.pi / 8)  # Rotate left
                elif event.key == pygame.K_RIGHT:
                    auto_proj.set_control_output('azimuth', np.pi / 8)  # Rotate right
                elif event.key == pygame.K_w:
                    auto_proj.set_control_output('elevation', np.pi / 16)  # Tilt up
                elif event.key == pygame.K_s:
                    auto_proj.set_control_output('elevation', -np.pi / 16)  # Tilt down

        # Simulate the AutoProjector
        auto_proj.simulate_step(time_step)

        # Get state for visualization
        state = auto_proj.get_state()
        current_length = state['current_length']
        transformation = state['output_transformation']

        # Define 3D points for visualization
        # Base of the shell (Side A)
        base_point = np.array([0, 0, 0])

        # End of the piston (Side B)
        piston_point = np.array([0, 0, current_length])

        # Apply transformation to the piston point
        piston_transformed = transformation @ np.array([piston_point[0], piston_point[1], piston_point[2], 1])

        # Convert to 2D points
        base_2d = project_to_2d(base_point)
        piston_2d = project_to_2d(piston_transformed[:3])

        # Draw the shell and piston as sticks
        pygame.draw.line(screen, BLUE, base_2d, piston_2d, 5)
        pygame.draw.circle(screen, RED, base_2d, 8)  # Base joint
        pygame.draw.circle(screen, WHITE, piston_2d, 8)  # Piston joint

        # Update the display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
