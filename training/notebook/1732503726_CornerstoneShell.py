from .physicalobject import PhysicalObject
from .threadmanager import ThreadManager
from Devices.computer_screen import ComputerScreen
from Devices.cpu import CPU
from Devices.dvibus import DVIBus
from typing import Dict, Any, List
import logging
import torch
import pybullet_data
import pybullet as p
from torch_geometric.data import Data
from Devices.window import Window

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CornerstoneShell(PhysicalObject):
    """
    CornerstoneShell is a specialized subclass of PhysicalObject, representing the cornerstone
    shell within the simulation. It is characterized by six square faces, each possessing
    slightly imperfect cube vertices generated within defined machining tolerances. This
    imperfection ensures realistic physical simulations while maintaining consistent mass and
    moment of inertia across all vertex buffer representations.

    Attributes:
        machining_tolerances (Dict[str, float]): Defines the maximum permissible error for each
                                                face of the cornerstone shell, ensuring that
                                                surface imperfections remain within manufacturing
                                                constraints.

    Methods:
        generate_imperfect_cube(base_size: float = 1.0) -> torch.Tensor:
            Generates a unit cube with slight random perturbations on each vertex based on
            machining tolerances, simulating real-world manufacturing imperfections.

        determine_face(vertex: torch.Tensor) -> str:
            Determines the specific face of the cube to which a given vertex belongs,
            based on its coordinates, facilitating targeted perturbations within tolerances.

        perform_update(dt: float):
            Implements the cornerstone shell's specific update logic, such as simulating
            expansion or contraction due to environmental factors, while ensuring thread-safe
            modifications.

        prepare_graph_node() -> Data:
            Overrides the base method to include additional features, such as mass, in the
            graph node representation, enhancing the node's informational richness for
            integration within a Graph Neural Network (GNN).
    """

    def __init__(self, object_id: str, position: List[float], orientation: List[float], 
                 machining_tolerances: Dict[str, float], thread_manager: ThreadManager):
        """
        Initializes the CornerstoneShell with its unique identifier, position, orientation,
        machining tolerances, and a reference to the ThreadManager. Upon initialization, it
        generates an imperfect cube to represent its geometry and prepares its graph node data
        for GNN integration.

        Args:
            object_id (str): The unique identifier for the CornerstoneShell.
            position (List[float]): A list of three floats representing the 3D position (x, y, z).
            orientation (List[float]): A list of three floats representing the orientation (roll, pitch, yaw).
            machining_tolerances (Dict[str, float]): A dictionary defining the maximum permissible
                                                     error for each face of the shell.
            thread_manager (ThreadManager): The ThreadManager instance responsible for thread safety.
        """
        super().__init__(object_id, position, orientation, thread_manager)
        self.machining_tolerances = machining_tolerances  # e.g., {'+X': 0.01, '-X': 0.01, ...}

        self.physics_client = p.connect(p.DIRECT)  # Headless PyBullet instance
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For accessing PyBullet assets
        
        # Set up the environment
        self.setup_simulation_environment()
        self.remove_dc_offset()
        
        # Initialize imperfect cube vertices
        imperfect_cube = self.generate_imperfect_cube()
        self.set_arbitrary_shape(imperfect_cube)
        logger.info(f"CornerstoneShell '{self.object_id}' initialized with imperfect cube shape.")
                # Initialize devices dictionary
        self.devices = {}
        # Create and register devices
        self.devices['cpu'] = CPU('cpu_id', position, orientation, thread_manager,
                                  heat_generation=50.0, energy_consumption=100.0)
        self.devices['screen'] = ComputerScreen('screen_id', position, orientation, thread_manager,
                                                heat_generation=30.0, energy_consumption=80.0,
                                                width=800, height=600)
        # In CornerstoneShell __init__
        # Create DVIBus instance
        self.dvibus = DVIBus(self.devices['cpu'], self.devices['screen'])
        # Pass DVIBus to devices
        self.devices['cpu'].dvibus = self.dvibus
        self.devices['screen'].dvibus = self.dvibus
        self.devices['window'] = Window('window_id', position, orientation, thread_manager)
 
        logging.info(f"CornerstoneShell '{self.object_id}' initialized with devices.")

    def generate_imperfect_cube(self, base_size: float = 1.0) -> torch.Tensor:
        """
        Generates an imperfect unit cube by introducing slight random perturbations to each
        vertex based on the defined machining tolerances. This simulation of manufacturing
        imperfections ensures that the cornerstone shell behaves realistically within the
        simulation's physics prediction modeling framework.

        Args:
            base_size (float, optional): The base size of the cube (edge length). Defaults to 1.0.

        Returns:
            torch.Tensor: A tensor of shape (8, 3) representing the vertices of the imperfect cube.
        """
        # Define the 8 vertices of a perfect cube centered at the origin
        base_vertices = torch.tensor([
            [-base_size, -base_size, -base_size],
            [-base_size, -base_size,  base_size],
            [-base_size,  base_size, -base_size],
            [-base_size,  base_size,  base_size],
            [ base_size, -base_size, -base_size],
            [ base_size, -base_size,  base_size],
            [ base_size,  base_size, -base_size],
            [ base_size,  base_size,  base_size],
        ], dtype=torch.float64)
        
        # Introduce imperfections based on machining tolerances
        perturbations = []
        for idx, vertex in enumerate(base_vertices):
            face = self.determine_face(vertex)
            tolerance = self.machining_tolerances.get(face, 0.01)  # Default tolerance if not specified
            perturbation = torch.empty(3, dtype=torch.float64).uniform_(-tolerance, tolerance)
            perturbed_vertex = vertex + perturbation
            perturbations.append(perturbed_vertex)
            logger.debug(f"Vertex {idx}: Face {face}, Perturbation {perturbation.tolist()}, New Position {perturbed_vertex.tolist()}")
        
        imperfect_cube = torch.stack(perturbations)
        return imperfect_cube

    def determine_face(self, vertex: torch.Tensor) -> str:
        """
        Determines the specific face of the cube to which a given vertex belongs based on
        its coordinates. This is essential for applying targeted perturbations within the
        defined machining tolerances for each face.

        Args:
            vertex (torch.Tensor): A tensor representing the coordinates of the vertex.

        Returns:
            str: The identifier of the face (e.g., '+X', '-Y') to which the vertex belongs.
        """
        max_coord = torch.abs(vertex).max()
        if torch.abs(vertex[0]) == max_coord:
            return '+X' if vertex[0] > 0 else '-X'
        elif torch.abs(vertex[1]) == max_coord:
            return '+Y' if vertex[1] > 0 else '-Y'
        else:
            return '+Z' if vertex[2] > 0 else '-Z'


    def prepare_graph_node(self) -> Data:
        """
        Overrides the base method to include additional features, such as mass, in the graph
        node representation. This enriched node data facilitates more nuanced interactions
        and integrations within a Graph Neural Network (GNN), enhancing the simulation's
        physics prediction modeling capabilities.

        Returns:
            Data: A PyTorch Geometric Data object representing the node with extended features.
        """
        # Example: Include mass as an additional feature (assuming a default mass value)
        mass = 1.0  # This should be defined appropriately
        node_features = torch.cat([self.position, self.orientation, torch.tensor([mass], dtype=torch.float64)]).unsqueeze(0)  # Shape: (1, 7)
        
        data = Data(x=node_features)
        logger.debug(f"CornerstoneShell '{self.object_id}' prepared graph node with features {node_features.tolist()}.")
        return data


    def setup_simulation_environment(self):
        """Set up a PyBullet simulation environment with zero gravity."""
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
        # Optionally add plane or other objects for debugging
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        logger.debug(f"Plane URDF loaded with ID {plane_id}.")

    def remove_dc_offset(self):
        """Remove the DC offset in the gravitational field for smooth simulation."""
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
        logger.info(f"DC offset removed. Gravity set to zero in PyBullet instance {self.physics_client}.")

    def wrap_coordinates(self, position):
        """Implement periodic boundary conditions in a cube space."""
        cube_size = 10  # Example cube size
        wrapped_position = [(pos + cube_size) % (2 * cube_size) - cube_size for pos in position]
        logger.debug(f"Position {position} wrapped to {wrapped_position} within cube space.")
        return wrapped_position

    def apply_gravitational_variation(self, position):
        """Apply localized gravitational variation to the object."""
        
        # Set a manual seed for reproducibility (optional)
        torch.manual_seed(42)

        # Define mean and standard deviation tensors
        mean = torch.zeros(3)          # Mean = 0 for each element
        std = torch.full((3,), 0.01)  # Std = 0.01 for each element

        # Generate Gaussian noise
        noise = torch.normal(mean, std)
        
        varied_gravity = torch.tensor(position) + noise
        logger.debug(f"Gravitational variation applied. Position: {position}, Noise: {noise}.")
        return varied_gravity

    def update(self, dt: float):
        super().update(dt)
        # Update devices
        for device in self.devices.values():
            # Acquire lock for the device
            device_token = self.thread_manager.get_new_token(device.object_id)
            if self.thread_manager.acquire_lock(device.object_id, device_token):
                try:
                    device.update(dt)
                finally:
                    self.thread_manager.release_lock(device.object_id)
            else:
                logging.warning(f"Failed to acquire lock for device '{device.object_id}'.")
        """Update the physical state of the CornerstoneShell."""
        # Simulate object motion or other behaviors
        wrapped_position = self.wrap_coordinates(self.position.tolist())
        varied_position = self.apply_gravitational_variation(wrapped_position)
        self.position = torch.tensor(varied_position, dtype=torch.float64)
        logger.info(f"CornerstoneShell '{self.object_id}' updated to position {self.position.tolist()}.")
 

    def __del__(self):
        """Ensure the PyBullet client is properly disconnected."""
        p.disconnect(self.physics_client)
        logger.info(f"PyBullet client {self.physics_client} disconnected for CornerstoneShell '{self.object_id}'.")
