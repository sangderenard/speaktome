from .ThreadManager import ThreadManager
from typing import Dict, Any, List
import logging
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PhysicalObject:
    """
    PhysicalObject is the cornerstone of our simulation framework, embodying any tangible entity
    within the virtual environment. Each PhysicalObject maintains comprehensive vertex buffers
    to represent its geometry in both spherical and rectangular coordinates. Additionally, an
    optional buffer stores the object's unique shape before normalization, ensuring consistent
    mass and moment of inertia across all representations.

    Attributes:
        object_id (str): The unique identifier of the PhysicalObject.
        position (torch.Tensor): The 3D position of the object within the simulation space.
        orientation (torch.Tensor): The orientation of the object, represented by Euler angles.
        thread_manager (ThreadManager): The ThreadManager instance overseeing thread safety.
        vertex_buffers (Dict[str, torch.Tensor]): A collection of vertex buffers for different
                                                  coordinate representations.
        arbitrary_shape_buffer (torch.Tensor): An optional buffer containing the object's original
                                               geometry before normalization.
        graph_data (Data): The representation of the object as a node within a Graph Neural Network (GNN).
    """

    def __init__(self, object_id: str, position: List[float], orientation: List[float], thread_manager: ThreadManager):
        """
        Initializes the PhysicalObject with its unique identifier, position, orientation,
        and a reference to the ThreadManager for managing thread-safe operations. Upon
        initialization, it registers itself with the ThreadManager, sets up its vertex
        buffers for spherical and rectangular representations, and prepares its graph
        node data for integration with a Graph Neural Network (GNN).

        Args:
            object_id (str): The unique identifier for the PhysicalObject.
            position (List[float]): A list of three floats representing the 3D position (x, y, z).
            orientation (List[float]): A list of three floats representing the orientation (roll, pitch, yaw).
            thread_manager (ThreadManager): The ThreadManager instance responsible for thread safety.
        """
        self.object_id = object_id
        self.position = torch.tensor(position, dtype=torch.float64, requires_grad=False)
        self.orientation = torch.tensor(orientation, dtype=torch.float64, requires_grad=False)
        
        # Register with ThreadManager
        self.thread_manager = thread_manager
        self.thread_manager.register_object(self.object_id)
        
        # Initialize vertex buffers
        self.vertex_buffers: Dict[str, torch.Tensor] = {}
        self.initialize_vertex_buffers()
        
        # Optional arbitrary shape buffer
        self.arbitrary_shape_buffer: torch.Tensor = None  # To be set if needed
        
        # Graph node attributes for PyG
        self.graph_data = self.prepare_graph_node()
        
        logger.info(f"PhysicalObject '{self.object_id}' initialized with position {self.position.tolist()} "
                    f"and orientation {self.orientation.tolist()}.")

    def initialize_vertex_buffers(self):
        """
        Initializes the two default vertex buffers for the PhysicalObject:
        
        - 'spherical': Represents a unit sphere using spherical coordinates, facilitating
                       hyperlocal evaluations and ensuring consistent mass and moment of
                       inertia across all spherical representations.
        
        - 'rectangular': Represents a unit cube using rectangular coordinates, providing
                         a standardized basis for physics prediction modeling and simulation.
        
        These buffers are essential for maintaining uniform geometric representations
        across different coordinate systems, enabling seamless integration with the
        coordinated system of physics prediction modeling.
        """
        # Spherical coordinates (unit sphere)
        sphere_vertices = self.generate_unit_sphere_vertices()
        self.vertex_buffers['spherical'] = sphere_vertices
        logger.debug(f"PhysicalObject '{self.object_id}' initialized spherical vertex buffer with {sphere_vertices.shape[0]} vertices.")
        
        # Rectangular coordinates (unit cube)
        cube_vertices = self.generate_unit_cube_vertices()
        self.vertex_buffers['rectangular'] = cube_vertices
        logger.debug(f"PhysicalObject '{self.object_id}' initialized rectangular vertex buffer with {cube_vertices.shape[0]} vertices.")

    def generate_unit_sphere_vertices(self, num_vertices: int = 1000) -> torch.Tensor:
        """
        Generates a PyTorch tensor containing vertices uniformly distributed on the surface
        of a unit sphere. This representation is crucial for maintaining consistent mass and
        moment of inertia across all spherical representations of the PhysicalObject.

        Args:
            num_vertices (int, optional): The number of vertices to generate for the sphere.
                                          Defaults to 1000.

        Returns:
            torch.Tensor: A tensor of shape (num_vertices, 3) representing the spherical vertices.
        """
        phi = torch.rand(num_vertices) * 2 * torch.pi  # Azimuthal angle
        theta = torch.acos(1 - 2 * torch.rand(num_vertices))  # Polar angle
        
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        
        sphere_vertices = torch.stack([x, y, z], dim=1)
        return sphere_vertices

    def generate_unit_cube_vertices(self) -> torch.Tensor:
        """
        Generates a PyTorch tensor containing the eight vertices of a unit cube centered
        at the origin. This rectangular representation provides a standardized basis for
        physics prediction modeling and simulation, ensuring uniformity across all cube
        representations of the PhysicalObject.

        Returns:
            torch.Tensor: A tensor of shape (8, 3) representing the rectangular vertices.
        """
        # Define the 8 vertices of a cube centered at the origin
        points = [
            [-1, -1, -1],
            [-1, -1,  1],
            [-1,  1, -1],
            [-1,  1,  1],
            [ 1, -1, -1],
            [ 1, -1,  1],
            [ 1,  1, -1],
            [ 1,  1,  1],
        ]
        cube_vertices = torch.tensor(points, dtype=torch.float64)
        return cube_vertices

    def set_arbitrary_shape(self, shape_vertices: torch.Tensor):
        """
        Assigns an arbitrary shape to the PhysicalObject by populating the
        arbitrary_shape_buffer with the provided vertices. This buffer stores the
        object's unique geometric configuration before normalization to a unit sphere
        and unit cube, ensuring that mass and moment of inertia remain consistent across
        all vertex buffer representations.

        Args:
            shape_vertices (torch.Tensor): A tensor of shape (N, 3) representing the object's
                                           original vertices before normalization.
        """
        self.arbitrary_shape_buffer = shape_vertices
        logger.info(f"PhysicalObject '{self.object_id}' set arbitrary shape buffer with {shape_vertices.shape[0]} vertices.")

    def prepare_graph_node(self) -> Data:
        """
        Prepares the PhysicalObject as a node within a Graph Neural Network (GNN) using
        PyTorch Geometric. This involves encapsulating the object's position and orientation
        as node features, enabling seamless integration into the coordinated system of
        physics prediction modeling.

        Returns:
            Data: A PyTorch Geometric Data object representing the node with its features.
        """
        # Example node features: position and orientation
        node_features = torch.cat([self.position, self.orientation]).unsqueeze(0)  # Shape: (1, 6)
        
        data = Data(x=node_features)
        logger.debug(f"PhysicalObject '{self.object_id}' prepared graph node with features {node_features.tolist()}.")
        return data

    def update(self, dt: float):
        """
        Executes a thread-safe update of the PhysicalObject's state based on the provided
        time step. This method ensures that all modifications to the object's state are
        synchronized and adhere to the thread safety protocols managed by the ThreadManager.

        Args:
            dt (float): The time step for the update, representing the elapsed time in seconds.
        """
        # Acquire lock via ThreadManager
        current_token = self.thread_manager.tokens.get(self.object_id)
        if self.thread_manager.acquire_lock(self.object_id, current_token):
            self.perform_update(dt)
            self.thread_manager.release_lock(self.object_id)
        else:
            logger.error(f"PhysicalObject '{self.object_id}' failed to acquire lock for update.")

    def perform_update(self, dt: float):
        """
        Defines the specific behaviors and state changes that the PhysicalObject undergoes
        during an update. This method is intended to be overridden by subclasses to implement
        customized update logic based on the simulation's requirements.

        Args:
            dt (float): The time step for the update, representing the elapsed time in seconds.
        """
        # Placeholder for update logic
        logger.debug(f"PhysicalObject '{self.object_id}' performing update with dt={dt}.")
        pass

    def __str__(self):
        """
        Provides a human-readable string representation of the PhysicalObject's current state,
        including its unique identifier, position, and orientation.

        Returns:
            str: A string summarizing the PhysicalObject's state.
        """
        return (f"PhysicalObject(ID: {self.object_id}, Position: {self.position.tolist()}, "
                f"Orientation: {self.orientation.tolist()})")

