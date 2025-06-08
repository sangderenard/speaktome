from .threadmanager import ThreadManager
from state import State
from torch_geometric.data import Data
import logging
import torch
from typing import Dict, Any, List, Optional
import os

# Configure logging for detailed debug and info messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PhysicalObject:
    """
    Represents a tangible entity within the simulation framework. Each PhysicalObject is tightly
    integrated with the State class to define its gestalt location (center of reference) and maintains
    comprehensive vertex buffers for accurate geometry representation. This dual system ensures precise
    physical modeling and facilitates hierarchical relationships within the simulation environment.
    
    ### **Universal Mapped Tensor State Dictionary Tutorial**
    
    The `PhysicalObject` class leverages the `State` class to manage its physical attributes systematically.
    This ensures consistency across different simulators and simplifies interactions within the simulation
    framework.
    
    #### **Key Components**
    
    - **State Integration**: Utilizes the `State` class to manage positional, rotational, and other critical features.
    - **Vertex Buffers**: Maintains vertex buffers for rendering and physics simulations, independent yet complementary
      to the state data.
    - **Graph Neural Network (GNN) Integration**: Prepares the object as a node within a GNN, encapsulating its
      physical features for advanced simulations.
    - **Hierarchical Relationships**: Supports relationships such as containment and fusion with other PhysicalObjects.
    
    ### **Attributes**
        object_id (str): The unique identifier of the PhysicalObject.
        state (State): The state object representing the PhysicalObject's physical attributes.
        thread_manager (ThreadManager): The ThreadManager instance overseeing thread safety.
        vertex_buffers (Dict[str, torch.Tensor]): A collection of vertex buffers for different coordinate representations.
        arbitrary_shape_buffer (Optional[torch.Tensor]): An optional buffer containing the object's original geometry before normalization.
        graph_data (Data): The representation of the object as a node within a Graph Neural Network (GNN).
        object_file (Optional[str]): Path to the `.obj` file representing the object.
        material_file (Optional[str]): Path to the material file for rendering.
        contains (List['PhysicalObject']): List of objects contained within this object.
        is_contained_by (Optional['PhysicalObject']): The object that contains this object.
        fused_with (List['PhysicalObject']): List of objects fused with this object.
    
    ### **Methods**
        __init__: Initializes the PhysicalObject with core properties.
        initialize_vertex_buffers: Initializes vertex buffers based on geometry.
        set_arbitrary_shape: Assigns an arbitrary shape to the object.
        prepare_graph_node: Prepares the object as a graph node for GNN.
        attach: Attaches this object to another (contain, contained_by, fuse).
        detach: Detaches this object from another.
        update_state: Updates the state using a new State object.
        validate_state: Validates the object's state against the mask.
        render_to_obj: Renders the object's geometry to `.obj` and `.mtl` files.
        inversion: Creates panels from normal polygons based on thickness.
        calculate_normals: Calculates normals for the object's geometry.
        formalize_fused_bullet_objects: Formalizes all connected objects as fused Bullet objects.
        __str__: Provides a detailed string representation of the object.
    """

    def __init__(
        self,
        object_id: str,
        initial_state: State,
        thread_manager: ThreadManager,
        object_file: Optional[str] = None,
        material_file: Optional[str] = None,
    ):
        """
        Initializes the PhysicalObject with a unique identifier, a State object for its gestalt location,
        and vertex buffers for geometry representation. Thread safety is managed via the ThreadManager.

        Args:
            object_id (str): Unique identifier for the PhysicalObject.
            initial_state (State): A State object defining the initial state of the PhysicalObject.
            thread_manager (ThreadManager): Manages thread-safe operations.
            object_file (Optional[str]): Path to the `.obj` file for the object's geometry.
            material_file (Optional[str]): Path to the material file for the object's rendering.
        
        Raises:
            TypeError: If initial_state is not an instance of State.
        """
        # Validate that initial_state is an instance of State
        if not isinstance(initial_state, State):
            raise TypeError("initial_state must be an instance of the State class.")

        # Core properties
        self.object_id = object_id
        self.state = initial_state
        self.thread_manager = thread_manager
        self.thread_manager.register_object(self.object_id)

        # Geometry properties
        self.vertex_buffers: Dict[str, torch.Tensor] = {}
        self.arbitrary_shape_buffer: Optional[torch.Tensor] = None
        self.object_file = object_file
        self.material_file = material_file

        # Hierarchical relationships with other PhysicalObjects
        self.contains: List["PhysicalObject"] = []
        self.is_contained_by: Optional["PhysicalObject"] = None
        self.fused_with: List["PhysicalObject"] = []

        # Graph Neural Network node data
        self.graph_data = self.prepare_graph_node()

        # Initialize vertex buffers
        self.initialize_vertex_buffers()

        logger.info(
            f"PhysicalObject '{self.object_id}' initialized with state: {self.state}."
        )

    def initialize_vertex_buffers(self):
        """
        Initializes the vertex buffers for the PhysicalObject's geometry.

        This method sets up the necessary vertex data for rendering and physics simulations.
        By default, it initializes a unit cube. Users should replace or extend these buffers
        with actual geometry data as needed.
        """
        logger.debug(f"Initializing vertex buffers for PhysicalObject '{self.object_id}'.")

        # Example: Unit cube vertices
        self.vertex_buffers["unit_cube"] = torch.tensor([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ], dtype=torch.float32)

        logger.debug(
            f"Vertex buffers for PhysicalObject '{self.object_id}' initialized with {self.vertex_buffers['unit_cube'].size(0)} vertices."
        )

    def set_arbitrary_shape(self, shape_vertices: torch.Tensor):
        """
        Assigns an arbitrary shape to the PhysicalObject by populating the
        arbitrary_shape_buffer with the provided vertices. This buffer stores the
        object's unique geometric configuration, ensuring consistent mass and moment of inertia
        across all representations.

        Args:
            shape_vertices (torch.Tensor): A tensor of shape (N, 3) representing the object's
                                           original vertices before normalization.
        
        Raises:
            ValueError: If shape_vertices is not a 2D tensor with 3 columns.
        """
        if not isinstance(shape_vertices, torch.Tensor) or shape_vertices.ndim != 2 or shape_vertices.size(1) != 3:
            raise ValueError("shape_vertices must be a 2D tensor with shape (N, 3).")

        self.arbitrary_shape_buffer = shape_vertices
        logger.info(
            f"PhysicalObject '{self.object_id}' set arbitrary shape buffer with {shape_vertices.size(0)} vertices."
        )

    def prepare_graph_node(self) -> Data:
        """
        Prepares the PhysicalObject as a node within a Graph Neural Network (GNN) using
        PyTorch Geometric. This involves encapsulating the object's position, velocity, and orientation
        as node features, enabling seamless integration into the coordinated system of
        physics prediction modeling.

        Returns:
            Data: A PyTorch Geometric Data object representing the node with its features.
        """
        # Extract features from the State object
        position = torch.tensor(
            self.state.state["features"]["position"]["subfeatures"][:3], dtype=torch.float32
        )
        velocity = torch.tensor(
            self.state.state["features"]["velocity"]["subfeatures"][:3], dtype=torch.float32
        )
        # Assuming quaternion components are stored starting at index 3
        orientation = torch.tensor(
            self.state.state["features"]["position"]["subfeatures"][3:7], dtype=torch.float32
        )

        # Combine position, velocity, and orientation into a single feature vector
        features = torch.cat([position, velocity, orientation]).unsqueeze(0)  # Shape: (1, 9)

        graph_node = Data(x=features)

        logger.debug(
            f"Prepared GNN node for PhysicalObject '{self.object_id}' with features: {features.tolist()}"
        )
        return graph_node

    def attach(self, target: "PhysicalObject", mode: str):
        """
        Attaches this PhysicalObject to another based on the specified mode.

        Args:
            target (PhysicalObject): The target object to attach to.
            mode (str): The type of attachment ('contain', 'contained_by', 'fuse').

        Raises:
            ValueError: If the mode is invalid.
        """
        if mode == "contain":
            self.contains.append(target)
            target.is_contained_by = self
            logger.info(
                f"PhysicalObject '{self.object_id}' now contains '{target.object_id}'."
            )
        elif mode == "contained_by":
            self.is_contained_by = target
            target.contains.append(self)
            logger.info(
                f"PhysicalObject '{self.object_id}' is now contained by '{target.object_id}'."
            )
        elif mode == "fuse":
            self.fused_with.append(target)
            target.fused_with.append(self)
            logger.info(
                f"PhysicalObject '{self.object_id}' is now fused with '{target.object_id}'."
            )
        else:
            raise ValueError(
                "Attachment mode must be 'contain', 'contained_by', or 'fuse'."
            )

    def detach(self, target: "PhysicalObject", mode: str):
        """
        Detaches this PhysicalObject from another based on the specified mode.

        Args:
            target (PhysicalObject): The target object to detach from.
            mode (str): The type of detachment ('contain', 'contained_by', 'fuse').

        Raises:
            ValueError: If the mode is invalid or the target is not attached in the specified mode.
        """
        if mode == "contain":
            if target in self.contains:
                self.contains.remove(target)
                target.is_contained_by = None
                logger.info(
                    f"PhysicalObject '{self.object_id}' no longer contains '{target.object_id}'."
                )
            else:
                raise ValueError(
                    f"PhysicalObject '{target.object_id}' is not contained by '{self.object_id}'."
                )
        elif mode == "contained_by":
            if self.is_contained_by == target:
                self.is_contained_by = None
                target.contains.remove(self)
                logger.info(
                    f"PhysicalObject '{self.object_id}' is no longer contained by '{target.object_id}'."
                )
            else:
                raise ValueError(
                    f"PhysicalObject '{self.object_id}' is not contained by '{target.object_id}'."
                )
        elif mode == "fuse":
            if target in self.fused_with:
                self.fused_with.remove(target)
                target.fused_with.remove(self)
                logger.info(
                    f"PhysicalObject '{self.object_id}' is no longer fused with '{target.object_id}'."
                )
            else:
                raise ValueError(
                    f"PhysicalObject '{target.object_id}' is not fused with '{self.object_id}'."
                )
        else:
            raise ValueError(
                "Detachment mode must be 'contain', 'contained_by', or 'fuse'."
            )

    def update_state(self, new_state: State):
        """
        Updates the state of the PhysicalObject using a new State object.

        Args:
            new_state (State): The new State object to update the PhysicalObject's state.

        Raises:
            TypeError: If new_state is not an instance of State.
        """
        if not isinstance(new_state, State):
            raise TypeError("new_state must be an instance of the State class.")

        # Update the State object
        self.state = new_state

        # Update the GNN node data to reflect the new state
        self.graph_data = self.prepare_graph_node()

        logger.info(
            f"PhysicalObject '{self.object_id}' updated state to: {self.state}."
        )

    def validate_state(self) -> bool:
        """
        Validates whether the current state meets the minimum required characteristics
        based on the mask and other criteria defined in the State object.

        Returns:
            bool: True if the state is valid, False otherwise.
        """
        is_valid = self.state.validate_state()
        if is_valid:
            logger.info(f"PhysicalObject '{self.object_id}' state validation passed.")
        else:
            logger.warning(f"PhysicalObject '{self.object_id}' state validation failed.")
        return is_valid

    def render_to_obj(self, output_directory: str):
        """
        Renders the object's geometry to `.obj` and `.mtl` files, saving the files to the specified directory.

        Args:
            output_directory (str): Directory where the `.obj` and `.mtl` files will be saved.
        
        Raises:
            FileNotFoundError: If the arbitrary_shape_buffer is not set.
        """
        if self.arbitrary_shape_buffer is None:
            logger.error(
                f"PhysicalObject '{self.object_id}' has no arbitrary shape to render."
            )
            raise FileNotFoundError(
                f"PhysicalObject '{self.object_id}' has no arbitrary shape to render."
            )

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        obj_path = os.path.join(output_directory, f"{self.object_id}.obj")
        mtl_path = os.path.join(output_directory, f"{self.object_id}.mtl")

        # Write the .obj file
        with open(obj_path, "w") as obj_file:
            obj_file.write(f"# Object: {self.object_id}\n")
            if self.material_file:
                obj_file.write(f"mtllib {os.path.basename(self.material_file)}\n")
            for vertex in self.arbitrary_shape_buffer.tolist():
                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            # Placeholder for faces and normals; users should populate based on actual mesh data
            logger.info(
                f"PhysicalObject '{self.object_id}' rendered to '{obj_path}'."
            )

        # Write the .mtl file if material information is provided
        if self.material_file:
            with open(mtl_path, "w") as mtl_file:
                mtl_file.write(f"# Material for {self.object_id}\n")
                # Placeholder for material properties; users should populate based on actual material data
                logger.info(
                    f"PhysicalObject '{self.object_id}' material written to '{mtl_path}'."
                )

        # Update object_file and material_file attributes
        self.object_file = obj_path
        self.material_file = mtl_path
        logger.debug(
            f"PhysicalObject '{self.object_id}' updated object and material file paths."
        )

    def inversion(self, thickness: float):
        """
        Creates panels from the normal polygons of the object's geometry based on the given thickness.
        This method generates new geometric data representing the object's structure with specified thickness.

        Args:
            thickness (float): The thickness parameter to define the panel depth.
        
        Raises:
            ValueError: If the arbitrary_shape_buffer is not set.
        """
        if self.arbitrary_shape_buffer is None:
            logger.error(
                f"PhysicalObject '{self.object_id}' has no arbitrary shape to invert."
            )
            raise ValueError(
                f"PhysicalObject '{self.object_id}' has no arbitrary shape to invert."
            )

        # Calculate normals for each vertex
        normals = self.calculate_normals()

        # Offset vertices along their normals by the specified thickness
        offset = normals * thickness
        inverted_vertices = self.arbitrary_shape_buffer + offset

        # Merge original and inverted vertices to create panels
        self.arbitrary_shape_buffer = torch.cat(
            [self.arbitrary_shape_buffer, inverted_vertices], dim=0
        )

        logger.info(
            f"PhysicalObject '{self.object_id}' performed inversion with thickness {thickness}."
        )

    def calculate_normals(self) -> torch.Tensor:
        """
        Calculates normals for the current arbitrary_shape_buffer.

        This is a placeholder method and should be implemented based on actual geometry processing.
        For demonstration purposes, this method returns upward normals.

        Returns:
            torch.Tensor: A tensor of normals corresponding to each vertex.
        """
        if self.arbitrary_shape_buffer is None:
            logger.error(
                f"PhysicalObject '{self.object_id}' has no geometry to calculate normals."
            )
            raise ValueError(
                f"PhysicalObject '{self.object_id}' has no geometry to calculate normals."
            )

        # Placeholder: Assign upward normals (0, 1, 0) to all vertices
        normals = torch.tensor(
            [[0.0, 1.0, 0.0]] * self.arbitrary_shape_buffer.size(0),
            dtype=torch.float32,
        )
        logger.debug(
            f"Calculated normals for PhysicalObject '{self.object_id}': {normals.tolist()}"
        )
        return normals

    def formalize_fused_bullet_objects(self):
        """
        Formalizes all connected (fused) objects as fused Bullet objects.

        This involves creating compound collision shapes in Bullet Physics to represent fused objects.

        Note:
            This method assumes integration with Bullet Physics is handled elsewhere.
        """
        if not self.fused_with:
            logger.info(
                f"PhysicalObject '{self.object_id}' has no fused objects to formalize."
            )
            return

        # Placeholder implementation
        fused_ids = [obj.object_id for obj in self.fused_with]
        logger.info(
            f"PhysicalObject '{self.object_id}' formalizing fused objects: {fused_ids}."
        )
        # Example: Create a compound shape by combining individual shapes
        # Actual implementation would interact with Bullet's API to create compound collision shapes

    def __str__(self):
        """
        Provides a detailed string representation of the PhysicalObject's current state,
        including its unique identifier, position, orientation, attachments, and physical
        manifestation files.

        Returns:
            str: A string summarizing the PhysicalObject's state.
        """
        contained_str = ", ".join(
            [obj.object_id for obj in self.contains]
        ) if self.contains else "None"
        fused_str = ", ".join(
            [obj.object_id for obj in self.fused_with]
        ) if self.fused_with else "None"
        contained_by_str = (
            self.is_contained_by.object_id if self.is_contained_by else "None"
        )
        return (
            f"PhysicalObject(ID: {self.object_id}, "
            f"Position: {self.state.state['features']['position']['subfeatures'][:3]}, "
            f"Orientation: {self.state.state['features']['position']['subfeatures'][3:7]}, "
            f"Contains: [{contained_str}], "
            f"Contained By: {contained_by_str}, "
            f"Fused With: [{fused_str}], "
            f"Object File: {self.object_file}, "
            f"Material File: {self.material_file})"
        )
# Assuming State and ThreadManager classes are properly defined and imported

# Initialize ThreadManager
thread_manager = ThreadManager()

# Create initial State for the PhysicalObject
initial_state = State(object_id="obj_001")
initial_state.state["features"]["position"]["subfeatures"][0] = 0.0  # x
initial_state.state["features"]["position"]["subfeatures"][1] = 0.0  # y
initial_state.state["features"]["position"]["subfeatures"][2] = 0.0  # z
initial_state.state["features"]["velocity"]["subfeatures"][0] = 1.0  # vx
initial_state.state["features"]["velocity"]["subfeatures"][1] = 0.0  # vy
initial_state.state["features"]["velocity"]["subfeatures"][2] = 0.0  # vz
initial_state.state["features"]["position"]["subfeatures"][3] = 0.0  # q1
initial_state.state["features"]["position"]["subfeatures"][4] = 0.0  # q2
initial_state.state["features"]["position"]["subfeatures"][5] = 0.0  # q3
initial_state.state["features"]["position"]["subfeatures"][6] = 1.0  # q4 (unit quaternion)
initial_state.state["features"]["temperature"]["subfeatures"][7] = 25.0  # Temperature

# Initialize PhysicalObject
obj = PhysicalObject(
    object_id="obj_001",
    initial_state=initial_state,
    thread_manager=thread_manager,
    object_file="obj_001.obj",
    material_file="obj_001.mtl"
)

# Assign an arbitrary shape
shape_vertices = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
], dtype=torch.float32)
obj.set_arbitrary_shape(shape_vertices)

# Perform inversion
obj.inversion(thickness=0.1)

# Render to OBJ files
obj.render_to_obj(output_directory="./rendered_objects")

# Validate the object's state
if obj.validate_state():
    logger.info("Object state is valid.")
else:
    logger.warning("Object state is invalid.")

# Display the object's details
print(obj)
