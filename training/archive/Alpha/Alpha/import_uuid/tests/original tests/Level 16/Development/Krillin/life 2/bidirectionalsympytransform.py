import torch
import sympy as sp
from typing import Callable, Dict, List, Tuple


class SymPyBidirectionalTransformer:
    """
    **SymPyBidirectionalTransformer**
    Converts SymPy equations defining relationships between symbolic dependencies
    into bidirectional transformation lambda functions.
    """

    @staticmethod
    def create_transform_pair(equation: sp.Equality, variables: List[sp.Symbol]) -> Tuple[Callable, Callable]:
        """
        Create a pair of forward and reverse lambda functions from a SymPy equation.
        
        Args:
            equation (sp.Equality): SymPy equation defining the relationship (e.g., Eq(y, x * 2)).
            variables (List[sp.Symbol]): List of variables in the equation [x, y].
        
        Returns:
            Tuple[Callable, Callable]: Forward and reverse transformation functions.
        """
        if not isinstance(equation, sp.Equality):
            raise ValueError("The equation must be a SymPy Equality (e.g., Eq(y, x * 2)).")
        
        lhs, rhs = equation.lhs, equation.rhs

        # Forward transformation: Solve for the last variable in the list
        forward_lambda = sp.lambdify(variables[:-1], rhs, "numpy")
        
        # Reverse transformation: Solve for the first variable
        reverse_solutions = sp.solve(equation, variables[0])
        if not reverse_solutions:
            raise ValueError("Unable to determine a reverse transformation.")
        reverse_lambda = sp.lambdify(variables[1:], reverse_solutions[0], "numpy")
        
        return forward_lambda, reverse_lambda


class TransformNode:
    """
    **TransformNode**
    Represents a node in the graph, storing its current state, history, and bidirectional logic.
    """

    def __init__(self, id: str):
        self.id = id
        self.state = None  # Current state of the node
        self.history = []  # Full history of values that have passed through this node
        self.edges = []  # List of edges connected to this node

    def record_transition(self, value):
        """
        Record the value passing through the node for use in blending or synchronization.
        """
        self.history.append(value)

    def get_full_history(self) -> List:
        """
        Retrieve the full history of values for the node.
        """
        return self.history


class TransformEdge:
    """
    **TransformEdge**
    Represents a bidirectional edge in the graph. Encodes forward and reverse transformations.
    """

    def __init__(
        self, 
        source: TransformNode, 
        target: TransformNode, 
        forward_transform: Callable, 
        reverse_transform: Callable
    ):
        self.source = source
        self.target = target
        self.forward_transform = forward_transform
        self.reverse_transform = reverse_transform

    def execute_forward(self, value):
        """
        Execute the forward transformation.
        """
        transformed_value = self.forward_transform(value)
        return transformed_value

    def execute_reverse(self, value):
        """
        Execute the reverse transformation.
        """
        transformed_value = self.reverse_transform(value)
        return transformed_value


class TransformGraph:
    """
    **TransformGraph**
    A graph structure that enables bidirectional transformations with complete historical tracking.
    """

    def __init__(self):
        self.nodes = {}  # Dictionary of nodes by ID
        self.edges = []  # List of all edges

    def add_node(self, node: TransformNode):
        """
        Add a node to the graph.
        """
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, forward_transform: Callable, reverse_transform: Callable):
        """
        Add a bidirectional edge between two nodes with defined transformations.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node not found.")
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        edge = TransformEdge(source, target, forward_transform, reverse_transform)
        source.edges.append(edge)
        target.edges.append(edge)
        self.edges.append(edge)

    def execute_forward(self, start_node_id: str, value):
        """
        Traverse the graph in the forward direction from a starting node.
        """
        if start_node_id not in self.nodes:
            raise ValueError("Starting node not found.")
        start_node = self.nodes[start_node_id]
        self._traverse_forward(start_node, value)

    def _traverse_forward(self, node: TransformNode, value):
        """
        Recursive traversal for forward direction execution.
        """
        node.record_transition(value)
        for edge in node.edges:
            if edge.source == node:
                transformed_value = edge.execute_forward(value)
                self._traverse_forward(edge.target, transformed_value)

    def execute_reverse(self, end_node_id: str, value):
        """
        Traverse the graph in the reverse direction from an ending node.
        """
        if end_node_id not in self.nodes:
            raise ValueError("Ending node not found.")
        end_node = self.nodes[end_node_id]
        self._traverse_reverse(end_node, value)

    def _traverse_reverse(self, node: TransformNode, value):
        """
        Recursive traversal for reverse direction execution.
        """
        node.record_transition(value)
        for edge in node.edges:
            if edge.target == node:
                transformed_value = edge.execute_reverse(value)
                self._traverse_reverse(edge.source, transformed_value)


# Example Usage
if __name__ == "__main__":
    # Define symbolic variables
    x, y = sp.symbols("x y")

    # Define the equation: y = x * 2
    equation = sp.Eq(y, x * 2)

    # Generate forward and reverse transformations
    forward_transform, reverse_transform = SymPyBidirectionalTransformer.create_transform_pair(equation, [x, y])

    # Test transformations
    forward_value = forward_transform(5)  # y = 5 * 2 => 10
    reverse_value = reverse_transform(10)  # x = 10 / 2 => 5

    print("Forward Transform Result:", forward_value)  # Output: 10
    print("Reverse Transform Result:", reverse_value)  # Output: 5

