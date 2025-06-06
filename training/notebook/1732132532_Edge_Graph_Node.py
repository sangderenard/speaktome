import uuid
import torch
class Node:
    def __init__(self, features=None):
        """
        Initializes a new node in the graph.

        Args:
            features (torch.Tensor or dict, optional): Initial feature vector or dictionary.
        """
        # Unique identifier
        self.id = self._generate_id()
        
        # Node features
        self.features = features if features is not None else torch.empty(0)

        # Connections
        self.edges = []  # List of Edge objects connecting this node

        # Message-passing queue
        self.message_queue = []

        # Metadata
        self.metadata = {
            "created_at": torch.tensor([torch.cuda.Event().elapsed_time(torch.cuda.Event())]),
            "type": "default"
        }

    def _generate_id(self):
        """Generates a unique identifier for the node."""
        return uuid.uuid4().hex

    def add_edge(self, edge):
        """
        Connects this node to another node via an Edge.

        Args:
            edge (Edge): The edge connecting this node to another.
        """
        self.edges.append(edge)

    def send_message(self, message, target_node_id):
        """
        Sends a message to a connected node through an edge.

        Args:
            message (torch.Tensor): The message to be sent.
            target_node_id (str): The ID of the target node.
        """
        for edge in self.edges:
            if edge.target.id == target_node_id and edge.state["active"]:
                edge.transmit_message(message)
                break

    def receive_message(self, message):
        """Adds an incoming message to the message queue."""
        self.message_queue.append(message)

    def aggregate_messages(self, agg_func=torch.mean):
        """
        Aggregates messages in the queue using the specified function.

        Args:
            agg_func (function): Aggregation function (default: torch.mean).

        Returns:
            torch.Tensor: Aggregated message.
        """
        if len(self.message_queue) == 0:
            return None
        aggregated = agg_func(torch.stack(self.message_queue), dim=0)
        self.message_queue = []  # Clear the queue after aggregation
        return aggregated

    def update_features(self, new_features):
        """
        Updates the node's feature vector.

        Args:
            new_features (torch.Tensor): Updated feature tensor.
        """
        self.features = new_features

    def __repr__(self):
        return f"Node(id={self.id}, num_edges={len(self.edges)}, features_shape={self.features.shape})"

class Edge:
    def __init__(self, source, target, weight=None, features=None):
        """
        Initializes an edge between two nodes.

        Args:
            source (Node): Source node of the edge.
            target (Node): Target node of the edge.
            weight (float, optional): Weight of the edge. Defaults to None.
            features (torch.Tensor, optional): Features associated with the edge. Defaults to None.
        """
        self.id = self._generate_id()
        self.source = source
        self.target = target
        self.weight = weight if weight is not None else torch.tensor(1.0)
        self.features = features if features is not None else torch.empty(0)
        self.state = {"active": True}  # Edge state can change dynamically
        self.metadata = {"created_at": torch.cuda.Event().elapsed_time(torch.cuda.Event())}

    def _generate_id(self):
        """Generates a unique identifier for the edge."""
        return uuid.uuid4().hex

    def activate(self):
        """Activates the edge."""
        self.state["active"] = True

    def deactivate(self):
        """Deactivates the edge."""
        self.state["active"] = False

    def update_weight(self, new_weight):
        """Updates the edge's weight."""
        self.weight = torch.tensor(new_weight)

    def update_features(self, new_features):
        """Updates the edge's features."""
        self.features = new_features

    def transmit_message(self, message):
        """
        Transmits a message from the source node to the target node.

        Args:
            message (torch.Tensor): Message to be transmitted.
        """
        if self.state["active"]:
            self.target.receive_message(message)

    def __repr__(self):
        return f"Edge(id={self.id}, source={self.source.id}, target={self.target.id}, active={self.state['active']})"
class Graph:
    def __init__(self):
        """Initializes a graph with nodes and edges."""
        self.nodes = {}
        self.edges = {}

    def add_node(self, features=None):
        """
        Adds a new node to the graph.

        Args:
            features (torch.Tensor or dict, optional): Feature vector or dictionary for the node.

        Returns:
            Node: The newly created node.
        """
        node = Node(features=features)
        self.nodes[node.id] = node
        return node

    def add_edge(self, source, target, weight=None, features=None):
        """
        Adds an edge to the graph.

        Args:
            source (Node): Source node.
            target (Node): Target node.
            weight (float, optional): Weight of the edge.
            features (torch.Tensor, optional): Features of the edge.

        Returns:
            Edge: The newly created edge.
        """
        edge = Edge(source, target, weight=weight, features=features)
        self.edges[edge.id] = edge
        source.add_edge(edge.id)
        return edge

    def get_node(self, node_id):
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id, None)

    def get_edge(self, edge_id):
        """Retrieves an edge by its ID."""
        return self.edges.get(edge_id, None)

    def remove_node(self, node_id):
        """Removes a node and its associated edges."""
        if node_id in self.nodes:
            # Remove edges associated with the node
            for edge_id in self.nodes[node_id].edges:
                del self.edges[edge_id]
            del self.nodes[node_id]

    def remove_edge(self, edge_id):
        """Removes an edge from the graph."""
        if edge_id in self.edges:
            del self.edges[edge_id]

    def __repr__(self):
        return f"Graph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"
