import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class AdaptiveGraphNetwork(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, device="cuda", precision=torch.float64):
        """
        Adaptive Graph Network for geometry refinement.

        Args:
            input_dim (int): Dimension of input features (vertex + edge + global).
            hidden_dim (int): Dimension of hidden layers.
            device (str): Device for computations.
            precision (torch.dtype): Precision for tensor operations.
        """
        super(AdaptiveGraphNetwork, self).__init__()
        self.device = torch.device(device)
        self.precision = precision

        # Graph convolution layers
        self.conv1 = pyg_nn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.SAGEConv(hidden_dim, hidden_dim)

        # Attention mechanism for connectivity adjustment
        self.edge_attention = pyg_nn.GATConv(hidden_dim, 1, heads=1)

        # MLP for vertex updates
        self.vertex_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output: new vertex positions (x, y, z)
        )

        # MLP for edge weight adjustment
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output: connectivity weight
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, data):
        """
        Forward pass for graph adaptation.

        Args:
            data (torch_geometric.data.Data): Graph data with vertices and edges.

        Returns:
            torch.Tensor: Updated vertex positions (N x 3).
            torch.Tensor: Updated adjacency weights (E x 1).
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # Compute new vertex positions
        new_positions = self.vertex_update(x)

        # Update edge connectivity using attention
        edge_weights = self.edge_attention(x, edge_index)
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        new_connectivity = self.edge_update(edge_features)

        return new_positions, new_connectivity

    @staticmethod
    def create_graph(intersection_tensor, offsets, edge_index, isosurface_func, device="cuda"):
        """
        Create a graph representation from offsets, edge indices, and intersection statistics.

        Args:
            intersection_tensor (torch.Tensor): Tensor with intersection statistics (N x F).
            offsets (torch.Tensor): Vertex positions (N x 3).
            edge_index (torch.Tensor): Edge indices (2 x E).
            isosurface_func (callable): Isosurface function f(x, y, z).
            device (str): Device for computations.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        # Extract global intersection points
        intersection_points = intersection_tensor.view(-1, 3)  # Flatten to (N, 3)

        # Ensure valid points only (filter NaNs)
        valid_mask = ~torch.isnan(intersection_points[:, 0])
        valid_points = intersection_points[valid_mask]
        

        # Split valid points into x, y, z tensors for the scalar function
        x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

        # Compute scalar values and gradients at the valid global positions
        scalar_values = isosurface_func(x, y, z)  # Shape: (N,)
        gradients = torch.autograd.grad(
            outputs=scalar_values,
            inputs=valid_points,
            grad_outputs=torch.ones_like(scalar_values, device=device),
            create_graph=True,
            retain_graph=True
        )[0]  # Shape: (N, 3)

        # Combine valid points, gradients, and scalar values into vertex features
        scalar_values = scalar_values.unsqueeze(1)  # Shape: (N, 1)
        vertex_features = torch.cat([valid_points, gradients, scalar_values], dim=1)  # Shape: (N, 7)

        # Compute edge features using offsets
        edge_vectors = offsets[edge_index[1].clone().detach().to(torch.int64)] - offsets[edge_index[0].clone().detach().to(torch.int64)]
        edge_lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
        edge_features = torch.cat([edge_vectors, edge_lengths], dim=1)  # Shape: (E, 4)

        # Combine features into graph data
        data = Data(
            x=vertex_features.to(device),  # Vertex features
            edge_index=edge_index.to(device, dtype=torch.int64),  # Edge indices
            edge_attr=edge_features.to(device)  # Edge features
        )
        return data


    def refresh_geometry(self, data, base_offsets, base_edge_index):
        """
        Apply the network's outputs to update the geometry.

        Args:
            data (Data): PyTorch Geometric Data object with graph features.
            base_offsets (torch.Tensor): Original vertex offsets (N x 3).
            base_edge_index (torch.Tensor): Original edge indices (2 x E).

        Returns:
            torch.Tensor: Updated vertex offsets (N x 3).
            torch.Tensor: Updated edge indices (2 x E).
        """
        new_positions, new_connectivity = self.forward(data)

        # Update vertex offsets
        updated_offsets = base_offsets + new_positions

        # Update edge connectivity
        edge_mask = new_connectivity.squeeze() > 0.5  # Threshold for keeping edges
        updated_edge_index = base_edge_index[:, edge_mask]

        return updated_offsets, updated_edge_index
