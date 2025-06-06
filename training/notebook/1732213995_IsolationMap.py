import torch
import networkx as nx

class IsolationMap:
    def __init__(self):
        """
        Initialize an IsolationMap with a directed graph to manage isolation modes.
        """
        self.graph = nx.DiGraph()

    def add_module(self, module_name):
        """
        Add a module (node) to the IsolationMap.

        Parameters:
        - module_name (str): The name of the module to add.
        """
        if module_name not in self.graph:
            self.graph.add_node(module_name, modes=[])

    def connect_modules(self, source, target, modes):
        """
        Connect two modules with a directional edge and assign isolation modes.

        Parameters:
        - source (str): The name of the source module.
        - target (str): The name of the target module.
        - modes (list[int]): Sequence array of isolation modes:
            0 -> Reset gradient
            1 -> Clone tensor
            2 -> Detach tensor
            3 -> Destructive unit conversion
        """
        if source not in self.graph or target not in self.graph:
            raise ValueError("Both source and target modules must be in the IsolationMap.")
        self.graph.add_edge(source, target, modes=modes)

    def isolate_tensor(self, tensor, modes):
        """
        Apply isolation modes to a tensor.

        Parameters:
        - tensor (torch.Tensor): The input tensor to isolate.
        - modes (list[int]): Sequence array of isolation modes.

        Returns:
        - torch.Tensor: The isolated tensor after applying the specified operations.
        """
        result = tensor
        for mode in modes:
            if mode == 0:
                result = result.detach().clone().requires_grad_()
            elif mode == 1:
                result = result.clone()
            elif mode == 2:
                result = result.detach()
            elif mode == 3:
                result = torch.tensor(result.cpu().numpy(), device=tensor.device)
            else:
                raise ValueError(f"Unknown isolation mode: {mode}")
        return result

    def forward(self, source, target, tensor):
        """
        Process a tensor through the isolation modes specified between two connected modules.

        Parameters:
        - source (str): The name of the source module.
        - target (str): The name of the target module.
        - tensor (torch.Tensor): The tensor to process.

        Returns:
        - torch.Tensor: The processed tensor.
        """
        if not self.graph.has_edge(source, target):
            raise ValueError(f"No connection exists between {source} and {target}.")

        modes = self.graph.edges[source, target]['modes']
        return self.isolate_tensor(tensor, modes)

    def visualize(self):
        """
        Visualize the IsolationMap graph structure.
        """
        nx.draw(self.graph, with_labels=True, node_color='lightblue', font_weight='bold', edge_color='gray')

# Example usage
if __name__ == "__main__":
    # Create an IsolationMap
    iso_map = IsolationMap()

    # Add modules
    iso_map.add_module("ModuleA")
    iso_map.add_module("ModuleB")
    iso_map.add_module("ModuleC")

    # Connect modules with isolation modes
    iso_map.connect_modules("ModuleA", "ModuleB", [0, 1])  # Reset gradient, then clone
    iso_map.connect_modules("ModuleB", "ModuleC", [2, 3])  # Detach, then destructive unit conversion

    # Example tensor
    tensor = torch.randn(4, 4, requires_grad=True)

    # Process tensor through ModuleA -> ModuleB -> ModuleC
    tensor_b = iso_map.forward("ModuleA", "ModuleB", tensor)
    tensor_c = iso_map.forward("ModuleB", "ModuleC", tensor_b)

    # Print results
    print("Original Tensor Requires Grad:", tensor.requires_grad)
    print("Tensor after ModuleA -> ModuleB Requires Grad:", tensor_b.requires_grad)
    print("Tensor after ModuleB -> ModuleC Requires Grad:", tensor_c.requires_grad)

    # Visualize the graph
    iso_map.visualize()
