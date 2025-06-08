import json

class BackupManager:
    """
    Handles saving and loading the graph workspace state.
    """

    def save_graph(self, meta_network, filepath="graph_backup.json"):
        """
        Serialize and save the graph state to a file.
        """
        graph_state = {
            "vertices": meta_network.vertices.tolist(),
            "edges": meta_network.edges.tolist(),
            "fields": meta_network.fields.tolist(),
            "edge_types": meta_network.edge_types,
        }
        with open(filepath, "w") as f:
            json.dump(graph_state, f)
        print(f"Graph state saved to {filepath}.")
        return graph_state

    def load_graph(self, meta_network, filepath="graph_backup.json"):
        """
        Load a saved graph state into the MetaNetwork.
        """
        with open(filepath, "r") as f:
            graph_state = json.load(f)

        meta_network.vertices = torch.tensor(graph_state["vertices"], device=meta_network.device)
        meta_network.edges = torch.tensor(graph_state["edges"], device=meta_network.device)
        meta_network.fields = torch.tensor(graph_state["fields"], device=meta_network.device)
        meta_network.edge_types = graph_state["edge_types"]
        print(f"Graph state loaded from {filepath}.")
