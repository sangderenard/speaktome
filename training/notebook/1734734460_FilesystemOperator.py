import os
import torch

class FilesystemOperator:
    def __init__(self, lazy_load=True, debug=False):
        self.lazy_load = lazy_load
        self.debug = debug

    def standard_node(self, node, state_tensor, depth=1):
        """
        Synchronize the state tensor with filesystem properties.
        Args:
            node (dict): Filesystem node with 'path' and current state.
            state_tensor (torch.Tensor): State tensor to update.
            depth (int): Folder depth to load.
        """
        path = node["path"]
        immediate_fields = ["inode_id", "depth", "type_id"]
        lazy_fields = ["file_size", "access_time", "modification_time"]

        # Ensure immediate fields are up-to-date
        if "inode_id" in immediate_fields:
            state_tensor[node["index"], 0] = os.stat(path).st_ino
        if "depth" in immediate_fields:
            state_tensor[node["index"], 4] = depth
        if "type_id" in immediate_fields:
            state_tensor[node["index"], 5] = 0 if os.path.isdir(path) else 1

        # Load lazy fields only if configured
        if not self.lazy_load:
            self._update_lazy_fields(node, state_tensor)

    def _update_lazy_fields(self, node, state_tensor):
        path = node["path"]
        stat = os.stat(path)
        state_tensor[node["index"], 1] = stat.st_size
        state_tensor[node["index"], 2] = stat.st_atime
        state_tensor[node["index"], 3] = stat.st_mtime
