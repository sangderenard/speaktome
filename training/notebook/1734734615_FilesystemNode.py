import os


class FilesystemNode:
    def __init__(self, path, index, depth=1):
        self.path = os.path.abspath(path)
        self.index = index
        self.depth = depth
        self.type_id = 0 if os.path.isdir(path) else 1
