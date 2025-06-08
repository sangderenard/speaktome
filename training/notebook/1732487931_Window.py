# window.py

from Primitives.physicalobject import PhysicalObject

class Window(PhysicalObject):
    """
    Represents the window object within the CornerstoneShell.
    Projects the MitsubaShell rendering as a texture.
    """

    def __init__(self, object_id, position, orientation, thread_manager):
        super().__init__(object_id, position, orientation, thread_manager)
        # Additional initialization if needed
