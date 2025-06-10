# framebuffer.py
import threading
import numpy as np

class AsciiFrameBuffer:
    def __init__(self, shape: tuple[int, int]):
        self.lock = threading.Lock()
        self.shape = shape
        self.buffer_render = np.full(shape, ' ', dtype='<U1')
        self.buffer_next = np.full(shape, ' ', dtype='<U1')
        self.buffer_display = np.full(shape, ' ', dtype='<U1')

    def update_render(self, new_data: np.ndarray):
        with self.lock:
            np.copyto(self.buffer_render, new_data)

    def get_diff_and_promote(self):
        with self.lock:
            np.copyto(self.buffer_next, self.buffer_render)
        diff = self.buffer_next != self.buffer_display
        coords = np.argwhere(diff)
        updated = [(y, x, self.buffer_next[y, x]) for y, x in coords]
        np.copyto(self.buffer_display, self.buffer_next)
        return updated
