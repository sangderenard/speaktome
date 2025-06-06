import threading
from queue import Queue
import torch
from hotrodprintingpress import GrandPrintingPress

class PrintShop:
    """
    **PrintShop**
    Manages the workflow of the Grand Printing Press, including inking, cutting, and other
    post-processing stages. Supports dynamic rule-based queues for blocking/non-blocking
    operation and parallel activity under heavy load.
    """

    def __init__(self, press: GrandPrintingPress, max_cache_size: int = 10):
        """
        Initialize the PrintShop with the Grand Printing Press.

        Args:
            press (GrandPrintingPress): Instance of the Grand Printing Press.
            max_cache_size (int): Maximum cache size for each stage.
        """
        self.press = press
        self.input_queue = Queue(maxsize=max_cache_size)
        self.press_cache = Queue(maxsize=max_cache_size)
        self.inker_cache = Queue(maxsize=max_cache_size)
        self.output_queue = Queue(maxsize=max_cache_size)

        # Worker threads
        self.threads = {
            "press_worker": threading.Thread(target=self.press_worker, daemon=True),
            "inker_worker": threading.Thread(target=self.inker_worker, daemon=True),
            "output_worker": threading.Thread(target=self.output_worker, daemon=True),
        }

        for thread in self.threads.values():
            thread.start()

    def submit_job(self, text_generator, num_sheets: int):
        """
        Submit a job to the PrintShop.

        Args:
            text_generator (generator): Yields text to print.
            num_sheets (int): Number of sheets to produce.
        """
        self.input_queue.put((text_generator, num_sheets))

    def press_worker(self):
        """Processes input text and generates raw grand sheets."""
        while True:
            text_generator, num_sheets = self.input_queue.get()
            for _ in range(num_sheets):
                text = next(text_generator, None)
                if text is None:
                    break
                sheet = self.press.print_text(
                    text, font_path="arial.ttf", font_size=24
                )
                self.press_cache.put(sheet)

    def inker_worker(self):
        """Applies inking and material qualities to raw grand sheets."""
        while True:
            sheet = self.press_cache.get()
            # Placeholder for inking/material properties
            inked_sheet = sheet * 0.9  # Simulate ink darkening
            self.inker_cache.put(inked_sheet)

    def output_worker(self):
        """Cuts and assembles the inked sheets into final outputs."""
        while True:
            sheet = self.inker_cache.get()
            # Placeholder for cutting/assembling operations
            final_sheet = sheet  # For now, pass through unmodified
            self.output_queue.put(final_sheet)

    def get_output(self):
        """
        Retrieve the final processed sheet.

        Returns:
            torch.Tensor: Final output tensor.
        """
        return self.output_queue.get()

    def is_busy(self):
        """
        Check if the PrintShop is actively processing jobs.

        Returns:
            bool: True if there are pending jobs in any queue.
        """
        return (
            not self.input_queue.empty()
            or not self.press_cache.empty()
            or not self.inker_cache.empty()
            or not self.output_queue.empty()
        )

import itertools
from hotrodprintingpress import GrandPrintingPress
from PIL import Image
import numpy as np

def example_text_generator():
    """Generates example text for the printing press."""
    lines = ["Line 1: Grand Printing Press Demo", "Line 2: Future-Proofed Pipeline"]
    return iter(itertools.cycle(lines))

if __name__ == "__main__":
    # Initialize the Grand Printing Press
    press = GrandPrintingPress(page_width=800, page_height=1200, margin=50)

    # Initialize the PrintShop
    shop = PrintShop(press, max_cache_size=5)

    # Submit a print job
    text_gen = example_text_generator()
    shop.submit_job(text_gen, num_sheets=10)

    # Retrieve and display outputs
    for _ in range(10):
        output = shop.get_output()
        image = Image.fromarray(output.numpy().astype(np.uint8), mode="L")
        image.show()

    # Check the PrintShop status
    print("PrintShop busy:", shop.is_busy())
