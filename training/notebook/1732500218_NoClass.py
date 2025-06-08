# main.py

import logging
from Primitives.threadmanager import ThreadManager
from Primitives.cornerstoneshell import CornerstoneShell

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    thread_manager = ThreadManager()
    # Initialize CornerstoneShell
    machining_tolerances = {'+X': 0.01, '-X': 0.01, '+Y': 0.01, '-Y': 0.01, '+Z': 0.01, '-Z': 0.01}
    cornerstone_shell = CornerstoneShell('cornerstone_shell', [0, 0, 0], [0, 0, 0], machining_tolerances, thread_manager)

    import time

    # Simulation loop
    dt = 0.1  # Time step in seconds
    while True:  # Run 100 time steps
        cornerstone_shell.update(dt)

        # Process input events from the ComputerScreen
        input_events = cornerstone_shell.devices['screen'].get_input_events()
        for event in input_events:
            # Handle input events as needed
            pass

        time.sleep(dt)

    # Stop the ComputerScreen thread at the end
    cornerstone_shell.devices['screen'].stop()
