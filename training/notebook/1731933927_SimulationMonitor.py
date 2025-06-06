import time
import threading
import Alpha
from Alpha.src.simulation import ThreadManager, CornerstoneShell

class SimulationMonitor:
    """
    Monitors the state of a CornerstoneShell object.
    """
    def __init__(self, thread_manager, cornerstone_shell):
        self.thread_manager = thread_manager
        self.cornerstone_shell = cornerstone_shell

    def monitor(self, duration=5):
        """
        Actively monitor the state of the cornerstone shell for a given duration.
        
        Args:
            duration (int): Duration to monitor in seconds.
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            object_id = self.cornerstone_shell.object_id

            # Try to acquire the lock for monitoring
            current_token = self.thread_manager.tokens.get(object_id, None)
            if current_token and self.thread_manager.acquire_lock(object_id, current_token):
                try:
                    with self.thread_manager.locks[object_id]:
                        position = self.cornerstone_shell.position.tolist()
                        orientation = self.cornerstone_shell.orientation.tolist()
                        print(
                            f"[Monitor] Object {object_id}: Position = {position}, Orientation = {orientation}"
                        )
                finally:
                    # Release the lock
                    self.thread_manager.release_lock(object_id)
            else:
                print(f"[Monitor] Unable to acquire lock for Object {object_id}.")
            
            time.sleep(0.5)  # Monitor at 2 Hz


def main():
    """
    Main function for setting up the simulation and monitoring the cornerstone shell.
    """
    # Set up the ThreadManager
    thread_manager = ThreadManager()

    # Initialize a CornerstoneShell
    object_id = "cornerstone_shell_1"
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    machining_tolerances = {'+X': 0.01, '-X': 0.01, '+Y': 0.01, '-Y': 0.01, '+Z': 0.01, '-Z': 0.01}

    cornerstone_shell = CornerstoneShell(
        object_id=object_id,
        position=position,
        orientation=orientation,
        machining_tolerances=machining_tolerances,
        thread_manager=thread_manager
    )

    # Set up the SimulationMonitor
    monitor = SimulationMonitor(thread_manager, cornerstone_shell)

    # Run updates in a separate thread
    def update_simulation():
        while True:
            dt = 0.1  # Simulate updates every 0.1 seconds
            cornerstone_shell.update(dt)
            time.sleep(dt)

    update_thread = threading.Thread(target=update_simulation, daemon=True)
    update_thread.start()

    # Start monitoring
    print("[Main] Starting simulation monitor...")
    monitor.monitor(duration=10)  # Monitor for 10 seconds


if __name__ == "__main__":
    main()
