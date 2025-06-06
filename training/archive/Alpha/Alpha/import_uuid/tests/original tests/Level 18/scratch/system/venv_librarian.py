import os
import sys
import subprocess
import json

class VenvLibrarian:
    """
    Handles switching virtual environments by restarting the process
    with the appropriate venv and reloading the graph state.
    """

    def __init__(self, venv_root="venvs"):
        self.venv_root = venv_root
        self.backup_file = "graph_backup.json"

    def list_venvs(self):
        """List all available virtual environments."""
        return [name for name in os.listdir(self.venv_root) if os.path.isdir(os.path.join(self.venv_root, name))]

    def switch_venv(self, venv_name, graph_state):
        """
        Switch to a different virtual environment and restart the process.

        Args:
            venv_name (str): Name of the target venv.
            graph_state (dict): Serialized state of the graph to reload after switching.
        """
        venv_path = os.path.join(self.venv_root, venv_name, "bin", "python")

        if not os.path.exists(venv_path):
            raise FileNotFoundError(f"Python executable for venv '{venv_name}' not found at {venv_path}.")

        # Save the graph state to a file
        with open(self.backup_file, "w") as f:
            json.dump(graph_state, f)
        print(f"Graph state saved to {self.backup_file}.")

        # Restart the process with the new venv
        print(f"Switching to venv '{venv_name}'...")
        subprocess.run([venv_path, sys.argv[0], "--reload", self.backup_file])

        # Exit the current process
        sys.exit(0)
