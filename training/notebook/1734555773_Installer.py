import os
import yaml
import shutil
import argparse

class Installer:
    def __init__(self):
        """
        Initialize the installer. Determines input paths from command line, YAML, or user input.
        """
        self.args = self.parse_arguments()
        self.yaml_path = self.get_yaml_path()
        self.load_yaml()

        # Determine source and destination directories
        self.source_dir = self.get_directory(
            "installation_source_directory",
            "Source directory (absolute path): "
        )
        self.destination_dir = self.get_directory(
            "installation_destination_directory",
            "Destination directory (absolute path): "
        )
        self.overwrite_all = False

        # Validate source directory
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"[ERROR] Source directory does not exist: {self.source_dir}")

        # Prevent source directory being overwritten
        if self.source_dir.startswith(self.destination_dir):
            raise ValueError(
                "[ERROR] Source directory is located within the destination directory. "
                "This configuration is forbidden to prevent overwriting source files."
            )

    def parse_arguments(self):
        """
        Parse command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Workspace Installer")
        parser.add_argument("--yaml", type=str, help="Path to the YAML configuration file")
        parser.add_argument("--source", type=str, help="Path to the source directory")
        parser.add_argument("--destination", type=str, help="Path to the destination directory")
        return parser.parse_args()

    def get_yaml_path(self):
        """
        Determine the YAML file path using command-line arguments, fallback to default.
        """
        if self.args.yaml:
            yaml_path = os.path.abspath(self.args.yaml)
        else:
            yaml_path = os.path.abspath("installer.yaml")
        
        if not os.path.exists(yaml_path):
            yaml_path = input("Enter the path to the YAML configuration file: ").strip()
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"[ERROR] YAML configuration not found: {yaml_path}")
        return yaml_path

    def load_yaml(self):
        """
        Load the installer configuration from the YAML file.
        """
        with open(self.yaml_path, "r") as f:
            self.structure = yaml.safe_load(f)

    def get_directory(self, key, prompt):
        """
        Retrieve a directory path using command-line arguments, YAML, or interactive input.
        """
        # Command-line argument takes precedence
        if key == "installation_source_directory" and self.args.source:
            return os.path.abspath(self.args.source)
        if key == "installation_destination_directory" and self.args.destination:
            return os.path.abspath(self.args.destination)

        # Next, check the YAML file
        if key in self.structure:
            return os.path.abspath(self.structure.pop(key)[0])

        # Fallback to interactive input
        while True:
            path = input(prompt).strip()
            if os.path.exists(path):
                return os.path.abspath(path)
            print(f"[ERROR] Invalid path: {path}")

    def ensure_directory(self, path):
        """
        Ensure a directory exists.
        """
        os.makedirs(path, exist_ok=True)

    def confirm_overwrite(self, path):
        """
        Prompt user for overwrite confirmation.
        """
        if self.overwrite_all:
            return True

        print(f"[WARNING] File already exists: {path}")
        while True:
            response = input("Overwrite? (yes/no/all/cancel): ").strip().lower()
            if response == "yes":
                return True
            elif response == "no":
                return False
            elif response == "all":
                self.overwrite_all = True
                return True
            elif response == "cancel":
                print("[INSTALLER] Installation cancelled.")
                exit(0)
            else:
                print("Invalid response. Please enter 'yes', 'no', 'all', or 'cancel'.")

    def copy_file(self, relative_path):
        """
        Copy a file from the source directory to the destination directory.
        """
        source_path = os.path.join(self.source_dir, relative_path)
        target_path = os.path.join(self.destination_dir, relative_path)

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Check for conflicts and handle overwrites
        if os.path.exists(target_path):
            if not self.confirm_overwrite(target_path):
                print(f"[INSTALLER] Skipped: {relative_path}")
                return

        # Copy file
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"[INSTALLER] Copied: {relative_path}")
        else:
            raise FileNotFoundError(f"[ERROR] Missing file in source: {source_path}")

    def process_structure(self, structure, base_path=""):
        """
        Recursively process the directory and file structure from the YAML.
        Skip any directories or files related to the source/destination definitions.
        """
        for key, value in structure.items():
            # Skip source/destination directories
            if key in ["installation_source_directory", "installation_destination_directory"]:
                continue

            current_path = os.path.join(base_path, key)

            if isinstance(value, list):
                for item in value:
                    item_path = os.path.join(current_path, item)
                    if item.endswith("/"):
                        # Directory
                        self.ensure_directory(os.path.join(self.destination_dir, item_path))
                    else:
                        # File
                        self.copy_file(item_path)
            elif isinstance(value, dict):
                # Process nested structure
                self.process_structure(value, current_path)

    def install(self):
        """
        Perform the full installation process.
        """
        print("[INSTALLER] Starting installation...")
        try:
            self.process_structure(self.structure)
            print("[INSTALLER] Installation complete.")
        except Exception as e:
            print(f"[ERROR] Installation failed: {e}")

if __name__ == "__main__":
    installer = Installer()
    installer.install()
