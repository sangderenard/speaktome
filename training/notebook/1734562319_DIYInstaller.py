import os
import yaml

class DIYInstaller:
    def __init__(self, yaml_path, install_dir):
        """
        Initialize DIY installer.
        :param yaml_path: Path to the installer.yaml file.
        :param install_dir: Installation root directory.
        """
        self.yaml_path = yaml_path
        self.install_dir = os.path.abspath(install_dir)
        self.missing_files = []
        self.missing_directories = []

    def load_yaml(self):
        """Load the YAML configuration to get the directory structure."""
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"Installer configuration file not found: {self.yaml_path}")
        with open(self.yaml_path, "r") as f:
            return yaml.safe_load(f)

    def check_missing(self, structure, base_path=""):
        """
        Recursively check for missing files and directories in the YAML-defined structure.
        """
        for key, value in structure.items():
            current_path = os.path.join(base_path, key)
            abs_path = os.path.join(self.install_dir, current_path)

            if isinstance(value, list):
                # Handle files and directories in a list
                for item in value:
                    if isinstance(item, str):
                        target = os.path.join(abs_path, item)
                        if item.endswith("/"):  # Directory
                            if not os.path.exists(target):
                                self.missing_directories.append(target)
                        else:  # File
                            if not os.path.exists(target):
                                self.missing_files.append(target)
            elif isinstance(value, dict):
                # Recurse into subdirectories
                self.check_missing(value, current_path)
            else:
                raise ValueError(f"Unexpected value in structure: {value}")

    def create_directories(self):
        """Create missing directories automatically."""
        for directory in self.missing_directories:
            os.makedirs(directory, exist_ok=True)
            print(f"[DIRECTORY CREATED] {directory}")
        self.missing_directories.clear()

    def diy_text_editor(self, file_path):
        """
        A simple line-by-line text editor with whiteout (undo) support.
        :param file_path: Path to the file being edited.
        """
        print(f"\n[DIY EDITOR] Creating file: {file_path}")
        print("Type each line and press ENTER to commit it.")
        print("Type '/whiteout' to delete the last line.")
        print("Type '/finish' to save and exit.\n")

        lines = []
        while True:
            line = input("> ").rstrip("\n")
            if line == "/finish":
                break
            elif line == "/whiteout":
                if lines:
                    removed = lines.pop()
                    print(f"[WHITEOUT] Removed line: {removed}")
                else:
                    print("[WHITEOUT] No lines to remove.")
            else:
                lines.append(line)

        # Save the file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[SAVED] File created: {file_path}")

    def run(self):
        """
        Main entry point for the DIY installer.
        """
        print("[DIY INSTALLER] Scanning for missing files and directories...")
        structure = self.load_yaml()
        self.check_missing(structure)
        
        # Automatically create directories
        if self.missing_directories:
            print(f"[DIY INSTALLER] Creating {len(self.missing_directories)} missing directories...")
            self.create_directories()
        
        if not self.missing_files:
            print("[DIY INSTALLER] All files are present. Nothing to do.")
            return

        print(f"[DIY INSTALLER] {len(self.missing_files)} missing files detected:")
        for idx, file in enumerate(self.missing_files, 1):
            print(f"{idx}. {file}")

        print("\nSelect files to create manually. Enter the number or 'all' to create all.")
        while self.missing_files:
            selection = input("File number to create (or 'all' to do all): ").strip()

            if selection.lower() == "all":
                for file_path in self.missing_files.copy():
                    self.diy_text_editor(file_path)
                    self.missing_files.remove(file_path)
            elif selection.isdigit() and 1 <= int(selection) <= len(self.missing_files):
                file_index = int(selection) - 1
                self.diy_text_editor(self.missing_files[file_index])
                self.missing_files.pop(file_index)
            else:
                print("[ERROR] Invalid selection. Try again.")

        print("[DIY INSTALLER] All selected files have been created.")

if __name__ == "__main__":
    import argparse

    # Parse arguments for YAML path and install directory
    parser = argparse.ArgumentParser(description="DIY Installer for missing files.")
    parser.add_argument("--yaml", default="installer.yaml", help="Path to installer.yaml")
    parser.add_argument("--install-dir", default="..", help="Root directory for installation")
    args = parser.parse_args()

    diy_installer = DIYInstaller(yaml_path=args.yaml, install_dir=args.install_dir)
    diy_installer.run()
