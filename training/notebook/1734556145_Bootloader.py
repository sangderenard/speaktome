import os
import yaml
import importlib.util
import inspect
import argparse

class Bootloader:
    def __init__(self, system_root):
        self.system_root = os.path.abspath(system_root)
        self.config_data = {}
        self.loaded_modules = {}
        self.errors = []
        self.debug = False

    def run_post(self):
        required_dirs = ["fields", "constants", "operators", "nodes", "aesthetics", "system"]
        for dir_name in required_dirs:
            dir_path = os.path.join(self.system_root, dir_name)
            if not os.path.exists(dir_path):
                choice = input(f"Directory '{dir_name}' is missing. Create it? (yes/no/all): ").lower()
                if choice in ["yes", "all"]:
                    os.makedirs(dir_path, exist_ok=True)
                else:
                    self.errors.append(f"Missing required directory: {dir_name}")
        if self.errors:
            raise FileNotFoundError("POST failed due to missing directories.")

    def load_yaml_files(self, directory):
        yaml_data = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".yaml"):
                    with open(os.path.join(root, file), "r") as f:
                        yaml_data[file] = yaml.safe_load(f)
        return yaml_data

    def load_python_files(self, directory, yaml_context):
        loaded_modules = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    module_name = file.replace(".py", "")
                    file_path = os.path.join(root, file)
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if self.debug:
                        print(f"Loaded module: {module_name}")
                    loaded_modules[module_name] = module
        return loaded_modules

    def initialize(self):
        self.run_post()
        self.config_data["constants"] = self.load_yaml_files(os.path.join(self.system_root, "constants"))
        self.loaded_modules["fields"] = self.load_python_files(os.path.join(self.system_root, "fields"), {})
        print("[BOOTLOADER] Initialization complete.")
        return self.config_data, self.loaded_modules

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--root", default="..", help="Path to the system root directory")
    args.add_argument("--debug", action="store_true", help="Enable debug logging")
    params = args.parse_args()

    bootloader = Bootloader(params.root)
    bootloader.debug = params.debug
    bootloader.initialize()
