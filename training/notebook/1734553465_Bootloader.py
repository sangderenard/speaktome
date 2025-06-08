import os
import yaml
import importlib.util
import inspect
import uuid

class Bootloader:
    def __init__(self, system_root=".."):
        """
        Initialize the Bootloader with a system directory root.
        """
        self.system_root = os.path.abspath(system_root)
        self.config_data = {}
        self.loaded_modules = {}
        self.errors = []

    def run_post(self):
        """
        Perform POST (Power-On Self Test) by validating folder structure and components.
        """
        print("[BOOTLOADER] Starting POST...")
        required_dirs = ["fields", "constants", "operators", "nodes", "aesthetics", "system"]
        for dir_name in required_dirs:
            dir_path = os.path.join(self.system_root, dir_name)
            if not os.path.exists(dir_path):
                self.errors.append(f"Missing required directory: {dir_name}")
        if self.errors:
            for err in self.errors:
                print(f"[ERROR] {err}")
            raise FileNotFoundError("POST failed due to missing directories.")
        print("[BOOTLOADER] POST successful. All required directories are present.")

    def load_yaml_files(self, directory):
        """
        Recursively load all YAML files from the specified directory.
        """
        yaml_data = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".yaml") or file.endswith(".yml"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as yaml_file:
                        try:
                            data = yaml.safe_load(yaml_file)
                            relative_path = os.path.relpath(file_path, directory)
                            yaml_data[relative_path] = data
                            print(f"[YAML] Loaded: {relative_path}")
                        except yaml.YAMLError as e:
                            self.errors.append(f"YAML parse error in {file_path}: {e}")
        return yaml_data

    def load_python_files(self, directory, yaml_context):
        """
        Load Python files programmatically, passing corresponding YAML context as input.
        """
        loaded_modules = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    module_name = file.replace(".py", "")
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        # Inject YAML context if available
                        if module_name in yaml_context:
                            module.context = yaml_context[module_name]
                        loaded_modules[module_name] = module
                        print(f"[PYTHON] Loaded: {file_path}")
                    except Exception as e:
                        self.errors.append(f"Error loading Python module {file}: {e}")
        return loaded_modules

    def initialize(self):
        """
        Full initialization pipeline: load YAML and Python files, validate graph readiness.
        """
        try:
            self.run_post()
            print("[BOOTLOADER] Loading constants...")
            constants_path = os.path.join(self.system_root, "constants")
            self.config_data["constants"] = self.load_yaml_files(constants_path)

            print("[BOOTLOADER] Loading fields...")
            fields_path = os.path.join(self.system_root, "fields")
            self.config_data["fields"] = self.load_yaml_files(fields_path)
            self.loaded_modules["fields"] = self.load_python_files(fields_path, self.config_data["fields"])

            print("[BOOTLOADER] Loading operators...")
            operators_path = os.path.join(self.system_root, "operators")
            self.config_data["operators"] = self.load_yaml_files(operators_path)
            self.loaded_modules["operators"] = self.load_python_files(operators_path, self.config_data["operators"])

            print("[BOOTLOADER] Loading nodes...")
            nodes_path = os.path.join(self.system_root, "nodes")
            self.config_data["nodes"] = self.load_yaml_files(nodes_path)
            self.loaded_modules["nodes"] = self.load_python_files(nodes_path, self.config_data["nodes"])

            print("[BOOTLOADER] Initialization complete.")
            return self.config_data, self.loaded_modules

        except Exception as e:
            print(f"[FATAL] Bootloader failed: {e}")
            exit(1)


if __name__ == "__main__":
    system_dir = os.path.join(os.getcwd(), "..")
    bootloader = Bootloader(system_root=system_dir)
    config_data, loaded_modules = bootloader.initialize()
    print("\n--- BOOTLOADER OUTPUT ---")
    print("Loaded Configurations:")
    print(config_data.keys())
    print("Loaded Modules:")
    print(loaded_modules.keys())
