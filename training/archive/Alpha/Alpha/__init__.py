import os
import glob
import logging
import sys
from importlib import import_module

# Initialize Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Alpha")

# Tee stdout to both console and a rotating log
class RotatingLog:
    def __init__(self, file_path, max_lines=1000):
        self.file_path = file_path
        self.max_lines = max_lines
        self.line_count = 0
        self.file = open(file_path, "w", buffering=1)

    def write(self, message):
        if self.line_count >= self.max_lines:
            self.file.seek(0)
            self.file.truncate(0)
            self.line_count = 0
        if message.strip():  # Only count non-empty lines
            self.line_count += 1
        self.file.write(message)

    def flush(self):
        self.file.flush()

log_file_path = "alpha_log.txt"
tee = RotatingLog(log_file_path)

# Redirect stdout
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

sys.stdout = Tee(sys.stdout, tee)

logger.info("Initializing Alpha package...")

# Dynamically Import All Submodules from Alpha.src
__all__ = []  # Collect all public symbols
package_dir = os.path.join(os.path.dirname(__file__), "src")
module_files = glob.glob(os.path.join(package_dir, "**", "*.py"), recursive=True)

for file_path in module_files:
    module_name = os.path.relpath(file_path, package_dir).replace(os.sep, ".")[:-3]
    if module_name.startswith("_"):
        continue  # Skip private modules
    full_module_name = f"Alpha.src.{module_name}"
    try:
        module = import_module(full_module_name)
        if hasattr(module, "__all__"):
            __all__.extend(module.__all__)  # Add public symbols from submodule
        else:
            __all__.append(module_name)  # Add module itself if no __all__
    except ModuleNotFoundError as e:
        logger.error(f"Failed to import {full_module_name}: {e}")
