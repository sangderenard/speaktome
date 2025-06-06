# logging_config.py

import os
import logging

# Create a 'logs' directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logger to write to a file instead of the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/simulation.log"),
    ],
)
