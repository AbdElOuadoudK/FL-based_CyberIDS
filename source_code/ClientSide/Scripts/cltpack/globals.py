"""
globals.py

This module defines global constants and configurations used across the project.

By @Ouadoud.
"""

from torch import cuda, device
from os import path, getcwd

# Configuration for loading training data
LOAD_TRAIN_DATA_KWARGS = {
    "size": 1.0,
    "valid_rate": 0.5,
    "batch_size": 1024,
    "sparse_y": True
}

# Device configuration: Use GPU if available, otherwise fallback to CPU
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# Define paths for data and logs
main_path = path.dirname(getcwd())
DATA_PATH = path.join(main_path, "Data")
LOGS_PATH = path.join(main_path, "Logs")

# Number of CPU cores to be used
NUM_CORES = 4
