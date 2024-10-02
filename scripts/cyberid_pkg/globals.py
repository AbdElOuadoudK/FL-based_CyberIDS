"""
This module defines global constants, configuration and sets up initialization used across the project.

By @Ouadoud
"""

from torch import cuda, device
from .neuralnet import NeuralNet
from numpy import load
from os import path, getcwd
import torch

# Number of clients in federated learning
NUM_CLIENTS = 2

# Batch size for data loaders
BATCH_SIZE = 8192

# Device configuration: Use GPU if available, otherwise fallback to CPU
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# Number of CPU cores to use for threading
NUM_CORES = 4
torch.set_num_threads(NUM_CORES)

# Initialize the global neural network model and move it to the specified device
GLOBALMODEL = NeuralNet(NUM_CORES).to(DEVICE)

# Define paths for data and logs
main_path = path.dirname(getcwd())
DATA_PATH = path.join(main_path, "Data")
LOGS_PATH = path.join(main_path, "Logs")

try:
    # Load global model weights if available
    arrays = load(path.join(LOGS_PATH, "global_weights"))
    WEIGHTS = [arrays[k] for k in arrays.files]
except FileNotFoundError:
    # If weights file is not found, get the initial parameters from the model
    WEIGHTS = GLOBALMODEL.get_parameters()


# Configuration for loading training data
LOAD_TRAIN_DATA_KWARGS = {
    "size": 1.0,
    "valid_rate": 0.5,
    "batch_size": 1024,
    "sparse_y": True
}