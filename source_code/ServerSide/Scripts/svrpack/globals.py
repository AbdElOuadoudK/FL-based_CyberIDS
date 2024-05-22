"""
Global configuration and initialization for the neural network.

This module sets up the global constants, device configuration, and initializes
the global neural network model with weights.

By @Ouadoud
"""

from torch import cuda, device
from .neuralnet import NeuralNet
from numpy import load
from os import path, getcwd

# Number of clients in federated learning
NUM_CLIENTS = 2

# Batch size for data loaders
BATCH_SIZE = 1024

# Set device to GPU if available, otherwise use CPU
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# Initialize the global neural network model and move it to the specified device
GLOBALMODEL = NeuralNet().to(DEVICE)

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
