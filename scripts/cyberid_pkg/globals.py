
"""
This module defines global constants, configuration and sets up initialization used across the project.

By @Ouadoud
"""

from torch import device, set_num_threads
from torch.cuda import is_available
from .learning_kit.neuralnet import NeuralNet
from numpy import load
from os.path import join, dirname
from os import getcwd

# Number of clients in federated learning
NUM_CLIENTS = 2

# Batch size for data loaders
BATCH_SIZE = 512

# Device configuration: Use GPU if available, otherwise fallback to CPU
DEVICE = device("cuda" if is_available() else "cpu")

# Number of CPU cores to use for threading
NUM_CORES = .5

# Initialize the global neural network model and move it to the specified device
GLOBALMODEL = NeuralNet(NUM_CORES).to(DEVICE)

# Define paths for data and logs
main_path = dirname(getcwd())
DATA_PATH = join(main_path, "data")
LOGS_PATH = join(main_path, "logs")

try:
    # Load global model weights if available
    arrays = load(join(LOGS_PATH, "global_weights"))
    WEIGHTS = [arrays[k] for k in arrays.files]
except FileNotFoundError:
    # If weights file is not found, get the initial parameters from the model
    WEIGHTS = GLOBALMODEL.get_parameters()

NUM_ROUNDS = 5
NUM_EPOCHS = 3

# Configuration for loading training data
LOAD_TRAIN_DATA_KWARGS = {
    "size": 1.0,
    "valid_rate": 0.5,
    "batch_size": 1024,
    "sparse_y": True
}