from torch import cuda, device
from .neuralnet import NeuralNet

"""
Global variables for the project.

This module sets up global configurations and variables such as device settings and model initialization.

By @Ouadoud
"""

# Configuration for loading training data
LOAD_TRAIN_DATA_KWARGS = {
    "size": 1.0,
    "valid_rate": 0.5,
    "batch_size": 1024,
    "sparse_y": True
}

# Determine the device to use (CUDA if available, else CPU)
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# Initialize the model and move it to the appropriate device
MODEL = NeuralNet().to(DEVICE)
