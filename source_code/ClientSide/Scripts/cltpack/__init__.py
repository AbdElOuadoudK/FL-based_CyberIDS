"""
__init__.py

This module initializes the package and imports all necessary components for the federated learning client implementation. 

By @Ouadoud.
"""

from .dataformat import Dataset, load_train_data
from .neuralnet import NeuralNet
from .inference import train, test
from .client import Client, Agent_Algorithm
from .globals import (LOAD_TRAIN_DATA_KWARGS, DEVICE, DATA_PATH, LOGS_PATH, NUM_CORES)
import warnings
import logging
import torch

# Set the number of threads for PyTorch
torch.set_num_threads(NUM_CORES)

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
logging.disable(logging.INFO)

# Define the list of publicly accessible objects from the package
__all__ = ["Dataset",
           "load_train_data",
           "NeuralNet",
           "train",
           "test",
           "Client",
           "Agent_Algorithm",
           "LOAD_TRAIN_DATA_KWARGS",
           "DEVICE",
           "DATA_PATH",
           "LOGS_PATH",
           "NUM_CORES"
          ]
