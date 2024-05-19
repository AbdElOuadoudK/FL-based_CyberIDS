"""
Initialization module for the client package.

This module imports key components and sets up initial configurations such as warnings and logging.

By @Ouadoud
"""

from .dataformat import Dataset, load_train_data
from .neuralnet import NeuralNet
from .train import _main, test
from .client import Client, Agent_Algorithm
from .globals import LOAD_TRAIN_DATA_KWARGS, DEVICE, MODEL
import warnings
import logging

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Disable logging for INFO level and below
logging.disable(logging.INFO)

__all__ = [
    "Dataset",
    "load_train_data",
    "NeuralNet",
    "_main",
    "test",
    "Client",
    "Agent_Algorithm",
    "LOAD_TRAIN_DATA_KWARGS",
    "DEVICE",
    "MODEL"
]
