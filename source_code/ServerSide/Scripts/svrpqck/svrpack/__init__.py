"""
Initialization module for the federated learning project.

This module initializes various components of the federated learning project,
including datasets, neural network models, training functions, server configuration,
and global variables.

By @Ouadoud
"""

import warnings
import logging
from os import path
from flwr.common.logger import configure

from .dataformat import Dataset, load_test_data
from .neuralnet import NeuralNet
from .inference import test
from .server import Server_Algorithm
from .globals import (
    NUM_CLIENTS,
    BATCH_SIZE,
    DEVICE,
    GLOBALMODEL,
    DATA_PATH,
    LOGS_PATH,
    WEIGHTS,
    NUM_CORES
)

# Configure logger for the project
configure(identifier="FLProjectExperiment", filename=path.join(LOGS_PATH, "server_logs.txt"))

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Disable INFO-level logging
logging.disable(logging.INFO)

# Exported symbols
__all__ = [
    "Dataset",
    "load_test_data",
    "NeuralNet",
    "test",
    "Server_Algorithm",
    "NUM_CLIENTS",
    "BATCH_SIZE",
    "DEVICE",
    "GLOBALMODEL",
    "DATA_PATH",
    "LOGS_PATH",
    "WEIGHTS",
    "NUM_CORES"
]
