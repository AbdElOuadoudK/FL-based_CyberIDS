from os import environ
environ["RAY_DEDUP_LOGS_ALLOW_REGEX"] = r"Training\s>>\sEpoch:\s\d+\s\|\sLoss:\s[\d\.e+-]+\s\|\sAccuracy:\s[\d\.]+ %"
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
from .dataformat import TrainDataset, load_test_data, TestDataset, load_train_data
from .neuralnet import NeuralNet
from .inference import  train_model, validate_model, test_model
from .server_kit.server import Server_Algorithm
import torch
from .client_kit.client import TrainingAgent #, Agent_Algorithm
from .globals import (
    NUM_CLIENTS,
    BATCH_SIZE,
    DEVICE,
    GLOBALMODEL,
    DATA_PATH,
    LOGS_PATH,
    WEIGHTS,
    NUM_CORES,
    LOAD_TRAIN_DATA_KWARGS
)

torch.set_num_threads(NUM_CORES)

# Configure logger for the project
configure(identifier="FLProjectExperiment", filename=path.join(LOGS_PATH, "server_logs.txt"))

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Disable INFO-level logging
#logging.disable(logging.INFO)

# Exported symbols
__all__ = [
    "TrainDataset",
    "load_test_data",
    "NeuralNet",
    "test_model",
    "Server_Algorithm",
    "NUM_CLIENTS",
    "BATCH_SIZE",
    "DEVICE",
    "GLOBALMODEL",
    "DATA_PATH",
    "LOGS_PATH",
    "WEIGHTS",
    "NUM_CORES",
    "TestDataset",
    "load_train_data",
    "train_model",
    "validate_model",
    "TrainingAgent",
    "Agent_Algorithm",
    "LOAD_TRAIN_DATA_KWARGS",
]