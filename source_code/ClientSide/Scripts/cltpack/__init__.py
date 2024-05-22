from .dataformat import Dataset, load_train_data
from .neuralnet import NeuralNet
from .train import _main, test
from .client import Client, Agent_Algorithm
from .globals import (LOAD_TRAIN_DATA_KWARGS, DEVICE, MODEL, DATA_PATH, LOGS_PATH)
import warnings
import logging
from flwr.common.logger import configure
from os import path

configure(identifier="FLProjectExperiment", filename=path.join(LOGS_PATH, "client_logs.txt"))
warnings.simplefilter(action='ignore', category=FutureWarning) # ,DeprecationWarning
warnings.simplefilter(action='ignore', category=DeprecationWarning)
logging.disable(logging.INFO)


__all__ = ["Dataset",
           "load_train_data",
           "NeuralNet",
           "_main",
           "test",
           "Client",
           "Agent_Algorithm",
           "LOAD_TRAIN_DATA_KWARGS",
           "DEVICE",
           "MODEL",
           "DATA_PATH",
           "LOGS_PATH"
          ]
