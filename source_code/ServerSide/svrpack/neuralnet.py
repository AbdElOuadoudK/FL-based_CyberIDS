"""
Neural network model definition for classification.

This module defines a simple neural network model for classification tasks,
including methods to get and set model parameters.

By @Ouadoud
"""

from collections import OrderedDict
from typing import List
import numpy
import torch

class NeuralNet(torch.nn.Module):
    """
    A simple neural network model for classification.

    By @Ouadoud
    """

    def __init__(self, num_cores=4):
        """
        Initialize the neural network model.

        Args:
            num_cores (int): Number of CPU cores to use for threading.

        By @Ouadoud
        """
        torch.set_num_threads(num_cores)
        super(NeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(50, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.

        By @Ouadoud
        """
        return self.sequential(x)

    def get_parameters(self) -> List[numpy.ndarray]:
        """
        Get model parameters as a list of numpy arrays.

        Returns:
            List[numpy.ndarray]: List of model parameters.

        By @Ouadoud
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
    def set_parameters(self, parameters: List[numpy.ndarray]) -> None:
        """
        Set model parameters from a list of numpy arrays.

        Args:
            parameters (List[numpy.ndarray]): List of model parameters.

        By @Ouadoud
        """
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
