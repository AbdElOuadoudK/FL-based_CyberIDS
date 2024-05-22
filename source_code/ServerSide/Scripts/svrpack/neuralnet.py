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
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    A simple neural network model for classification.

    By @Ouadoud
    """
    def __init__(self):
        """
        Initialize the neural network model.

        By @Ouadoud
        """
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.sequential = nn.Sequential(
            nn.Linear(122, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
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
        return self.sequential(self.dropout(x))

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
