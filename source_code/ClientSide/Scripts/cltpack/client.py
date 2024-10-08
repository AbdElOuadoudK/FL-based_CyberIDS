"""
client.py

This module defines the client-side implementation for federated learning using the Flower framework. 
It includes the client class and related methods for training, evaluation, and communication with the server.

By @Ouadoud.
"""

from .inference import train, test
from .globals import LOGS_PATH
from .neuralnet import NeuralNet
from typing import List
from os import path
import numpy
from flwr.client import Client, start_client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import configure

class Client(Client):
    """
    Client class for federated learning.
        
    By @Ouadoud.
    """
    
    def __init__(self, 
                 client_id: int, 
                 train_loader, 
                 valid_loader, 
                 distribution: dict, 
                 epochs: int):
        """
        Initialize the client with given parameters.

        Args:
            client_id (int): Unique identifier for the client.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            distribution (dict): Distribution dictionary.
            epochs (int): Number of training epochs.
        
        By @Ouadoud.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.distribution = distribution
        self.epochs = epochs
        self.__model = NeuralNet()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Get the current model parameters.

        Args:
            ins (GetParametersIns): Instructions for getting parameters.

        Returns:
            GetParametersRes: Response containing the model parameters.
        
        By @Ouadoud.
        """
        parameters = self.__model.get_parameters()
        logs_path = path.join(LOGS_PATH, "local_weights.npz")
        numpy.savez(logs_path, *parameters)
        parameters = ndarrays_to_parameters(parameters)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)

    def set_parameters(self, ins: FitIns | EvaluateIns):
        """
        Set the parameters of the model.

        Args:
            ins (FitIns | EvaluateIns): Instructions containing parameters.
        
        By @Ouadoud.
        """
        parameters = parameters_to_ndarrays(ins.parameters)
        self.__model.set_parameters(parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """
        Train the model with the given parameters and configuration.

        Args:
            ins (FitIns): Instructions for fitting the model.

        Returns:
            FitRes: Response containing updated parameters and training metrics.
        
        By @Ouadoud.
        """
        if ins.config.get("rounds") is not None:
            self.__rounds = ins.config.get("rounds")
        
        self.set_parameters(ins)
        self.__model, log_entry = train(ins.config["round"], self.__rounds, self.client_id, self.__model, self.train_loader, self.epochs)
        parameters = ndarrays_to_parameters(self.__model.get_parameters())
        status = Status(code=Code.OK, message="Success")
        return FitRes(status=status, parameters=parameters, num_examples=len(self.train_loader), metrics=log_entry)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the model with the given parameters.

        Args:
            ins (EvaluateIns): Instructions for evaluating the model.

        Returns:
            EvaluateRes: Response containing evaluation metrics.
        
        By @Ouadoud.
        """
        self.set_parameters(ins)
        log_entry = test(self.__model, self.valid_loader, self.client_id)
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(status=status, loss=float(log_entry['Loss']), num_examples=len(self.valid_loader), metrics=log_entry)

class Agent_Algorithm:
    """
    Agent_Algorithm class to handle client initialization and communication.

    By @Ouadoud.
    """
    def __init__(self, client_id: int, 
                 train_loader, 
                 valid_loader, 
                 distribution: dict = None, 
                 epochs: int = 1):
        """
        Initialize the agent with the given parameters.

        Args:
            client_id (int): Unique identifier for the client.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            distribution (dict, optional): Distribution dictionary. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 1.
        
        By @Ouadoud.
        """
        configure(identifier="IDS_Learning_Logs", filename=path.join(LOGS_PATH, f"client_{client_id}_logs.txt"))
        self.client = Client(client_id, train_loader, valid_loader, distribution, epochs)
    
    def __call__(self, server_address: str = '127.0.0.1', port: int = 1234):
        """
        Start the client and establish communication with the server.

        Args:
            server_address (str, optional): Server address. Defaults to '127.0.0.1'.
            port (int, optional): Server port. Defaults to 1234.
        
        By @Ouadoud.
        """
        print("\nStarting communication...")
        start_client(server_address=f"{server_address}:{port}", client=self.client)
        print("Ending communication...\n")
