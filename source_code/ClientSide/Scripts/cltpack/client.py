from .train import _main, test
from .globals import MODEL, LOGS_PATH
from collections import OrderedDict
from typing import List, Optional
from os import path
import torch
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

class Client(Client):
    """
    Client class for federated learning.
    """
    

    def __init__(self, client_id, train_loader, valid_loader, distribution, epochs) :
        self.client_id = client_id
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.distribution = distribution
        self.epochs = epochs
        self.__round = 0

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = MODEL.get_parameters()
        
        logs_path = path.join(LOGS_PATH, f"local_weights.npz")
        numpy.savez(logs_path, *parameters)
        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(parameters)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )


    def set_parameters(self, ins: FitIns|EvaluateIns):
        """
        Set the parameters of the model.
        
        Args:
            parameters (list): List of model parameters.
        """
        # Deserialize parameters to NumPy ndarray's
        parameters = parameters_to_ndarrays(ins.parameters)
        MODEL.set_parameters(parameters)
    
    def fit(self, ins: FitIns) -> FitRes:
        """
        
        Args:
            parameters (list): List of model parameters.
            config: Configuration for the model.
        
        Returns:
            tuple: A tuple containing updated parameters, length of training data, and an empty dictionary.
        """
        print(f"Fit, config: {ins.config}")

        # Update local model, train, get updated parameters
        self.set_parameters(ins)
        self.__round, loss, accuracy = _main(self.__round, self.client_id, self.train_loader, self.epochs)
        
        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(MODEL.get_parameters())
    
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_loader),
            metrics={"loss": float(loss),
                     "accuracy": float(accuracy),
                    },
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Evaluate, config: {ins.config}")
        
        self.set_parameters(ins)
        loss, accuracy = test(self.valid_loader)
        print(print(f"Evaluation >> Loss: {loss:.4f} | Accuracy: {accuracy:.4f}"))
        
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valid_loader),
            metrics={"loss": float(loss),
                     "accuracy": float(accuracy)
                    },
        )



class Agent_Algorithm :
    def __init__ (self, client_id, train_loader, valid_loader, distribution=None, epochs=1) :
        self.client = Client(client_id, train_loader, valid_loader, distribution, epochs)
    
    def __call__(self, server_address = '127.0.0.1', port = 1234) :
        print( "\n" + "Starting communication...")
        start_client(server_address=f"{server_address}:{port}", client=self.client.to_client())
        print("Ending communication..." + "\n")

