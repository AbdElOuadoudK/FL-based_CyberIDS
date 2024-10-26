
"""
client.py

This module defines the client-side implementation for federated learning using the Flower framework. 
It includes the client class and related methods for training, evaluation, and communication with the server.

By @Ouadoud.
"""


from torch import set_num_threads
from typing import List
from os.path import join
from os import listdir
from numpy import savez
from pandas import DataFrame
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
from ..learning_kit.neuralnet import NeuralNet
from ..learning_kit.inference import train_model, validate_model
from ..globals import NUM_CORES, DATA_PATH, CHKPTS_PATH
from ..dataformat import load_train_data


class TrainingAgent(Client):
    """
    TrainingAgent class for federated learning.
        
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
        super().__init__()
        self.client_id = client_id
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.distribution = distribution
        self.epochs = epochs
        self.__model = NeuralNet(num_cores=NUM_CORES)

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
        chkpts_path = join(CHKPTS_PATH, "local_weights.npz")
        savez(chkpts_path, *parameters)
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
        self.__rounds = ins.config.get("rounds")

        self.set_parameters(ins)
        self.__model, metrics_list, metrics_dict = train_model(ins.config["round"], self.__rounds, self.client_id, self.__model, self.train_loader, self.epochs)
        parameters = ndarrays_to_parameters(self.__model.get_parameters())
        status = Status(code=Code.OK, message="Success")
        
        #async
        self.__save_results(self.client_id, metrics_list, f"train_{self.client_id}_logs.csv")
        return FitRes(status=status, parameters=parameters, num_examples=len(self.train_loader), metrics=metrics_dict)

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
        metrics_dict = validate_model(self.__model, self.valid_loader, self.client_id)
        status = Status(code=Code.OK, message="Success")

        #async
        self.__save_results(self.client_id, [metrics_dict], f"valid_{self.client_id}_logs.csv")
        return EvaluateRes(status=status, loss=float(metrics_dict['BCELoss']), num_examples=len(self.valid_loader), metrics=metrics_dict)
    
    def __save_results(self, client_id, metrics, msg):

        chkpts_path = join(CHKPTS_PATH, msg)
        if msg in listdir(CHKPTS_PATH):
            DataFrame(metrics).to_csv(chkpts_path, header=False, index=False, mode='a')
        else:
            DataFrame(metrics).to_csv(chkpts_path, index=False, mode='w') 

        
    def __exception_management(self,):
        pass

