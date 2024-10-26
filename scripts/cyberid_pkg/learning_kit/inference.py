
"""
inference.py

This module provides functions for training and testing a neural network model using PyTorch. 

By @Ouadoud.
"""

from typing import Tuple
from torch import set_num_threads, float32, no_grad
from torch.nn import BCELoss, Module
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os.path import join
from pandas import DataFrame
from ..globals import GLOBALMODEL, DEVICE, NUM_CORES
#from ray import logger as ray_logger
from logging import getLogger
from .metrics import loss_function, confusion_matrix, get_metrics

#set_num_threads(NUM_CORES)


def _train(model: Module,
           train_loader: DataLoader,
           epochs: int = 3) -> tuple:
    """
    Train the neural network model.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        epochs (int, optional): Number of training epochs. Defaults to 3.

    Yields:
        tuple: A tuple containing epoch number, training loss, and metrics (TP, TN, FP, FN). 
        
    By @Ouadoud.
    """
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=5e-2, patience=2)
    
    train_loader.dataset.is_train = True    
    for epoch in range(epochs):
        train_loss = []
        tp, tn, fp, fn = 0, 0, 0, 0
        model.train()
        for items, targets in train_loader:
            items = items.to(float32).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(items)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy().item())

            tp_, tn_, fp_, fn_ = confusion_matrix(outputs, targets)
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_

        train_loss = sum(train_loss) / len(train_loader.dataset)
        scheduler.step(train_loss)
        
        metrics_dict = get_metrics(tp, tn, fp, fn, train_loss)
        yield epoch, metrics_dict

def train_model(round: int,
          rounds: int,
          client_id: int,
          model: Module,
          train_loader: DataLoader,
          epochs: int = 3) -> tuple:
    """
    Main function to train the model and visualize the training process.

    Args:
        round (int): Current training round.
        rounds (int): Total number of training rounds.
        client_id (int): ID of the client.
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        epochs (int, optional): Number of training epochs. Defaults to 3.

    Returns:
        tuple: The trained model and the final log entry. 
        
    By @Ouadoud.
    """   
    getLogger("ray." + "client_" + client_id).info(f'Round: {round}')
    
    metrics_list = []
    for epoch, metrics_dict in _train(model=model, train_loader=train_loader, epochs=epochs):
        
        metrics_list.append(metrics_dict)
        
        loss = "{:.4e}".format(metrics_dict['BCELoss'])
        accuracy = "{:.4f}%".format(metrics_dict['Accuracy'] * 100)

        getLogger("ray." + "client_" + client_id).info(f"Training >> Epoch: {epoch+1} | Loss: {loss} | Accuracy: {accuracy}")

    return model, metrics_list, metrics_dict

def validate_model(model: Module, 
         valid_loader: DataLoader, 
         client_id: int) -> dict:
    """
    Test the neural network model.
    
    Args:
        model (torch.nn.Module): The neural network model.
        valid_loader (torch.utils.data.DataLoader): DataLoader for test data.
        client_id (int): ID of the client.

    Returns:
        dict: A dictionary containing test metrics and loss. 
        
    By @Ouadoud.
    """    
    loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    valid_loader.dataset.is_train = False
    model.eval()
    with no_grad():
        for items, targets in valid_loader:
            items = items.to(float32).to(DEVICE)
            outputs = model(items)
            loss += loss_function(outputs, targets).item()
            
            tp_, tn_, fp_, fn_ = confusion_matrix(outputs, targets)
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_

        valid_loss = loss / len(valid_loader.dataset)
        metrics_dict = get_metrics(tp, tn, fp, fn, valid_loss)
    
    loss = "{:.4e}".format(metrics_dict['BCELoss'])
    accuracy = "{:.4f}%".format(metrics_dict['Accuracy'] * 100)

    getLogger("ray." + "client_" + client_id).info(f"Validation >> Loss: {loss} | Accuracy: {accuracy}")

    return metrics_dict




"""
Inference module for testing neural network models.

This module provides functions for testing a neural network model and calculating performance metrics.

By @Ouadoud
"""




# Define the binary cross-entropy loss function

def test_model(test_loader) -> dict:
    """
    Test the neural network model and calculate performance metrics.
    
    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        
    Returns:
        dict: A dictionary containing test loss, accuracy, precision, recall, F1 score, false positive rate, and counts of TP, TN, FP, and FN.

    By @Ouadoud
    """
    # Initialize metrics
    loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Set the model to evaluation mode
    GLOBALMODEL.eval()
    
    with no_grad():
        for items, targets in test_loader:
            # Move items to the correct device and data type
            items = items.to(dtype=float32, device=DEVICE)
            
            # Get model outputs
            outputs = GLOBALMODEL(items)
            
            # Calculate the loss
            loss += loss_function(outputs, targets.float()).item()

            tp_, tn_, fp_, fn_ = confusion_matrix(outputs, targets)
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_

        test_loss = loss / len(test_loader.dataset)
        metrics_dict = get_metrics(tp, tn, fp, fn, test_loss)

    # Return the log entry with all the metrics
    return metrics_dict
