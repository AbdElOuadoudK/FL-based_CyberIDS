"""
inference.py

This module provides functions for training and testing a neural network model using PyTorch. 

By @Ouadoud.
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy
from pandas import DataFrame
from matplotlib import pyplot, cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from os import path, listdir
from .globals import DEVICE, LOGS_PATH, NUM_CORES

torch.set_num_threads(NUM_CORES)

loss_function = torch.nn.BCELoss()

def _train(model: torch.nn.Module,
           train_loader: torch.utils.data.DataLoader,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=5e-2, patience=2)
    
    train_loader.dataset.is_train = True    
    for epoch in range(epochs):
        train_loss = []
        tp, tn, fp, fn = 0, 0, 0, 0
        model.train()
        for items, targets in train_loader:
            items = items.to(torch.float32).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(items)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy().item())
            
            # Calculate TP, TN, FP, FN
            predictions = outputs >= 0.5
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()
                
        train_loss = sum(train_loss) / len(train_loader.dataset)
        scheduler.step(train_loss)
        
        yield epoch, train_loss, tp, tn, fp, fn

def train(round: int,
          rounds: int,
          client_id: int,
          model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
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
    print(f'\nRound: {round}')
    logs_path = path.join(LOGS_PATH, f"client_{client_id}_logs.txt")
    with open(logs_path, 'a') as file:
        file.write(f'\nRound: {round}')
    
    logs = []
    for epoch, train_loss, tp, tn, fp, fn in _train(model=model, train_loader=train_loader, epochs=epochs):
        
        train_accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        log_entry = {'Epoch': epoch+1, 
                     'Accuracy': train_accuracy,
                     'Precision': (tp) / (tp + fp) if (tp + fp) > 0 else 0,
                     'Recall': (tp) / (tp + fn) if (tp + fn) > 0 else 0,
                     'F1': (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
                     'FPR': (fp) / (tn + fp) if (tn + fp) > 0 else 0,
                     'TP': tp,
                     'TN': tn,
                     'FP': fp,
                     'FN': fn,
                     'Loss': train_loss,
                    }
        logs.append(log_entry)
        
        loss = "{:.3e}".format(train_loss)
        accuracy = "{:.4f} %".format(train_accuracy * 100)
        print(f"Training >> Epoch: {epoch+1} | Loss: {loss} | Accuracy: {accuracy}")
        with open(logs_path, 'a') as file:
            file.write(f"\nTraining >> Epoch: {epoch+1} | Loss: {train_loss} | Accuracy: {train_accuracy}")
    
    logs_path = path.join(LOGS_PATH, f"train_{client_id}_logs.csv")
    if f"train_{client_id}_logs.csv" in listdir(LOGS_PATH):
        DataFrame(logs).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame(logs).to_csv(logs_path, index=False, mode='w') 

    del log_entry['Epoch']
    return model, log_entry

def test(model: torch.nn.Module, 
         valid_loader: torch.utils.data.DataLoader, 
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
    with torch.no_grad():
        for items, targets in valid_loader:
            items = items.to(torch.float32).to(DEVICE)
            outputs = model(items)
            loss += loss_function(outputs, targets).item()
            
            # Calculate TP, TN, FP, FN
            predictions = outputs >= 0.5
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()
        
        valid_loss = loss / len(valid_loader.dataset)
        valid_accuracy = (tp + tn) / (tp + tn + fp + fn)

        log_entry = {'Accuracy': valid_accuracy,
                     'Precision': (tp) / (tp + fp) if (tp + fp) > 0 else 0,
                     'Recall': (tp) / (tp + fn) if (tp + fn) > 0 else 0,
                     'F1': (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
                     'FPR': (fp) / (tn + fp) if (tn + fp) > 0 else 0,
                     'TP': tp,
                     'TN': tn,
                     'FP': fp,
                     'FN': fn,
                     'Loss': valid_loss,
                    }
    
    loss = "{:.3e}".format(valid_loss)
    accuracy = "{:.4f} %".format(valid_accuracy * 100)
    
    print(f"Validation >> Loss: {loss} | Accuracy: {accuracy}")
    logs_path = path.join(LOGS_PATH, f"client_{client_id}_logs.txt")
    with open(logs_path, 'a') as file:
        file.write(f"\nValidation >> Loss: {valid_loss} | Accuracy: {valid_accuracy}")
    
    logs_path = path.join(LOGS_PATH, f"valid_{client_id}_logs.csv")
    if f"valid_{client_id}_logs.csv" in listdir(LOGS_PATH):
        DataFrame([log_entry]).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame([log_entry]).to_csv(logs_path, index=False, mode='w')

    return log_entry
