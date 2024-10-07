
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
from os import listdir
from os.path import join
from pandas import DataFrame
from .globals import GLOBALMODEL, DEVICE, LOGS_PATH, NUM_CORES



set_num_threads(NUM_CORES)

loss_function = BCELoss()

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
            
            # Calculate TP, TN, FP, FN
            predictions = outputs >= 0.5
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()
                
        train_loss = sum(train_loss) / len(train_loader.dataset)
        scheduler.step(train_loss)
        
        yield epoch, train_loss, tp, tn, fp, fn

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
    print(f'\nRound: {round}')
    logs_path = join(LOGS_PATH, f"client_{client_id}_logs.txt")
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
    
    logs_path = join(LOGS_PATH, f"train_{client_id}_logs.csv")
    if f"train_{client_id}_logs.csv" in listdir(LOGS_PATH):
        DataFrame(logs).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame(logs).to_csv(logs_path, index=False, mode='w') 

    del log_entry['Epoch']
    return model, log_entry

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
    logs_path = join(LOGS_PATH, f"client_{client_id}_logs.txt")
    with open(logs_path, 'a') as file:
        file.write(f"\nValidation >> Loss: {valid_loss} | Accuracy: {valid_accuracy}")
    
    logs_path = join(LOGS_PATH, f"valid_{client_id}_logs.csv")
    if f"valid_{client_id}_logs.csv" in listdir(LOGS_PATH):
        DataFrame([log_entry]).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame([log_entry]).to_csv(logs_path, index=False, mode='w')

    return log_entry




"""
Inference module for testing neural network models.

This module provides functions for testing a neural network model and calculating performance metrics.

By @Ouadoud
"""




# Define the binary cross-entropy loss function
loss_function = BCELoss()

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

            # Calculate TP, TN, FP, FN
            predictions = outputs >= 0.5
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()

        # Calculate performance metrics
        test_loss = loss / len(test_loader.dataset)
        test_accuracy = (tp + tn) / (tp + tn + fp + fn)

        log_entry = {
            'Accuracy': test_accuracy,
            'Precision': (tp) / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': (tp) / (tp + fn) if (tp + fn) > 0 else 0,
            'F1': (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
            'FPR': (fp) / (tn + fp) if (tn + fp) > 0 else 0,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Loss': test_loss,
        }
    
    # Return the log entry with all the metrics
    return log_entry
