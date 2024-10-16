
"""
Utility functions for federated learning logging and evaluation.

This module provides functions to log training and validation metrics and
evaluate the global model on test data.

By @Ouadoud
"""

from os import listdir
from os.path import join
from typing import List, Dict, Optional, Tuple
from pandas import DataFrame
from flwr.common import NDArrays, Scalar
from ..globals import LOGS_PATH, BATCH_SIZE, GLOBALMODEL
from ..learning_kit.inference import test_model
from ..dataformat import load_test_data

def log_fit_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Log and save training metrics from multiple clients.

    Args:
        metrics (List[Tuple[int, Dict[str, float]]]): List of metrics with client counts.

    Returns:
        Dict[str, float]: Averaged training metrics.

    By @Ouadoud
    """
    avg_len = sum(n for n, _ in metrics)
    train_accuracy = sum(n * m['Accuracy'] for n, m in metrics) / avg_len
    train_precision = sum(n * m['Precision'] for n, m in metrics) / avg_len
    train_recall = sum(n * m['Recall'] for n, m in metrics) / avg_len
    train_f1 = sum(n * m['F-measure'] for n, m in metrics) / avg_len
    train_fpr = sum(n * m['FPR'] for n, m in metrics) / avg_len
    train_loss = sum(n * m['BCELoss'] for n, m in metrics) / avg_len

    # Save training logs as a CSV file
    metrics_list = [train_accuracy, train_precision, train_recall, train_f1, train_fpr, train_loss]
    metrics_cols = ["Accuracy", "Precision", "Recall", "F-measure", "FPR", "BCELoss"]



    logs_path = join(LOGS_PATH, "train_logs.csv")
    if "train_logs.csv" in listdir(LOGS_PATH):
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, index=False, mode='w')
        
    loss = "{:.4e}".format(train_loss)
    accuracy = "{:.4f} %".format(train_accuracy * 100)
    
    # Print distributed training metrics
    print(f"Train performance of local models (Averaged): Loss: {loss} | Accuracy: {accuracy}")
    return {
        "Accuracy": train_accuracy,
        "Precision": train_precision,
        "Recall": train_recall,
        "F-measure": train_f1,
        "FPR": train_fpr,
        "BCELoss": train_loss,
    }

def log_eval_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Log and save validation metrics from multiple clients.

    Args:
        metrics (List[Tuple[int, Dict[str, float]]]): List of metrics with client counts.

    Returns:
        Dict[str, float]: Averaged validation metrics.

    By @Ouadoud
    """
    avg_len = sum(n for n, _ in metrics)
    valid_accuracy = sum(n * m['Accuracy'] for n, m in metrics) / avg_len
    valid_precision = sum(n * m['Precision'] for n, m in metrics) / avg_len
    valid_recall = sum(n * m['Recall'] for n, m in metrics) / avg_len
    valid_f1 = sum(n * m['F-measure'] for n, m in metrics) / avg_len
    valid_fpr = sum(n * m['FPR'] for n, m in metrics) / avg_len
    valid_loss = sum(n * m['BCELoss'] for n, m in metrics) / avg_len

    # Save validation logs as a CSV file
    metrics_list = [valid_accuracy, valid_precision, valid_recall, valid_f1, valid_fpr, valid_loss]
    metrics_cols = ["Accuracy", "Precision", "Recall", "F-measure", "FPR", "BCELoss"]
    
    logs_path = join(LOGS_PATH, "valid_logs.csv")
    if "valid_logs.csv" in listdir(LOGS_PATH):
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, index=False, mode='w')
    
    loss = "{:.4e}".format(valid_loss)
    accuracy = "{:.4f} %".format(valid_accuracy * 100)
    
    # Print distributed validation metrics
    print(f"Validation performance of local models (Averaged): Loss: {loss} | Accuracy: {accuracy}")
    return {
        "Accuracy": valid_accuracy,
        "Precision": valid_precision,
        "Recall": valid_recall,
        "F-measure": valid_f1,
        "FPR": valid_fpr,
        "BCELoss": valid_loss,
    }

def global_evaluate(X: NDArrays, y: NDArrays):
    """
    Evaluate the global model on the provided test data.

    Args:
        X (NDArrays): Test features.
        y (NDArrays): Test labels.

    Returns:
        Callable: Function to evaluate the global model with the given parameters.

    By @Ouadoud
    """
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        nonlocal X, y
        test_loader = load_test_data(data=(X, y), batch_size=BATCH_SIZE)
        GLOBALMODEL.set_parameters(parameters)
        metrics_dict = test_model(test_loader)
        
        msg = f"Round: {server_round}" if server_round > 0 else "Initial state"
        print("\n" + msg)
        
        loss = "{:.4e}".format(metrics_dict['BCELoss'])
        accuracy = "{:.4f} %".format(metrics_dict['Accuracy'] * 100)
        
        # Print centralized testing metrics
        print(f"Test performance of global model: Loss: {loss} | Accuracy: {accuracy}")
            
        logs_path = join(LOGS_PATH, "test_logs.csv")
        if "test_logs.csv" in listdir(LOGS_PATH):
            DataFrame([metrics_dict]).to_csv(logs_path, header=False, index=False, mode='a')
        else:
            DataFrame([metrics_dict]).to_csv(logs_path, index=False, mode='w') 
                
        return metrics_dict['BCELoss'], {"Accuracy": metrics_dict['Accuracy']}
    
    return evaluate

def fit_config(rounds: int):
    def _fit_config(server_round: int):
        """Return training configuration dict for each round."""
        nonlocal rounds
        
        #if server_round == 1:
        config = {"public_key": 123, "rounds": rounds, "round": server_round}
        
        return config
    
    return _fit_config
