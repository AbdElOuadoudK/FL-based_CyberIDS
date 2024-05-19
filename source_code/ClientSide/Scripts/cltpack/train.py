import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .globals import MODEL, DEVICE

def count_identical_rows(mat1: np.ndarray, mat2: np.ndarray) -> int:
    """
    Count the number of identical rows between two matrices.
    
    Args:
        mat1 (np.ndarray): First matrix.
        mat2 (np.ndarray): Second matrix.
        
    Returns:
        int: Number of identical rows between the two matrices.

    By @Ouadoud
    """
    # Convert matrices to tuples of rows
    rows_mat1 = [tuple(row) for row in mat1]
    rows_mat2 = [tuple(row) for row in mat2]

    # Count identical rows
    same_rows_count = sum(row1 == row2 for row1, row2 in zip(rows_mat1, rows_mat2))

    return same_rows_count

# Initialize optimizer, scheduler, and loss function
optimizer = torch.optim.SGD(MODEL.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=5e-2, patience=10, verbose=False)
loss_function = torch.nn.CrossEntropyLoss()

def train(train_loader: torch.utils.data.DataLoader, epochs: int = 3):
    """
    Train the neural network model.
    
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        epochs (int, optional): Number of epochs for training. Defaults to 3.
    
    Yields:
        tuple: A tuple containing epoch number, training loss, and accuracy.

    By @Ouadoud
    """
    train_loader.dataset.is_train = True    
    for epoch in range(epochs):
        train_loss = []
        correct = 0
        MODEL.train()
        for items, targets in train_loader:
            items = items.to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            outputs = MODEL(items)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy().item())
            outputs = outputs.detach().numpy()
            max_values = np.max(outputs, axis=1)
            mask = outputs == max_values[:, np.newaxis]
            correct += count_identical_rows(targets.numpy().astype(int), mask.astype(int))
        
        train_loss = sum(np.array(train_loss))
        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)
        
        yield epoch, train_loss, correct / len(train_loader.dataset)

def _main(round: int, client_id: str, train_loader: torch.utils.data.DataLoader, epochs: int = 3):
    """
    Main function to train the model and visualize the training process.
    
    Args:
        round (int): Current training round.
        client_id (str): Identifier for the client.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        epochs (int, optional): Number of epochs for training. Defaults to 3.
    
    Returns:
        tuple: A tuple containing the updated round, training loss, and training accuracy.

    By @Ouadoud
    """
    logs = []
    round += 1
    print(f'\nRound: {round}')

    for epoch, train_loss, train_accuracy in train(train_loader=train_loader, epochs=epochs):
        log_entry = {'epoch': epoch, 'train_loss': train_loss}
        print(f"Training >> Epoch: {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
        logs.append(log_entry)
    
    logs_df = pd.DataFrame(logs)
    logs_df.set_index('epoch', inplace=True)
    
    plt.style.use('fivethirtyeight')
    ax = logs_df['train_loss'].plot(kind='line', figsize=(15, 4), title=f'Agent {client_id}: NeuralNet Model', ylabel='MSE Loss', label=f'Round {round}')
    _ = ax.legend()

    # Saving MODEL
    
    return round, train_loss, train_accuracy

def test(valid_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """
    Test the neural network model.
    
    Args:
        valid_loader (torch.utils.data.DataLoader): DataLoader for test data.
        
    Returns:
        tuple: A tuple containing test loss and test accuracy.

    By @Ouadoud
    """
    correct, total, loss = 0, 0, 0.0
    valid_loader.dataset.is_train = False
    with torch.no_grad():
        for items, targets in valid_loader:
            items = items.to(torch.float).to(DEVICE)
            outputs = MODEL(items)
            loss += loss_function(outputs, targets).item()
            
            max_values = np.max(outputs.numpy(), axis=1)
            mask = outputs.numpy() == max_values[:, np.newaxis]
            
            correct += count_identical_rows(targets.numpy().astype(int), mask.astype(int))
            
    return loss / len(valid_loader.dataset), correct / len(valid_loader.dataset)
