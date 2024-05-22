import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy
from pandas import DataFrame
from matplotlib import pyplot
from .globals import MODEL, DEVICE



def count_identical_rows(mat1, mat2):

    # Convert matrices to tuples of rows
    rows_mat1 = [tuple(row) for row in mat1]
    rows_mat2 = [tuple(row) for row in mat2]
    
    # Count identical rows
    same_rows_count = sum(row1 == row2 for row1, row2 in zip(rows_mat1, rows_mat2))
    
    return same_rows_count



optimizer=torch.optim.SGD(MODEL.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=5e-2, patience=10)
loss_function = torch.nn.CrossEntropyLoss()



def train(train_loader,
          epochs: int = 3):

    train_loader.dataset.is_train = True    
    for epoch in range(epochs):
        train_loss = []
        correct = 0
        MODEL.train()
        for items, targets in train_loader :
            items = items.to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            outputs = MODEL(items)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy().item())
            outputs = outputs.detach().numpy()
            max_values = numpy.max(outputs, axis=1)
            mask = outputs == max_values[:, numpy.newaxis]
            correct += count_identical_rows(targets.numpy().astype(int), mask.astype(int))
        
        train_loss = sum(numpy.array(train_loss))
        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)
        
        yield epoch, train_loss, correct / len(train_loader.dataset)

def _main (round,
           client_id,
           train_loader,
           epochs=3
          ) :
    """
    Main function to train the model and visualize the training process.
    """
    logs = []
    round+= 1
    print(f'\nRound: {round}')
    


    
    
    for epoch, train_loss, train_accuracy in train(train_loader=train_loader, epochs=epochs):
        log_entry = {'epoch': epoch, 'train_loss': train_loss}
        print(f"Training >> Epoch: {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
        logs.append(log_entry)
    
    logs_df = DataFrame(logs)
    logs_df.set_index('epoch', inplace=True)
    
    pyplot.style.use('fivethirtyeight')
    ax = logs_df['train_loss'].plot(kind='line', figsize=(15, 4), title=f'Agent {client_id}: NeuralNet Model', ylabel='MSE Loss', label=f'Round {round}')
    _ = ax.legend()

    #saving MODEL
    
    return round,train_loss ,train_accuracy


def test(valid_loader):
    """
    Test the neural network model.
    
    Args:
        model (torch.nn.Module): The neural network model.
        valid_loader (torch.utils.data.DataLoader): DataLoader for test data.
        
    Returns:
        tuple: A tuple containing test loss and test accuracy.
    """    
    correct, total, loss = 0, 0, .0
    valid_loader.dataset.is_train = False
    with torch.no_grad():
        for items, targets in valid_loader :
            items = items.to(torch.float).to(DEVICE)
            outputs = MODEL(items)
            loss += loss_function(outputs, targets).item()
            
            max_values = numpy.max(outputs.numpy(), axis=1)
            mask = outputs.numpy() == max_values[:, numpy.newaxis]
            
            correct += count_identical_rows(targets.numpy().astype(int), mask.astype(int))
            
    return loss / len(valid_loader.dataset), correct / len(valid_loader.dataset)
