from .dataformat import Dataset, load_train_data
from .neuralnet import NeuralNet
from .train import _main, test
from .client import Client
from .globals import *

import warnings
import logging



train_loader, test_loader = load_data(data = (X_train, Y_train),
                                      size = 1.,
                                      valid_rate = .5,
                                      batch_size = 1024,
                                      distribution = None,
                                      sparse_y = True
                                     )
client_id = 1

distribution= {(0, 0, 0, 0, 1): 0,
                (0, 0, 0, 1, 0): 1,
                (0, 0, 1, 0, 0): 0,
                (0, 1, 0, 0, 0): 0,
                (1, 0, 0, 0, 0): 1}
class Agent_Algorithm :
    def __init__ (self, client_id, model) :
        warnings.simplefilter(action='ignore', category=FutureWarning)
        #warnings.simplefilter(action='ignore', category=DeprecationWarning)
        logging.disable(logging.INFO)

        self.client_id = client_id
        self.client = Client()
    
    def __call__(self, server_address = '127.0.0.1', port = 1234) :
        
        flower.client.start_client(server_address=f"{server_address}:{port}", client=self.client.to_client())
        print("Starting communication..." + "\n")










        


