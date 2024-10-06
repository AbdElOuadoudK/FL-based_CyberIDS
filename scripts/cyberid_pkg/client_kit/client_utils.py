
from ..globals import LOGS_PATH, DATA_PATH
from ..dataformat import load_train_data
from .client import TrainingAgent
from flwr.common.logger import configure
from os import path
from pandas import read_feather

def generate_client_fn(distribution: dict = None, epochs: int = 1):

    def client_fn(cid:str):
        
        configure(identifier="IDS_Learning_Logs", filename=path.join(LOGS_PATH, f"client_{int(cid)}_logs.txt"))
        
        data_path = path.join(DATA_PATH, "train.feather")
        train_data = read_feather(data_path)
        train_loader, valid_loader = load_train_data(data = (train_data.drop(['intrusion'], axis=1), train_data[['intrusion']]),
                                                     size = 1/100, ### add int(cid) generation logic...
                                                     valid_rate = .45,
                                                     batch_size = 512,
                                                     distribution = distribution,
                                                     sparse_y = False
                                                    )
        
        return TrainingAgent(cid, train_loader, valid_loader, distribution, epochs)

    return client_fn