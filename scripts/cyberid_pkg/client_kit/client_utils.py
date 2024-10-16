import os
from ..globals import LOGS_PATH, DATA_PATH, BATCH_SIZE
from ..dataformat import load_train_data
from .client import TrainingAgent
from logging import Formatter, FileHandler, DEBUG, getLogger
from flwr.common.logger import logger
#from ray import logger as ray_logger
from os.path import join
from pandas import read_feather

ray_logger = getLogger("ray." + "client_1")
ray_logger.setLevel(DEBUG)
file_handler = FileHandler(filename=join(LOGS_PATH, f'client_{1}.log'))
formatter = Formatter(f'%(name)s | %(filename)s | %(asctime)s | %(thread)d (client_{1})| %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)
ray_logger.addHandler(file_handler)

def generate_client_fn(train_data, distribution: dict = None, epochs: int = 1):

    def client_fn(cid:str):
        train_loader, valid_loader = load_train_data(data = (train_data.drop(['intrusion'], axis=1), train_data[['intrusion']]),
                                                     size = 1/10000, ### add int(cid) generation logic...
                                                     valid_rate = .45,
                                                     batch_size = BATCH_SIZE,
                                                     distribution = distribution,
                                                     sparse_y = False
                                                    )
        
        return TrainingAgent(cid, train_loader, valid_loader, distribution, epochs)

    return client_fn