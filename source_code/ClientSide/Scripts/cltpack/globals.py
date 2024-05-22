from torch import cuda, device
from .neuralnet import NeuralNet
from os import path, getcwd

LOAD_TRAIN_DATA_KWARGS = {"size" : 1.,
                          "valid_rate" : .5,
                          "batch_size" : 1024,
                          "sparse_y" : True}

DEVICE = device("cuda" if cuda.is_available() else "cpu")

#if MODEL exist : imortih 
#else MODEL = NeuralNet().to(DEVICE)
MODEL = NeuralNet().to(DEVICE)


main_path = path.dirname(getcwd())
DATA_PATH =  path.join(main_path, "Data")
LOGS_PATH =  path.join(main_path, "Logs")
