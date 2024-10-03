from cyberid_pkg.globals import LOAD_TRAIN_DATA_KWARGS, DATA_PATH, LOGS_PATH, NUM_CLIENTS, WEIGHTS, NUM_CORES
import pandas, numpy, torch
from cyberid_pkg import Server_Algorithm, Agent_Algorithm
from matplotlib import pyplot, cm
from matplotlib.ticker import FuncFormatter
from os import path


epochs=
num_rounds=

num_rounds=




def main():

    data_path = path.join(DATA_PATH, "test.feather")
    test_data = pandas.read_feather(data_path)
    test_data = test_data[:50000] ###
    X = test_data.drop(['intrusion'], axis=1)
    y = test_data[['intrusion']]

    
    history = start_simulation(client_fn=generate_client_fn(
                                   distribution=None, 
                                   epochs= epochs
                               ),
                               num_clients=NUM_CLIENTS,
                               server= ,
                               client_resources={
                                   "num_cpus": ,
                                   "num_gpus": ,
                               },
                               config=ServerConfig(num_rounds=num_rounds),
                               strategy=Aggregation(num_clients=NUM_CLIENTS, 
                                                    weights=WEIGHTS, 
                                                    X=X, y=y,
                                                    num_rounds=num_rounds),
                              )



if __name__ == "__main__" :
    main()

return 0