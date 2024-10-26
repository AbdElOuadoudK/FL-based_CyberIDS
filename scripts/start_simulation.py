from cyberid_pkg.globals import DATA_PATH, LOGS_PATH, NUM_CLIENTS, WEIGHTS, NUM_CORES, NUM_ROUNDS, NUM_EPOCHS, MEMORY_AMOUNT, BATCH_SIZE
from pandas import read_feather
from os.path import join
from cyberid_pkg import load_train_data
from cyberid_pkg.client_kit import generate_client_fn
from cyberid_pkg.server_kit import Aggregation
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from logging import Formatter, FileHandler, DEBUG
from flwr.common.logger import logger as flwr_logger

# Configure logger for the project
flwr_logger.name = 'FLNetIDS_v2.0'
server_formatter = Formatter('%(name)s | %(filename)s | %(asctime)s | %(thread)d | %(levelname)s | %(message)s')
server_file_handler = FileHandler(filename=join(LOGS_PATH, 'main.log'), mode='w')
server_file_handler.setFormatter(server_formatter)
flwr_logger.addHandler(server_file_handler)

def run(logs=True, display=True):

    data_path = join(DATA_PATH, "test.feather")
    test_data = read_feather(data_path)
    test_data = test_data[:50000]
    X = test_data.drop(['intrusion'], axis=1)
    y = test_data[['intrusion']]
    
    aggregation_strategy = Aggregation(num_clients=NUM_CLIENTS, 
                                       weights=WEIGHTS, 
                                       X=X, y=y,
                                       num_rounds=NUM_ROUNDS
                                      )
    
    data_path = join(DATA_PATH, "train.feather")
    train_data = read_feather(data_path)


    
 
    
    train_loaders_ = list()
    valid_loaders_ = list()
    distributions_ = None
    sparse_y_ = None  
    
    train_loaders, valid_loaders = load_train_data(data=(train_data.drop(['intrusion'], axis=1), train_data[['intrusion']]), 
                                                   num_clients=NUM_CLIENTS, 
                                                   size=1/10, 
                                                   valid_rate=.45, 
                                                   batch_size=BATCH_SIZE)
    
    client_function = generate_client_fn(train_loaders, valid_loaders, 
                                         epochs= NUM_EPOCHS
                                        )
    
    server_config = ServerConfig(num_rounds=NUM_ROUNDS)
    
    client_resources = {
        "num_cpus": NUM_CORES,
        "num_gpus": 0,
        #"memory": MEMORY_AMOUNT, # GB
    }
    
    history = start_simulation(client_fn=client_function,
                               num_clients=NUM_CLIENTS,
                               #server= server_algorithm,
                               client_resources=client_resources,
                               config=server_config,
                               strategy=aggregation_strategy,
                               ray_init_args={"runtime_env": {"heartbeat-interval": 5, "heartbeat-timeout": 30}}
                              )

if __name__ == "start_simulation" :
    run()
    
simulation_configs = {client_function, 
                      aggregation_strategy, 
                      server_config, 
                      client_resources, 
                      server_algorithm
                     }

log_configs(simulation_configs)