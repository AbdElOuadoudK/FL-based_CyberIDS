from cyberid_pkg.globals import DATA_PATH, NUM_CLIENTS, WEIGHTS, NUM_CORES, NUM_ROUNDS, NUM_EPOCHS
from pandas import read_feather
from path import join
from cyberid_pkg.client_kit import generate_client_fn
from cyberid_pkg.server_kit import Aggregation
from flwr.server import ServerConfig
from flwr.simulation import start_simulation






def run(logs=True, display=True):

    data_path = join(DATA_PATH, "test.feather")
    test_data = read_feather(data_path)
    test_data = test_data[:5000]
    X = test_data.drop(['intrusion'], axis=1)
    y = test_data[['intrusion']]
    
    aggregation_strategy = Aggregation(num_clients=NUM_CLIENTS, 
                                       weights=WEIGHTS, 
                                       X=X, y=y,
                                       num_rounds=NUM_ROUNDS
                                      )
    
    client_function = generate_client_fn(distribution=None, 
                                         epochs= NUM_EPOCHS
                                        )

    
    server_config = ServerConfig(num_rounds=NUM_ROUNDS)
    
    client_resources = {"num_cpus": NUM_CORES, 
                        "num_gpus": 0,
                       }
           
    history = start_simulation(client_fn=client_function,
                               num_clients=NUM_CLIENTS,
                               #server= server_algorithm,
                               client_resources=client_resources,
                               config=server_config,
                               strategy=aggregation_strategy,
                               ray_init_args=ray_init_args
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