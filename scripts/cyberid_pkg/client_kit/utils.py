def generate_client_fn(distribution: dict = None, epochs: int = 1):

    def client_fn(cid:str):
        
        configure(identifier="IDS_Learning_Logs", filename=path.join(LOGS_PATH, f"client_{int(cid)}_logs.txt"))
        
        data_path = path.join(DATA_PATH, "train.feather")
        train_data = pandas.read_feather(data_path)
        train_loader, valid_loader = load_train_data(data = (train_data.drop(['intrusion'], axis=1), train_data[['intrusion']]),
                                                     size = 1/10, ### add int(cid) generation logic...
                                                     valid_rate = .45,
                                                     batch_size = 8192,
                                                     distribution = distribution,
                                                     sparse_y = False
                                                    )
        
        return TrainingAgent(client_id, train_loader, valid_loader, distribution, epochs)

    return client_fn