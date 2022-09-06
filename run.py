from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager

from Models.SAS_Model import SAS


config = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "optimizer": "ADAM",
    "max_epoch": 200,

    "movies_path": "ml-1m/movies.dat",
    "ratings_path": "ml-1m/ratings.dat",
    "users_path": "ml-1m/users.dat",

    "loss": "bpr",  # top_1, cross_entropy
    "embedding": True,  # True when using embedding layer

    "numpy_seed": 10,
    "split_ratio": 0.8,
    "hidden_dim": 50,              # hidden layer dimension of embedding layer
    "sequence_length": 20,          # sequence count of input
    "attention_layer_count" : 2, # count of attention layer   
    "negative_sample_count" : 100   # count of negative sample for each user
}


if __name__ == "__main__":
    
    '''
    Use String to Int function
    '''

    dataloader = DataLoader(config)
    item_count = dataloader.get_movie_len()
        
    model = SAS(i_dim = item_count, 
                d_dim = config["hidden_dim"], 
                n_dim = config["sequence_length"], 
                b_num = config["attention_layer_count"])
    
    trainmanger = TrainManager(model, dataloader, config)

    trainmanger.start()
    