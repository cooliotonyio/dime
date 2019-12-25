import faiss

class Index():
    def __init__(self, index_params):
        """Index class
        
        index_params:{
            "name":             name of the param
            "model_name":       name of the model of index
            "dataset_name":     name of the dataset of index
            "embeddings_dir":   directory of embeddings
            "binarized":        boolean of whether the index embedding are binarized or not
        }


        """
        self.name = index_params["name"]
