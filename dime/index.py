import faiss
import os
import json

def load_index(engine, index_name):
    """Loads a saved index"""
    with open(f"{engine.index_dir}/{index_name}.index", "r") as f:
        index_params = json.loads(f.read())
    index = Index(engine, index_params)
    #TODO: add embeddings to index
    return index

class Index():
    def __init__(self, engine, index_params):
        """Index class
        
        index_params: {
            "name":             name of the index
            "model_name":       name of the model of index
            "dataset_name":     name of the dataset of index
            "embeddings_dir":   directory of embeddings
            "binarized":        boolean of whether the index embedding are binarized or not
        }
        """
        self.params = index_params
        self.engine = engine

        self.name = index_params["name"]
        self.model_name = index_params["model_name"]
        self.dataset_name = index_params["dataset_name"]
        self.embeddings_dir = index_params["embeddings_dir"]
        self.binarized = index_params["binarized"]
        self.desc = index_params["desc"]

        self.dim = tuple(self.engine.models[self.model_name].output_dimension)
        self.index = faiss.IndexFlatL2(self.dim)
    
    def add(self, embeddings):
        """Add embeddings to index"""
        self.index.add(embeddings)

    def search(self, embeddings, n):
        """Returns (distances, indices) of n closest neighbors to each embedding"""
        return self.index.search(embeddings, n)

    def save(self):
        """Saves index and index information to index_dir"""
        info = self.params
        info["dim"] = self.dim
        with open(f"{self.engine.index_dir}/{self.name}.index", "w+") as f:
            f.write(info)

    def __len__(self):
        """Returns length of index"""
        return len(self.index)
        