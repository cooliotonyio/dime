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
        """
        Index class
        
        Parameters:
        engine (SearchEngine): SearchEngine instance that model is part of
        index_params (dict): {
            "name":         (str) name of the index
            "model_name":   (str) name of the model of index
            "dataset_name": (str) name of the dataset of index
            "modality":     (str) modality of dataset being indexed
            "binarized":    (bool) boolean of whether the index embedding are binarized or not
            "threshold":    (float) threshold for binarization
            "desc":         (str) A description 
        }
        """
        self.engine = engine
        self.params = index_params
        
        self.name = index_params["name"]
        self.model_name = index_params["model_name"]
        self.dataset_name = index_params["dataset_name"]
        self.modality = index_params["modality"]
        self.binarized = index_params["binarized"]
        self.desc = index_params["desc"] if "desc" in index_params else index_params["name"]
        
        if self.binarized:
            self.threshold = index_params["threshold"]

        self.dim = tuple(self.engine.models[self.model_name].output_dim)
        assert len(self.dim) == 1, "FAISS search only supports 1 dimensional vectors"
        self.index = faiss.IndexFlatL2(self.dim[0])
    
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
        info = json.dumps(info)
        with open(f"{self.engine.index_dir}/{self.name}.index", "w+") as f:
            f.write(info)

    def __len__(self):
        """Returns length of index"""
        return len(self.index)
        