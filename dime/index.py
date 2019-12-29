import faiss
import os
import json

def load_index(engine, index_name):
    """Loads a saved index"""
    with open(f"{engine.index_dir}/{index_name}.index", "r") as f:
        index_params = json.loads(f.read())
    index = Index(engine, index_params)
    model = engine.models[index.model_name]
    embedding_dir = f"{engine.embedding_dir}/{index.model_name}/{index.dataset_name}/{index.post_processing}/"
    for _, embeddings in engine.load_embeddings(embedding_dir, model, index.post_processing):
        index.add(embeddings)
    return index

class Index():
    def __init__(self, engine, index_params):
        """
        Index class
        
        Parameters:
        engine (SearchEngine): SearchEngine instance that model is part of
        index_params (dict): {
            "name":             (str) name of the index
            "model_name":       (str) name of the model of index
            "dataset_name":     (str) name of the dataset of index
            "modality":         (str) modality of dataset being indexed
            "post_processing":  (str) "binarized" or ""
            "threshold":        (float) threshold for binarization
            "desc":             (str) A description 
        }
        """
        self.engine = engine
        self.params = index_params
        
        self.name = index_params["name"]
        self.model_name = index_params["model_name"]
        self.dataset_name = index_params["dataset_name"]
        self.modality = index_params["modality"]
        self.desc = index_params["desc"] if "desc" in index_params else index_params["name"]

        if "post_processing" in index_params:
            self.post_processing = index_params["post_processing"]
            if "binarized" == self.post_processing:
                self.threshold = index_params["threshold"]
        else:
            self.post_processing = ""

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
        return self.index.ntotal
        