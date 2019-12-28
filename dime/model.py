import numpy as np
import warnings
import os
import pickle
import json

def load_model(engine, model_name):
    """Loads a saved model"""
    model_dir = f"{engine.model_dir}/{model_name}"
    with open(f"{model_dir}/model.txt", "r") as f:
        model_params = json.loads(f.read())
    model_params["embedding_nets"] = []
    for f in os.listdir(f"{model_dir}/embedding_nets/"):
        with open(f, "rb"):
            model_params["embedding_nets"].append(pickle.load(f))
    return Model(engine, model_params)

class Model():
    def __init__(self, engine, model_params):
        """
        Initializes Model object
        
        Parameters:
        engine (SearchEngine): SearchEngine instance that model is part of
        model_params (dict): {
            "name":             (str) Name of the model
            "output_dim":       (tuple) Dimension of an output of the model
            "modalities":       (list) A list of modalities the model supports
            "embedding_nets":   (list) A list of callables corresponding to each modality
            "input_dim":        (list) A list of tuples corresponding to each modality
            "desc":             (str) A description
        }
        """
        self.engine = engine
        self.params = model_params

        self.name = model_params["name"]
        self.output_dim = model_params["output_dim"]
        self.modalities = {modality: i for i, modality in enumerate(model_params["modalities"])}
        self.embedding_nets = model_params["embedding_nets"]
        self.input_dim = model_params["input_dim"]
        self.cuda = engine.cuda
        self.desc = model_params["desc"] if "desc" in model_params else model_params["name"]

        self.preprocessors = [None for _ in range(len(self.modalities))]

        assert len(self.modalities) == len(self.embedding_nets) == len(self.input_dim), \
            "Unexpected number of modalities/embedding_nets/input_dim"
        
        for embedding_net in self.embedding_nets:
            try:
                embedding_net.eval()
                if self.cuda:
                    embedding_net.cuda()
            except:
                continue

        if [modality for modality in self.modalities.keys() if modality not in self.engine.modalities]:
            warnings.warn("Model created with unsupported modalities")

    def can_call(self, modality, input_dim):
        """Returns if model can be called on 'modality' with 'input_dim'"""
        modality_index = self.modalities[modality]
        if tuple(self.input_dim[modality_index]) == tuple(input_dim):
            return True
        elif self.preprocessors[modality_index]:
            if type(self.preprocessors[modality_index]) == str:
                preprocessor_name = self.preprocessors[modality_index]
                return self.engine.models[preprocessor_name].can_build(modality, input_dim)
            else:
                warnings.warn("Unable to confirm compatibility with model '{}' with input_dim '{}' for " + \
                    "modality '{}' with unknown preprocessor")
                return True
        return False
                
    def add_preprocessor(self, modality, preprocessor):
        """
        Adds a preprocessing method to a specific embedding_net
        
        Parameters:
        modality (str): Modality of corresponding embedding_net
        preprocessor_name (str or callable): Either name of a preprocessing model or a callable
        """
        i = self.modalities[modality]
        self.preprocessors[i] = preprocessor

    def get_embedding(self, batch, modality, preprocessing = True):
        """Get embedding of a batch"""
        i = self.modalities[modality]
        num_batch = (len(batch),)
        if preprocessing and self.preprocessors[i] is not None:
            preprocessor = self.preprocessors[i]
            if type(preprocessor) == str:
                #TODO: change this so that preprocessed items are loaded instead of being calculated
                batch = self.engine.models[preprocessor].get_embedding(batch, modality, preprocessing = preprocessing)
            else:
                batch = preprocessor(batch)
        return self.embedding_nets[i](batch).view(num_batch + self.output_dim)

    def get_info(self):
        """
        Returns a dictionary summarizing basic information about the model

        Parameters:
        None

        Returns
        dictionary: Dictionary with basic info
        """
        info = {
            "name": self.name,
            "modalities": list(self.modalities.keys()),
            "output_dim": self.output_dim,
            "desc": self.desc
        }
        return info
    
    def save(self):
        """Save the model"""
        info = {
            "name": self.name,
            "modalities": self.params["modalities"],
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "desc": self.desc
        }
        model_dir = f"{self.engine.model_dir}/{self.name}"
        with open(f"{model_dir}/model.txt", "w+") as f:
            f.write(json.dumps(info))
        for modality_index, embedding_net in enumerate(self.embedding_nets):
            with open(f"{model_dir}/embedding_nets/{self.modalities[modality_index]}.pkl", "wb+") as f:
                pickle.dump(embedding_net, f)
