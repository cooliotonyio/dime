import numpy as np
import warnings
from sklearn.preprocessing import binarize

def load_model(engine, model_name):
    """Loads a saved index"""
    with open(f"{engine.model_dir}/{model_name}/model.txt", "r") as f:
        model_params = json.loads(f.read())
    #TODO: load embedding nets
    model_params["embedding_nets"] = "WEWEWEWEWEWEWEWE"
    return Model(engine, model_params)

class Model():
    
    def __init__(self, engine, model_params):
        """
        Initializes Model obkect
        
        Parameters:
        name (string): Name of network, used as dictionary key in SearchEngine object
        modalities (list of strings): Ordered list of modalities
        embedding_nets (list of callables): Ordered list of either functions or callable networks
        input_dimensions (list of ints): Ordered list of input dimensions for each embedding_net
        output dimension (int): Output dimension of the model
        desc (string): Description of model
        
        Returns:
        Model: Model object
        """
        self.engine = engine
        self.params = model_params

        self.name = model_params["name"]
        self.output_dim = model_params["output_dim"]
        self.modalities = {modality: i for i, modality in enumerate(model_params["modalities"])}
        self.embedding_nets = model_params["embedding_nets"]
        self.input_dim = model_params["input_dim"]
        self.cuda = model_params["cuda"]
        self.desc = model_params["desc"]

        self.preprocessors = [None for _ in range(len(self.modalities))]

        assert len(self.modalities) == len(self.embedding_nets) == len(self.input_dim)
        
        if self.cuda:
            for embedding_net in self.embedding_nets:
                try:
                    embedding_net.cuda()
                except:
                    continue

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
                
        
    def add_preprocessing(self, modality, preprocessor):
        """
        Adds a preprocessing method to a specific embedding_net
        #TODO: change to use a preprocessing model
        
        Paramaters:
        modality (str): Modality of corresponding embedding_net
        preprocessor_name (str or callable): Either name of a preprocessing model or a callable
        """
        i = self.modalities[modality]
        self.preprocessors[i] = preprocessor

    def get_embedding(self, batch, modality, preprocessing = False):
        """Get embedding of a batch"""
        i = self.modalities[modality]
        if preprocessing:
            preprocessor = self.preprocessors[i]
            if type(preprocessor) == str:
                batch = self.engine.models[preprocessor].batch_embedding(batch, modality, preprocessing = False)
            else:
                batch = preprocessor(*batch)
        return self.embedding_nets[i](*batch)

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
        #TODO: Save embedding nets
        with open(f"{self.engine.model_dir}/{self.name}/model.txt", "w+") as f:
            f.write(info)
