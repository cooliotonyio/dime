import numpy as np
import os
import time
import json
import random
from sklearn.preprocessing import binarize
import warnings

from dime.dataset import Dataset, ImageDataset, TextDataset, load_dataset
from dime.index import Index, load_index
from dime.model import Model, load_model
from dime.utils import load_batch, save_batch

def load_engine(engine_path):
    start_time = time.time()
    with open(engine_path, "r") as f:
        engine_params = json.loads(f.read())
    engine = SearchEngine(engine_params)
    engine.vprint(f"Finished loading engine in {round(time.time() - start_time, 3)} seconds")
    return engine

class SearchEngine():
    def __init__(self, engine_params):
        """
        Initializes SearchEngine object
        
        Parameters: 
        engine_params (dict): {
            "name":             (str) what to name the instance
            "cuda":             (bool) True if using CUDA
            "verbose":          (bool) True if messages/information are to be printed out
            "dataset_dir":      (str) Directory of Datasets
            "index_dir":        (str) Directory of Indexes
            "model_dir":        (str) Directory of Models
            "embedding_dir":    (str) Directory of embeddings
            "modalities":       (list of str) The modalities support by this instance
        }
        """
        self.params = engine_params

        self.name = engine_params["name"]
        self.cuda = engine_params["cuda"]
        self.verbose = engine_params["verbose"]

        self.dataset_dir = engine_params["dataset_dir"]
        self.index_dir = engine_params["index_dir"]
        self.model_dir = engine_params["model_dir"]
        self.embedding_dir = engine_params["embedding_dir"]
        
        self.indexes = {}
        self.models = {}
        self.datasets = {}
        self.modalities = {}

        self.modalities = {m: {
            "dataset_names":[],
            "index_names":[], 
            "model_names":[], 
            } for m in engine_params["modalities"] }
        
        if "modality_dicts" in engine_params:
            for modality in self.modalities:
                modality_dict = engine_params["modality_dicts"][modality]
                for dataset_name in modality_dict["dataset_names"]:

                    self.vprint(f"Loading dataset '{dataset_name}'... ", end = "")
                    start_time = time.time()
                    dataset = load_dataset(self, dataset_name)
                    self.datasets[dataset.name] = dataset
                    self.modalities[dataset.modality]["dataset_names"].append(dataset.name)
                    self.vprint(f"done in {round(time.time() - start_time, 4)} seconds!")

                for model_name in modality_dict["model_names"]:
                    if model_name not in self.models:
                        self.vprint(f"Loading model '{model_name}'... ", end = "")
                        start_time = time.time()
                        model = load_model(self, model_name)
                        self.models[model.name] = model
                        for model_modality in model.modalities:
                            self.modalities[model_modality]["model_names"].append(model.name)
                        self.vprint(f"done in {round(time.time() - start_time, 4)} seconds!")
                            
                for index_name in modality_dict["index_names"]:
                    self.vprint(f"Loading index '{index_name}'... ", end = "")
                    start_time = time.time()
                    index = load_index(self, index_name)
                    self.indexes[index.name] = index
                    self.modalities[index.modality]["index_names"].append(index.name)
                    self.vprint(f"done in {round(time.time() - start_time, 4)} seconds!")

    def valid_index_names(self, modality, tensor = None):
        """
        Returns a list of names of all indexes that are valid given a tensor and modality

        Parameters:
        tensor (arraylike): Tensor to be processed
        modality (str): Modality of tensor

        Returns:
        list of tuples: Keys of valid indexes
        """
        if tensor is not None:
            valid_model_names = [m.name for m in list(self.models.values()) if m.can_call(modality, tensor.shape)]
            valid = [i.name for i in list(self.indexes.values()) if i.model_name in valid_model_names]
        else:
            valid = [i.name for i in list(self.indexes.values()) if modality in i.input_modalities]
        return valid

    def buildable_indexes(self):
        """Returns (model, dataset) pairs that are compatible"""
        pass
        #TODO: Write this functiont"
    
    def get_embedding(self, model_name, batch, modality, preprocessing = True):
        model = self.models[model_name]
        if self.cuda:
            batch = batch.cuda()
        embeddings = model.get_embedding(batch, modality, preprocessing = preprocessing)
        if self.cuda:
            embeddings = embeddings.detach().cpu()
        return embeddings.numpy()

    def search(self, tensor, tensor_modality, index_name, n = 5, preprocessing = True):
        """
        Searches index for nearest n neighbors for embedding(s)

        Parameters
        tensor (arraylike or list of arraylikes): Input embeddings to search with
        index_name (str): Name of index to search in
        n (int): Number of results to be returned per embedding
        preprocessing (bool): if tensor should be preprocessed before embedding extraction

        Returns:
        float(s), int(s): Distances and indicies of each result in dataset
        """
        assert index_name in self.indexes, "index_name not recognized"
        index = self.indexes[index_name]
        model = self.models[index.model_name]
        assert tensor_modality in model.modalities, f"Model '{model.name}' does not support modality '{tensor_modality}'"

        m = model
        t_shape = tuple(tensor.shape)
        while m:
            m_dim = tuple(m.input_dim[m.modalities[tensor_modality]])
            preprocessor = m.preprocessors[m.modalities[tensor_modality]]
            if t_shape == m_dim:
                batch = tensor[None,:]
                is_single_vector = True
                break
            elif len(t_shape) == (len(m_dim) + 1) and t_shape[-len(m_dim)] == m_dim:
                is_single_vector = False
                break
            elif preprocessing and preprocessor:
                if type(preprocessor) == str:
                    m = self.models[preprocessor]
                    continue
                else:
                    break
            else:
                print(t_shape, m_dim)
                raise RuntimeError(f"Provided tensor of shape '{t_shape}' not compatible with index model '{model.name}'")

        embeddings = self.get_embedding(model.name, batch, tensor_modality, preprocessing = preprocessing)
        if index.post_processing == "binarized":
            embeddings = binarize(embeddings)

        distances, idxs = index.search(embeddings, n)

        if is_single_vector:
            return distances[0], idxs[0]
        else:
            return distances, idxs
    
    def add_model(self, model_params, force_add = False):
        """
        Adds model to SearchEngine

        Parameters:
        model_params (dict): See Model.__init__
        force_add (bool): True if forcefully overwriting any Model with the same name

        Returns:
        None
        """
        if not force_add:
            assert (model_params["name"] not in self.models), "Model with given name already in self.models"
        assert not [m for m in model_params["modalities"] if m not in self.modalities], f"Modalities not supported by {str(self)}"

        model = Model(self, model_params)
        self.models[model.name] = model
        for modality in model.modalities:
            self.modalities[modality]["model_names"].append(model.name)
        self.vprint("Model '{}' added".format(model.name))

    def add_preprocessor(self, model_name, modality, preprocessor):
        """
        Adds a preprocessing method for a modality for a model
        
        Parameters:
        model_name (str): The model that should have a preprocessor
        modality (str): Modality of corresponding embedding_net
        preprocessor_name (str or callable): Either name of a preprocessing model or a callable
        """
        model = self.models[model_name]
        if type(preprocessor) == str:
            if (modality not in self.models[preprocessor].modalities):
                warnings.warn(f"Preprocessor {preprocessor} is not compatible with modality {modality}")
        model.add_preprocessor(modality, preprocessor)
    
    def add_dataset(self, dataset_params, force_add = False):
        """
        Initializes dataset object

        Called by User

        Parameters:
        dataset_params (dict): See Dataset.__init__
        force_add (bool): True if forcefully overwriting any Dataset with the same name

        Returns:
        None
        """
        if not force_add:
            assert (dataset_params["name"] not in self.datasets), "Dataset with given name already in self.datasets"
        assert (dataset_params["modality"] in self.modalities), f"Modality not supported by {str(self)}"

        if "image" == dataset_params["modality"]:
            dataset = ImageDataset(self, dataset_params)
        elif "text" == dataset_params["modality"]:
            dataset = TextDataset(self, dataset_params)
        else:
            dataset = Dataset(self, dataset_params)
        
        self.datasets[dataset.name] = dataset
        self.modalities[dataset.modality]["dataset_names"].append(dataset.name)
        
        self.vprint("Dataset '{}' added".format(dataset.name))

    def build_index(self, index_params, load_embeddings = True, save_embeddings = True, batch_size = 128, message_freq = 1000, force_add = False):
        """
        Adds model embeddings of dataset to index

        Parameters:
        index_params (dict): See Index.__init__
        load_embeddings (bool): True if function should use previously extracted embeddings if they exist
        save_embeddings (bool): True if extracted embeddings should be saved during the function
        batch_size (int): The size of a batch of data being processed
        message_freq (int): How many batches before printing any messages if verbose
        force_add (bool): True if forcefully overwriting any Index with the same name

        Returns:
        tuple: Key of index
        """
        dataset = self.datasets[index_params["dataset_name"]]
        model = self.models[index_params["model_name"]]
        assert dataset.modality in model.modalities, "Model does not support dataset modality"
        assert force_add or index_params["name"] not in self.indexes, "Index with given name already exists"
        index_params["modality"] = dataset.modality

        post_processing = ""
        if "post_processing" in index_params:
            warnings.warn(f"Index being built is being post processed with {index_params['post_processing']}")
            post_processing = index_params["post_processing"]

        index = Index(self, index_params)

        embedding_dir = f"{self.embedding_dir}/{model.name}/{dataset.name}/{post_processing}/"
        if not os.path.exists(embedding_dir) and save_embeddings:
            os.makedirs(embedding_dir)
        
        start_time = time.time()
        self.vprint("Building {}, {} index".format(model.name, dataset.name))

        num_batches = int(np.ceil(len(dataset) / batch_size))
        batch_magnitude = len(str(num_batches))

        start_index = 0
        if load_embeddings:
            for batch_idx, embeddings in self.load_embeddings(embedding_dir, model, post_processing):
                if not (batch_idx % message_freq):
                    self.vprint("Loading batch {} of {}".format(batch_idx, num_batches))
                start_index = batch_idx + 1
                index.add(embeddings)

        if start_index < num_batches:
            for batch_idx, batch in dataset.get_data(batch_size, start_index = start_index):
                if not (batch_idx % message_freq):
                    self.vprint("Processing batch {} of {}".format(batch_idx, num_batches))

                embeddings = model.get_embedding(batch, dataset.modality)
                if post_processing == "binarized":
                    embeddings = binarize(embeddings)

                embeddings = embeddings.detach().cpu().numpy()
                index.add(embeddings)

                if save_embeddings:
                    filename = "batch_{}".format(str(batch_idx).zfill(batch_magnitude))
                    save_batch(embeddings, filename, embedding_dir, post_processing = post_processing)

        time_elapsed = time.time() - start_time
        self.vprint("Finished building index {} in {} seconds.".format(index.name, round(time_elapsed, 4)))
        
        self.indexes[index.name] = index
        self.modalities[index.modality]["index_names"].append(index.name)

        return index.name
            
    def idx_to_target(self, indicies, dataset_name):
        """Takes either an int or a list of ints and returns corresponding targets of dataset

        Parameters:
        indices (int or list of ints): Indices of interest
        dataset_name (str): Name of dataset or index to retrieve from

        Returns:
        list: Targets corresponding to provided indicies in specified dataset
        """
        if dataset_name in self.datasets:
            dataset = self.datasets[dataset_name]
        elif dataset_name in self.indexes:
            dataset = self.datasets[self.indexes[dataset_name].dataset_name]
        else:
            raise RuntimeError(f"'{dataset_name}' not found in self.datasets or self.indexes")

        return dataset.idx_to_target(indicies)

    def target_to_tensor(self, target, dataset_name = None, modality = None):
        """
        Convert a raw target data into a useable tensor as if it came from a given dataset

        Uses the first dataset of a modality if modality is specified instead of dataset_name

        Parameters:
        target (object): some target to to turn into a tensor
        dataset_name (str): name of dataset that tensor should look like it came from
        """
        if modality is not None:
            dataset_name = self.modalities[modality]["dataset_names"][0]
        elif dataset_name is None:
            raise RuntimeError("dataset_name or modality was not specified in target_to_tensor")
        dataset = self.datasets[dataset_name]
        return dataset.target_to_tensor(target)
            
    def load_embeddings(self, embedding_dir, model, post_processing):
        """
        Loads previously saved embeddings from save_directory
        
        Parameters:
        embedding_dir (string): Directory of embeddings
        model (Model): Model object with the output dimensions embedddings should be reshaped to
        post_processing (str): "binarized" if embeddings are binarized
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through model
        """
        filenames = sorted([f for f in os.listdir(embedding_dir) if f[-4:] == ".npy"])
        for batch_idx in range(len(filenames)):
            embeddings = load_batch(filenames[batch_idx], embedding_dir, model.output_dim, post_processing=post_processing)
            yield batch_idx, embeddings
     
    def save(self, shallow = False, save_data = False):
        info = {
            "name": self.name,
            "cuda": self.cuda,
            "verbose": self.verbose,
            "dataset_dir": self.dataset_dir,
            "index_dir": self.index_dir,
            "model_dir": self.model_dir,
            "embedding_dir": self.embedding_dir,
            "modality_dicts": self.modalities,
            "modalities": list(self.modalities.keys())
        }

        if not shallow:
            for _, dataset in self.datasets.items():
                dataset.save(save_data = save_data)
            for _, index in self.indexes.items():
                index.save()
            for _, model in self.models.items():
                model.save()

        with open(f"{self.name}.engine", "w+") as f:
            f.write(json.dumps(info))
    
    def __repr__(self):
        """Representation of SearchEngine object, quick summary of assets"""
        return "SearchEngine<" + \
            f"name={self.name}, " + \
            f"{len(self.modalities)} modalities, " + \
            f"{len(self.models)} models, " + \
            f"{len(self.datasets)} datasets, " + \
            f"{len(self.indexes)} indexes>"
    
    def __str__(self):
        """String representation of SearchEngine object, uses __repr__"""
        return self.__repr__()

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)