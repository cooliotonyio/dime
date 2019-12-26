import torch
import numpy as np
import PIL
import os
import faiss
import time
from torchvision import transforms
from sklearn.preprocessing import binarize
import warnings

from dime.dataset import Dataset
from dime.index import Index, load_index
from dime.model import Model, load_model

class SearchEngine():
    """
    Search Engine Class
    """

    def __init__(self, engine_params):
        """Initializes SearchEngine object
        
        Parameters:
        modalities (list of strings): Modalities supported by this SearchEngine object
        save_directory (string): Directory to save embeddings
        cuda (bool): True if using CUDA
        verbose (bool): True if messages/information are to be printed out

        dataset_dir:
        embedding_dir:
        model_dir:
        index_dir:
        
        Returns:
        SearchEngine: SearchEngine object
        """
        self.name = engine_params["name"]
        self.cuda = engine_params["cuda"]
        self.verbose = engine_params["verbose"]

        # directories TODO
        self.dataset_dir = engine_params["dataset_dir"]
        self.index_dir = engine_params["index_dir"]
        self.model_dir = engine_params["model_dir"]
        self.embedding_dir = engine_params["embedding_dir"]
        
        # Initialize index
        self.indexes = {}
        self.models = {}
        self.datasets = {}
        self.modalities = {}

        self.modalities = {m: {
            "model_names":[], 
            "index_names":[], 
            "dataset_names":[]
            } for m in engine_params["modalities"] }
            
    def valid_index_names(self, tensor, modality):
        """
        Returns a list of names of all indexes that are valid given a tensor and modality

        Parameters:
        tensor (arraylike): Tensor to be processed
        modality (string): Modality of tensor

        Returns:
        (list of tuples): Keys of valid indexes
        """
        valid_model_names = [m.name for m in list(self.models.values()) if m.can_call(modality, tensor.shape)]
        return [i.name for i in list(self.indexes.values()) if i.model_name in valid_model_names]

    def buildable_indexes(self):
        """Returns (model, dataset) pairs that are compatible"""
        pass
        #TODO: Write this function
        
    def get_embedding(self, tensor, model_name, modality, preprocessing = False, binarized = False, threshold = 0):
        """
        Transforms tensor to an embedding using a model

        Parameters:
        tensor (arraylike): Tensor to be processed
        model_name (string): Name of model to process with
        modality (string): Modality of tensor, should supported by model
        preprocessing (bool): True if tensor should be preprocessed before using model
        binarized (bool): True if embedding should be binarized in post-processing
        threshold (float): Threshold for binarization, only used in binarization

        Returns:
        arraylike: Embedding of the tensor with the model
        """
        assert model_name in self.models, "Model not found"
        batch = tensor[None,:]
        if self.cuda:
            #TODO: fix this if necessary?
            pass
        model = self.models[model_name]
        embedding = model.get_embedding(batch, modality, preprocessing = preprocessing)[0]
        if binarized:
            embedding = binarize(embedding, threshold = threshold)
        return embedding
            
    def search(self, embeddings, index_name, n = 5):
        """
        Searches index for nearest n neighbors for embedding(s)

        Parameters
        embeddings (arraylike or list of arraylikes): Input embeddings to search with
        index_key (tuple): Key of index to search in
        n (int): Number of results

        Returns:
        float(s), int(s): Distances and indicies of each result in dataset
        """
        assert index_name in self.indexes, "index_name not recognized"
        index = self.indexes[index_name]
        if tuple(embeddings.shape) == index.dim:
            embeddings = embeddings[None,:]
            is_single_vector = True
        elif len(embeddings.shape) == (index.dim + 1) and tuple(embeddings.shape)[-len(index.dim)] == index.dim:
            is_single_vector = False
        else:
            raise RuntimeError(f"Provided embeddings of '{embeddings.shape}' " + \
                "not compatible with index '{index.name}' of shape {index.dim} ")

        distances, idxs = index.search(embeddings, n)

        if is_single_vector:
            return distances[0], idxs[0]
        else:
            return distances, idxs
    
    def add_model(self, model_params, force_add = False):
        """
        Add model to SearchEngine

        Parameters:
        name (string): Name of model, used as dictionary key in self.models
        modalities (list of strings): Ordered list of modalities supported by model
        embedding_nets (list of callables): Ordered list of either functions or callable networks
        input_dimensions (list of ints): Ordered list of input dimensions for each embedding_net
        output dimension (int): Output dimension of the model
        desc (string): Description of model

        Returns:
        None
        """
        if not force_add:
            assert (model_params["name"] not in self.models), "Model with given name already in self.models"
        assert [m for m in model_params["modalities"] if m not in self.modalities], f"Modalities not supported by {str(self)}"

        if "desc" not in model_params:
            warnings.warn("'desc' not provided in model_params, using 'name' as default")
            model_params["desc"] = model_params["name"]

        model = Model(self, model_params)
        self.models[model.name] = model
        for modality in model_params["modalities"]:
            self.modalities[modality]['model_names'].append(model.name)
        if self.verbose:
            print("Model '{}' added".format(model.name))

    def add_dataset(self, dataset_params, force_add = False):
        """
        Initializes dataset object

        Called by User

        Parameters:
        name (string): Name of dataset
        data (iterable): Data in tensor form i.e. transformed PIL.Images 
        targets (iterable): Data in canonical form i.e. filenames for image datasets
        modality (string): modality of dataset
        dimension (int): dimension of each element of dataset

        Returns:
        None
        """
        if not force_add:
            assert (dataset_params["name"] not in self.datasets), "Dataset with given name already in self.datasets"
        assert (dataset_params["modality"] in self.modalities), f"Modality not supported by {str(self)}"

        dataset = Dataset(self, dataset_params)
        
        self.datasets[dataset.name] = dataset
        self.modalities[dataset.modality]['datasets'].append(dataset.name)
        
        if self.verbose:
            print("Dataset '{}' added".format(dataset.name))

    def build_index(self, index_params, load_embeddings = True, save_embeddings = True, batch_size = 128, step_size = 1000, force_add = False):
        """
        Adds model embeddings of dataset to index

        Parameters:
        dataset_name (string): Name of dataset
        model_name (model): Name of model
        binarized (bool): True if embeddings should be binarized
        threshold (float): Threshold for binarization, only used in binarization
        load_embeddings (bool): True if loading embeddings, False will process entire dataset with model
        save_embeddings (bool): True if embeddings should be saved to self.save_directory
        batch_size (int): How many elements of dataset are processed at a time
        step_size (int): How many batches before printing messages, only used when self.verbose is True

        Returns:
        tuple: Key of index
        """
        dataset = self.datasets[index_params["dataset_name"]]
        model = self.models[index_params["model_name"]]
        assert dataset.modality in model.modalities, "Model does not support dataset modality"
        assert force_add or index_params["name"] not in model.indexes, "Index with given name already exists"

        post_processing = ""
        if "binarized" in index_params and index_params["binarized"]:
            warnings.warn("Index being built is binarized")
            post_processing = "binarized"

        index = Index(self, index_params)

        embedding_dir = f"{self.embedding_dir}/{model.name}/{dataset.name}/{post_processing}/"
        if not os.path.exists(embedding_dir) and save_embeddings:
            os.makedirs(embedding_dir)
        
        if self.verbose:
            start_time = time.time()
            print("Building {}, {} index".format(model.name, dataset.name))

        num_batches = np.ceil(len(dataset) / batch_size)
        batch_magnitude = len(str(num_batches))

        if load_embeddings:
            for batch_idx, embeddings in self.load_embeddings(embedding_dir, model, post_processing):
                start_index = batch_idx
                index.add(embeddings)
            start_index += 1
        else:
            start_index = 0

        for batch_idx, batch in dataset.get_data(batch_size, start_index = start_index):
            if self.verbose and not (batch_idx % step_size):
                print("Processing batch {} of {}".format(batch_idx, num_batches))

            embeddings = model.get_embedding(batch)
            if post_processing == "binarized":
                embeddings = binarize(embeddings)

            index.add(embeddings)

            if save_embeddings:
                filename = "batch_{}".format(str(batch_idx).zfill(batch_magnitude))
                self.save_batch(embeddings, filename, embedding_dir, post_processing = post_processing)

        if self.verbose:
            time_elapsed = time.time() - start_time
            print("Finished building index {} in {} seconds.".format(index.name, round(time_elapsed, 4)))
        
        self.indexes[index.name] = index
        self.modalities[dataset.modality]['index_names'].append(index.name)

        return index.name
        
            
    def target_from_idx(self, indicies, dataset_name):
        """Takes either an int or a list of ints and returns corresponding targets of dataset

        Parameters:
        indices (int or list of ints): Indices of interest
        dataset_name (string or tuple): Name of dataset or index key to retrieve from
        """
        dataset = self.datasets[dataset_name]
        return dataset.idx_to_target(indicies)

    def load_embeddings(self, embedding_dir, model, post_processing):
        """
        Loads previously saved embeddings from save_directory
        
        Parameters:
        directory (string): Directory of embeddings
        model (EmbeddingModel): Model object that outputted the saved embeddings
        binarized (bool): True if the saved embedding is binarized. False otherwise.
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        """
        filenames = sorted([f for f in os.listdir(embedding_dir) if f[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx], embedding_dir, model.output_dim, post_processing=post_processing)
            yield batch_idx, embeddings

    def load_batch(self, filename, embedding_dir, dim, post_processing=""):
        """
        Load batch from a filename, does bit unpacking if embeddings are binarized
        
        Called by SearchEngine.load_embeddings()
        
        Parameters:
        filename (string): Path to batch, which should be a .npy file
        model (EmbeddingModel): Model that created the batches, need to correctly format binarized arrays
        binarized (bool): True if arrays are binarized
        
        Returns:
        arraylike: loaded batch
        """
        if post_processing == "binarized":
            #TODO: confirm this works
            batch = np.array(np.unpackbits(np.load(filename)), dtype="float32")
            rows = len(batch) // dim
            batch = batch.reshape(rows, dim)
        else:
            batch = np.load(filename).astype("float32")

        if tuple(batch.shape[-len(dim)]) != tuple(dim):
            warnings.warn(f"Loaded batch has dimension {batch.shape[-len(dim)]} but was expected to be {dim}")

        return batch

    def save_batch(self, embeddings, filename, embedding_dir, post_processing = ""):
        """
        Saves batch into a filename into .npy file
        Does bitpacking if batches are binarized to drastically reduce size of files
        
        Parameters:
        batch (arraylike): Batch to save
        filename (string): Path to save batch to
        binarized (bool): True if batch is binarized
        save_directory (string): Directory to save .npy files
        
        Returns:
        None
        """
        path = "{}/{}.npy".format(embedding_dir, filename)
        if post_processing == "binarized":
            #TODO: Confirm this works
            np.save(path, np.packbits(embeddings.astype(bool)))
        else:
            np.save(path, embeddings.astype('float32'))
     
    def __repr__(self):
        """Representation of SearchEngine object, quick summary of assets"""
        return f"SearchEngine< \
            {len(self.modalities)} modalities, \
            {len(self.models)} models, \
            {len(self.datasets)} datasets, \
            {len(self.indexes)} indexes>"
    
    def __str__(self):
        return self.__repr__()

