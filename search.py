import torch
import numpy as np
import PIL
import os
import faiss
import time
from torchvision import transforms
from sklearn.preprocessing import binarize

class Model():
    '''
    Wrapper Class around a multimodal network
    '''
    
    def __init__(
        self, name, modalities, embedding_nets, input_dimensions, output_dimension, desc, cuda = False):
        '''
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
        '''
        assert len(modalities) == len(embedding_nets) == len(input_dimensions)
        self.modalities = {}           
        for i in range(len(modalities)):
            self.modalities[modalities[i]] = {
                'embedding_net': embedding_nets[i],
                'input_dimension': tuple(input_dimensions[i]),
                'preprocessing': None}
        self.name = name
        self.output_dimension = output_dimension
        self.cuda = cuda
        self.desc = desc
        if cuda:
            for embedding_net in embedding_nets:
                try:
                    embedding_net.cuda()
                except:
                    continue
        
    def add_preprocessing(self, modality, preprocessor):
        '''
        Adds a preprocessing method to a specific embedding_net
        
        Paramaters:
        modality (string): Modality of corresponding embedding_net
        preprocessor (callable): Preprocessor that is called
        '''
        self.modalities[modality]['preprocessing_method'] = preprocessor
    
    def batch_embedding(self, batch, modality, preprocessing = False):
        modality = self.modalities[modality]
        if preprocessing:
            batch = modality['preprocessing'](*batch)
        return modality['embedding_net'](*batch)
    
    def get_embedding(self, tensor, modality, preprocessing = False):
        '''
        Transforms tensor into an embedding based on modality
        
        Parameters:
        tensor (arraylike): Tensor to be transformed
        modality (string): Modality of tensor
        preprocessing (bool): True if preprocessing method of modality should be used
        
        Returns:
        arraylike: Embedding produced by model based on tensor and the modality
        '''
        modality = self.modalities[modality]
        if preprocessing:
            assert modality['preprocessing'], "Preprocessing method does not exist"
            tensor = modality['preprocessing'](tensor)
        assert modality['input_dimension'] == tuple(tensor.shape), "Tensor shape '{}' incompatible with {}".format(tuple(tensor.shape), modality['input_dimension'])
        return modality['embedding_net'](tensor)
    
    def get_info(self):
        '''
        Returns a dictionary summarizing basic information about the model

        Parameters:
        None

        Returns
        dictionary: Dictionary with basic info
        '''
        info = {
            "name": self.name,
            "modalities": self.modalities.keys(),
            "output_dimension": self.output_dimension,
            "desc": self.desc
        }
        return info
    
    def to_cpu(self):
        self.cuda = False
        for modality in self.modalities:
            try:
                modality['embedding_net'].cpu()
            except:
                continue
     
    def to_cuda(self):
        self.cuda = True
        for modality in self.modalities:
            try:
                modality['embedding_net'].cuda
            except:
                continue
                
class Dataset():
    '''
    Wrapper class around a dataset
    '''
    def __init__(self, name, data, targets, modality, dimension):
        '''
        Initializes dataset object

        Parameters:
        name (string): Name of dataset
        data (iterable): Data in tensor form i.e. transformed PIL.Images 
        targets (iterable): Data in canonical form i.e. filenames for image datasets
        modality (string): modality of dataset
        dimension (int): dimension of each element of dataset
        '''
        self.data = data
        self.name = name
        self.targets = targets
        self.modality = modality
        self.dimension = dimension

    def create_loader(self, model, load_embeddings, save_directory, binarized, threshold, cuda):
        '''
        Creates iterable for processing

        Parameters:
        model (Model): Model object used for processing (necessary for loading embeddings)
        load_embeddings (bool): True if loading previous embeddings instead of extracting embeddings
        save_directory (string): Path to directory where embeddings should be saved/loaded
        binarized (bool): True if embeddings should be binarized
        threshold (float): Threshold for binarization
        cuda (bool): True if using CUDA

        Returns:
        iterable: Iterable that yields batches
        '''
        
        #TODO: enable loading saved_embeddings midway
        
        if load_embeddings:
            directory = "{}/{}/{}/{}".format(
                save_directory, self.name, model.name, "binarized" if binarized else "unbinarized")
            loader = self.load_embeddings(directory, model, binarized)
        else:
            loader = self.process_data(model, binarized, threshold, 0, cuda)
        return loader
    
    def save_batch(self, batch, filename, binarized, save_directory):
        '''
        Saves batch into a filename into .npy file
        Does bitpacking if batches are binarized to drastically reduce size of files
        
        Parameters:
        batch (arraylike): Batch to save
        filename (string): Path to save batch to
        binarized (bool): True if batch is binarized
        save_directory (string): Directory to save .npy files
        
        Returns:
        None
        '''
        path = "{}/{}.npy".format(save_directory, filename)
        if binarized:
            np.save(path, np.packbits(batch.astype(bool)))
        else:
            np.save(path, batch.astype('float32'))
                
    def load_batch(self, filename, model, binarized):
        '''
        Load batch from a filename, does bit unpacking if embeddings are binarized
        
        Called by SearchEngine.load_embeddings()
        
        Parameters:
        filename (string): Path to batch, which should be a .npy file
        model (EmbeddingModel): Model that created the batches, need to correctly format binarized arrays
        binarized (bool): True if arrays are binarized
        
        Returns:
        arraylike: loaded batch
        '''
        if binarized:
            batch = np.unpackbits(np.load(filename)).astype('float32')
            dims, rows = model.output_dimension, len(batch) // model.output_dimension
            batch = batch.reshape(rows, dims)
        else:
            batch = np.load(filename).astype('float32')
        return batch
    
    def load_embeddings(self, directory, model, binarized):
        '''
        Loads previously saved embeddings from save_directory
        
        Parameters:
        directory (string): Directory of embeddings
        model (EmbeddingModel): Model object that outputted the saved embeddings
        binarized (bool): True if the saved embedding is binarized. False otherwise.
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        '''
        filenames = sorted(["{}/{}".format(directory, filename) for filename in os.listdir(directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx], model, binarized)
            yield batch_idx, embeddings
    
    def process_data(self, model, binarized, threshold, offset, cuda):
        '''
        Generator function that takes in a model and returns the embeddings of dataset
        
        Parameters:
        model (Model): Model used to extract features
        binarized (bool): Whether resulting feature vectors should be binarized
        offset (int): Which data index to start yielding feature vectors from
        cuda (bool): Whether CUDA is being used
        threshold (float): Threshold for binarization, used only for binarization
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        '''
        for batch_idx, batch in enumerate(self.data):
            if batch_idx >= offset:
                if not type(batch) in (tuple, list):
                    batch = (batch,)
                if cuda:
                    batch = tuple(d.cuda() for d in batch)
                embeddings = model.batch_embedding(batch, self.modality)
                if binarized:
                    embeddings = binarize(embeddings.detach(), threshold=threshold)
                yield batch_idx, embeddings.cpu().detach().numpy()

class SearchEngine():
    '''
    Search Engine Class
    '''

    def __init__(self, modalities, save_directory = None, cuda = False, verbose = False):
        '''
        Initializes SearchEngine object
        
        Parameters:
        modalities (list of strings): Modalities supported by this SearchEngine object
        save_directory (string): Directory to save embeddings
        cuda (bool): True if using CUDA
        verbose (bool): True if messages/information are to be printed out
        
        Returns:
        SearchEngine: SearchEngine object
        '''
        
        self.cuda = cuda
        self.save_directory = save_directory
        self.verbose = verbose
        
        # Initialize index
        self.indexes = {}
        self.models = {}
        self.datasets = {}
        self.modalities = {}

        for modality in modalities:
            self.modalities[modality] = {
                'models': [],
                'indexes': [],
                'datasets': []
            }
            
    def valid_indexes(self, tensor, modality):
        '''
        Returns a list of all indexes that are valid given a tensor and modality

        Parameters:
        tensor (arraylike): Tensor to be processed
        modality (string): Modality of tensor

        Returns:
        (list of tuples): Keys of valid indexes
        '''
        valid_models = []
        valid_indexes_keys = []
        for model in list(self.models.values()):
            if modality in model.modalities:
                submodel = model.modalities[modality]
                if submodel['input_dimension'] == tuple(tensor.shape) or submodel['preprocessing_method']:
                    valid_models.append(model.name)
        for key in self.indexes:
            dataset_name, model_name, binarized = key
            if model_name in valid_models:
                valid_indexes_keys.append(key)
        return valid_indexes_keys
        
    def get_embedding(self, tensor, model_name, modality, preprocessing = False, binarized = False, threshold = 0):
        '''
        Transforms tensor to an embedding using a model

        Parameters:
        tensor (arraylike): Tensor to be processed
        model_name (string): Name of model to process with
        modality (string): Modality of tensor, should supported by model
        preprocessing (bool): True if tensor should be preprocessed before using model
        binarized (bool): True if embedding should be binarized in postprocessing
        threshold (float): Threshold for binarization, only used in binarization

        Returns:
        arraylike: Embedding of the tensor with the model
        '''
        assert model_name in self.models, "Model not found"
        embedding = self.models[model_name].get_embedding(tensor, modality, preprocessing)
        if binarized:
            embedding = binarize(embedding, threshold = threshold)
        return embedding
            
    def search(self, embeddings, index_key, n=5):
        '''
        Searches index for nearest n neighbors for embedding(s)

        Parameters
        embeddings (arraylike or list of arraylikes): Input embeddings to search with
        index_key (tuple): Key of index to search in
        n (int): Number of results

        Returns:
        float(s), int(s): Distances and indicies of each result in dataset
        '''
        assert index_key in self.indexes, "Index key not recognized"
        index = self.indexes[index_key]
        single_vector = False
        if len(embeddings.shape) == 1:
            embeddings = embeddings[None,:]
            single_vector = True
        embeddings = embeddings.cpu().detach().numpy()
        distances, idxs = index.search(embeddings, n)
        if single_vector:
            return distances[0], idxs[0]
        else:
            return distances, idxs
    
    def add_model(self, name, modalities, embedding_nets, 
                  input_dimensions, output_dimension, desc=None, force_add = False):
        '''
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
        '''
        if not force_add:
            assert (name not in self.models), "Model with given name already in self.models"

        if desc is None:
            desc = name
        model = Model(name, modalities, embedding_nets, input_dimensions, output_dimension, desc)
        self.models[name] = model
        for modality in modalities:
            self.modalities[modality]['models'].append(name)
        
        if self.verbose:
            print("Model '{}' added".format(name))

    def add_dataset(self, name, data, targets, modality, dimension, force_add = False):
        '''
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
        '''
        if not force_add:
            assert (name not in self.datasets), "Dataset with dataset_name already in self.datasets"
        assert (modality in self.modalities), "Modality not supported by SearchEngine"
        dataset = Dataset(name, data, targets, modality, dimension)
        self.datasets[name] = dataset
        self.modalities[dataset.modality]['datasets'].append(dataset.name)
        
        if self.verbose:
            print("Dataset '{}' added".format(name))

    def build_index(self, dataset_name, model_name, binarized=False, threshold = 0, load_embeddings = True, 
        save_embeddings = True, batch_size = 128, step_size = 1000):
        '''
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
        '''

        dataset = self.datasets[dataset_name]
        model = self.models[model_name]

        assert not save_embeddings or self.save_directory, "save_directory not specified"
        assert dataset.modality in model.modalities, "Model does not support dataset modality"
        save_directory = "{}/{}/{}/{}/".format(self.save_directory, dataset.name, model.name,
                                                      "binarized" if binarized else "unbinarized")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        if self.verbose:
            start_time = time.time()
            print("Building {}, {} index".format(model.name, dataset.name))
            
        #TODO: Cuda() index
        index = faiss.IndexFlatL2(model.output_dimension)
        loader = dataset.create_loader(model, load_embeddings, self.save_directory, binarized, threshold, self.cuda)

        #TODO: fix num_batches
        num_batches = 10000
        batch_magnitude = len(str(num_batches))

        for batch_idx, embeddings in loader:
            if self.verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx, num_batches))
            if save_embeddings:
                filename = "batch_{}".format(str(batch_idx).zfill(batch_magnitude))
                dataset.save_batch(embeddings, filename, binarized, save_directory)
            index.add(embeddings)


        key = (dataset.name, model.name, binarized)
        
        self.indexes[key] = index
        self.modalities[dataset.modality]['indexes'].append(key)
        self.modalities[dataset.modality]['indexes'] = list(set(self.modalities[dataset.modality]['indexes']))
        
        if self.verbose:
            time_elapsed = time.time() - start_time
            print("Finished building {} index in {} seconds.".format(str(key), round(time_elapsed, 4)))

        return key
        
            
    def target_from_idx(self, indicies, dataset_name):
        '''
        Takes either an int or a list of ints and returns corresponding targets of dataset

        Parameters:
        indices (int or list of ints): Indices of interest
        dataset_name (string or tuple): Name of dataset or index key to retrieve from
        '''
        if type(dataset_name) == tuple and len(dataset_name) == 3:
            dataset_name = dataset_name[0]
        if type(indicies) == int:
            indicies = [indicies]
        dataset = self.datasets[dataset_name]
        return [dataset.targets[i] for i in indicies]
     
    def __repr__(self):
        '''
        Representation of SearchEngine object, quick summary of assets
        '''
        return "SearchEngine<{} modalities, {} models, {} datasets, {} indexes>".format(
            len(self.modalities), len(self.models), len(self.datasets), len(self.indexes))
