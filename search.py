import torch
import numpy as np
import PIL
import os
import faiss
import time
from torchvision import transforms
from sklearn.preprocessing import binarize

class EmbeddingModel():
    '''
    Wrapper Class around a neural network
    '''
    def __init__(self, net, name, modality, input_dimension, output_dimension, cuda):
        self.net = net
        self.name = name
        self.modality = modality
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.cuda = cuda
        if self.cuda:
            self.net.cuda()
    
    def get_embedding(self, tensor):
        return self.net.get_embedding(tensor)

class SearchEngine():
    '''
    Search Engine Class
    
    Attributes: 
        embedding_net (neural network): Neural network to pass in tensors and receive embeddings
        embedding_dimension (int): Dimension of embeddings outputted by neural network
        cuda (bool): If True, enables CUDA for featurizing data
        is_binarized (bool): If True, all embeddings are binarized
        threshold (float): Threshold level for binarization
        save_directory (string): Directory to save/load embeddings
        embeddings_name (string): Name for filenames of embeddings
        index (faiss index): Index of featurized data
    
    User Methods:
        fit()
        text_to_tensor()
        image_to_tensor()
        get_embedding()
        search()
        
    '''
    
    def __init__(self, modalities, cuda = None, save_directory = None, verbose = False):
        '''
        Initializes SearchEngine object
        
        Called by User
        
        Parameters:
        embedding_net (neural network): Neural network to pass in tensors and receive embeddings
        embedding_dimension (int): Dimension of embeddings outputted by neural network
        cuda (bool): If True, enables CUDA for featurizing data
        is_binarized (bool): If True, all embeddings are binarized
        threshold (float): Threshold level for binarization
        save_directory (string): Directory to save/load embeddings
        embeddings_name (string): Name for filenames of embeddings
        
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
                'models': []
                'indexes': []
            }

    def build_index(self, dataset_name, models = None, binarized = False, threshold = 0):

        dataset = self.datasets[dataset_name]

        if models is None:
            models = self.modalities[modality]['models']

        for model in models:
            assert model.input_dimension == dataset['dimension'], "Model input dimension does not match dimension of dataset"
            index = faiss.IndexFlatL2(model.output_dimension)
            data = dataset['data']

            key = (model.name, dataset_name, binarized)

            self.indexes[key] = index
            self.modalities[dataset.modality]['indexes'].append(key)

            #TODO: Cuda() index
            #TODO: fit data
    
    def add_model(self, net, name, modality, input_dimension, output_dimension):

        assert (name not in self.models), "Model with given name already in self.models"
        assert (modality in self.modalities), "Modality not supported by SearchEngine"

        model = EmbeddingModel(net, name, modality, input_dimension, output_dimension, self.cuda)
        self.models[name] = model
        self.modalities[modality]['models'].append(name)

    def print_models(self):
        print("{} \t {} \t {}".format("Index", "Model Name", "Modality"))
        for i, key in enumerate(self.models):
            model = self.models[key]
            print("{} \t {} \t {}".format(i, model.name, model.modality))

    def add_dataset(self, dataset_name, data, modality, dimension):
        assert (dataset_name not in self.datasets), "Dataset with dataset_name already in self.datasets"
        assert (modality in self.modalities), "Modality not supported by SearchEngine"
        dataset = {
            'name': dataset_name,
            'data': data,
            'modality': modality
            'dimension': dimension
        }
        self.datasets[dataset_name] = dataset

    def featurize_data(self, data):
        '''
        Generator function that yields embeddings of data in batches
        Binarizes the embeddings if self.is_binarized
        
        Called by fit(load_embeddings = False)
        
        Parameters:
        data (arraylike): Data to featurize
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        '''
        for batch_idx, (data, target) in enumerate(data):
            if not type(data) in (tuple, list):
                data = (data,)
            if self.cuda:
                data = tuple(d.cuda() for d in data)
            embeddings = self.embedding_net.get_embedding(*data)
            if self.is_binarized:
                embeddings = binarize(embeddings.detach(), threshold=self.threshold)
            yield batch_idx, embeddings.cpu().detach().numpy()
    
    def load_embeddings(self):
        '''
        Loads previously saved embeddings from save_directory
        
        Called by SearchEngine.fit(load_embeddings = True)
        
        Parameters:
        None
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        '''
        #TODO: fix
        filenames = sorted([filename for filename in os.listdir(self.save_directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx])
            yield batch_idx, embeddings
            
    def resolve_loader(self, data, save_embeddings, load_embeddings, verbose, step_size):
        '''
        Just sanity checks before fitting data. Moved this here for cleaer code
        
        Called by SearchEngine.fit()
        
        Parameters:
        data (iterable): Some iterable that has interations of (batch index, batch embeddings)
        save_embeddings (bool): If True, write each batch into a file (overwrites)
        load_embeddings (bool): If True, load embeddings from self.save_directory
        verbose (bool): If True, prints metadata every step_size
        step_size (int): Number of batches to process before printing metadata
        
        Returns:
        bool: Whether the parameters satisfy the sanity checks or not
        string: Reason for failing sanity checks
        iterable: Loader that yields baches
        '''
        if save_embeddings:
            if not self.save_directory:
                return False, "save_directory not specified", None
            else:
                if not os.path.isdir(self.save_directory):
                    if input("save_directory does not exist. Create directory? Y/N \n") == "Y":
                        os.makedirs(self.save_directory)
                    else:
                        return False, "Cannot save embeddings", None
                    
        if (not load_embeddings) and (not data):
            return False, "Data not provided", None
        
        if load_embeddings:
            loader = self.load_embeddings()
        else:
            loader = self.featurize_data(data)
        #TODO: enable loading .npy AND data
        return True, "Sanity checks passed.", loader
        

    def fit(self, index, model_name, dataset_name, save_embeddings = False, load_embeddings = False):
        '''
        Main function to
            1) Featurize (and optionally binarize) data into embeddings OR load embeddings without data
            2) Optionally save embeddings
            3) Add embeddings into index
        
        Called by User
            
        Parameters:
        data (iterable): Some iterable that has interations of (batch index, batch embeddings)
        save_embeddings (bool): If True, write each batch into a file (overwrites)
        load_embeddings (bool): If True, load embeddings from self.save_directory
        verbose (bool): If True, prints metadata every step_size
        step_size (int): Number of batches to process before printing metadata
        
        Returns:
        None
        '''
        start_time = time.time()
        
        passed, message, loader = self.resolve_loader(data, save_embeddings, load_embeddings, verbose, step_size)
        if not passed:
            raise Exception("Sanity checks not passed")
        if self.verbose:
            print(message)
            
        num_batches = len(data)
        batch_magnitude = len(str(num_batches))

        for batch_idx, embeddings in loader:
            if self.verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx, num_batches))
            if save_embeddings:
                filename = "{}/{}/batch_{}".format(dataset_name, model_name, str(batch_idx).zfill(batch_magnitude))
                self.save_batch(embeddings, filename)
            index.add(embeddings)
            
        if self.verbose:
            time_elapsed = time.time() - start_time
            print("Finished building index in {} seconds.".format(round(time_elapsed, 4)))
        
    def save_batch(self, batch, filename):
        '''
        Saves batch into a filename into .npy file
        Does bitpacking if batches are binarized to drastically reduce size of files
        
        Called by SearchEngine.fit(save_embeddings = True)
        
        Parameters:
        batch (arraylike): Batch to save
        filename (string): Path to save batch to
        
        Returns:
        None
        '''
        
        path = "{}/{}.npy".format(self.save_directory, filename)
        if self.is_binarized:
            np.save(path, np.packbits(batch.astype(bool)))
        else:
            np.save(path, batch.astype('float32'))
                
    def load_batch(self, filename):
        '''
        Load batch from a filename
        Unpacks bits if self.is_binarized
        
        Called by SearchEngine.load_embeddings()
        
        Parameters:
        filename (string): Path to batch, which should be a .npy file
        
        Returns:
        arraylike: loaded batch
        '''
        path = "{}/{}".format(self.save_directory, filename)
        if self.is_binarized:
            batch = np.unpackbits(np.load(path)).astype('float32')
            dims, rows = self.embedding_dimension, len(batch) // self.embedding_dimension
            batch = batch.reshape(rows, dims)
        else:
            batch = np.load(path).astype('float32')
        return batch

    def search(self, embeddings, n=5, verbose=False):
        '''
        Searches index for nearest n neighbors for each embedding in embeddings
        
        Called by User
        
        Parameters:
        embeddings (np.array): Vector or list of vectors to search
        n (int): Number of neighbors to return for each embedding
        verbose (bool): Prints information about search
        
        Returns
        list: List of lists of distances of neighbors
        list: List of lists of indexes of neighbors
        '''
        # TODO: Fix
        start_time = time.time()
        distances, idxs = self.index.search(embeddings, n)
        elapsed_time = time.time() - start_time
            
        if verbose:
            print("Median distance: {}".format(np.median(distances)))
            print("Mean distance: {}".format(np.mean(distances)))
            print("Time elapsed: {}".format(round(elapsed_time, 5)))
            
        return distances, idxs
    