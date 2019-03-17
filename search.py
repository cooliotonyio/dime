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
          
    def build_index(self, dataset_name, models = None, binarized = False, threshold = 0, load_embeddings = True, save_embeddings = True, step_size = 1000):

        dataset = self.datasets[dataset_name]

        if save_embeddings and not self.save_directory:
            raise Exception("save_directory not specified")

        if models is None:
            models = self.modalities[modality]['models']

        for model in models:
            assert dataset['modality'] == model.modality, "Model modality does not match dataset modality"
            assert model.input_dimension == dataset['dimension'], "Model input dimension does not match dimension of dataset"

            data_loader = create_loader(dataset, model, binarized, threshold, load_embeddings)
            index = faiss.IndexFlatL2(model.output_dimension)

            key = (model.name, dataset_name, binarized)

            self.indexes[key] = index
            self.modalities[dataset.modality]['indexes'].append(key)

            #TODO: Cuda() index

            self.fit(index, model, data_loader, dataset_name, save_embeddings, step_size)

    def fit(self, index, model, data_loader, dataset_name, save_embeddings, step_size):
        '''
        1) Featurize (and optionally binarize) data into embeddings OR load embeddings without data
        2) Optionally save embeddings
        3) Add embeddings into index
        
        Called by build_index
            
        Parameters:
        data (iterable): Some iterable that has interations of (batch index, batch embeddings)
        save_embeddings (bool): If True, write each batch into a file (overwrites)
        load_embeddings (bool): If True, load embeddings from self.save_directory
        verbose (bool): If True, prints metadata every step_size
        step_size (int): Number of batches to process before printing metadata
        
        Returns:
        None
        '''
        if self.verbose
            start_time = time.time()
            print("Building {} index".format(model.name))

        num_batches = len(data_loader)
        batch_magnitude = len(str(num_batches))

        for batch_idx, embeddings in data_loader:
            if self.verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx, num_batches))
            if save_embeddings:
                filename = "{}/{}/batch_{}".format(dataset_name, model.name, str(batch_idx).zfill(batch_magnitude))
                self.save_batch(embeddings, filename, binarized)
            index.add(embeddings)
        
        if self.verbose:
            time_elapsed = time.time() - start_time
            print("Finished building {} index in {} seconds.".format(model.name, round(time_elapsed, 4)))
        
    def save_batch(self, batch, filename, binarized):
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
        if binarized:
            np.save(path, np.packbits(batch.astype(bool)))
        else:
            np.save(path, batch.astype('float32'))
                
    def load_batch(self, filename, model, binarized):
        '''
        Load batch from a filename
        
        Called by SearchEngine.load_embeddings()
        
        Parameters:
        filename (string): Path to batch, which should be a .npy file
        
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
        
        Called by SearchEngine.fit(load_embeddings = True)
        
        Parameters:
        
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        '''
        filenames = sorted(["{}/{}".format(directory, filename) for filename in os.listdir(directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx], model, binarized)
            yield batch_idx, embeddings
    
    def featurize_data(self, data, model, binarized, offset):
        '''
        Generator function that yields embeddings of data in batches
        
        Called by fit(load_embeddings = False)
        
        Parameters:
        data (arraylike): Data to featurize
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        '''
        for batch_idx, (data, target) in enumerate(data):
            if batch_idx >= offset:
                if not type(data) in (tuple, list):
                    data = (data,)
                if self.cuda:
                    data = tuple(d.cuda() for d in data)
                embeddings = model.get_embedding(*data)
                if binarized:
                    embeddings = binarize(embeddings.detach(), threshold=self.threshold)
                yield batch_idx, embeddings.cpu().detach().numpy()
     

    def create_loader(self, dataset, model, binarized, threshold, load_embeddings):
        '''
        Just sanity checks before fitting data. Moved this here for cleaer code
        
        Called by SearchEngine.fit()
        
        Parameters:
        data (iterable): Some iterable that has interations of (batch index, batch embeddings)
        save_embeddings (bool): If True, write each batch into a file (overwrites)
        load_embeddings (bool): If True, load embeddings from self.save_directory
        
        Returns:
        iterable: Loader that yields baches
        '''
        if load_embeddings:
            directory = "{}/{}/{}".format(self.save_directory, dataset['name'], model.name)
            loader = self.load_embeddings(directory, model, binarized)
        else:
            data = dataset['data']
            loader = self.featurize_data(data, model, binarized, 0)
        return loader

    def __repr__(self):
        '''
        '''
        return "SearchEngine<{} modalities, {} models, {} datasets>".format(len(self.modalities), len(self.models), len(self.datasets))