import torch
import numpy as np
import PIL
import os
import faiss
import time
from torchvision import transforms
from sklearn.preprocessing import binarize

class SearchEngine():
    '''
    Search Engine Class
    '''
    
    def __init__(self, embedding_net, embedding_dimension, cuda = None, is_binarized = True, threshold = 0, save_directory = None, embeddings_name = "embeddings"):
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
        
        self.embedding_net = embedding_net
        self.embedding_dimension = embedding_dimension
        self.cuda = cuda
        self.is_binarized = is_binarized
        self.threshold = threshold
        self.save_directory = save_directory
        self.embeddings_name = embeddings_name
        
        # Initialize index
        self.index = faiss.IndexFlatL2(embedding_dimension)

        # GPU acceleration of net and index
        if self.cuda:
            self.embedding_net.cuda()
#             res = faiss.StandardGpuResources()
#             self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


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
            yield batch_idx, embeddings
        
    def update_index(self, embeddings):
        '''
        Adds embeddings into index non-destructively
        
        Called by SearchEngine.fit()
        
        Parameters:
        embeddings (arraylike): embeddings to add to index
        
        Returns:
        None
        '''
        assert self.index.is_trained
        self.index.add(embeddings)
    
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
        filenames = sorted([filename for filename in os.listdir(self.save_directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx])
            yield batch_idx, embeddings
            
    def resolve_loader(data, save_embeddings, load_embeddings, verbose, step_size):
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
                    
        if (not load embeddings) and (not data):
            return False, "Data not provided", None
        
        if load_embeddings:
            save_embeddings = False
            loader = self.load_embeddings()
        else:
            loader = self.featurize_data(data)
        
        return True, "Sanity checks passed.", loader
        

    def fit(self, data=None, save_embeddings = False, load_embeddings = False, verbose = False, step_size = 100):
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
        
        passed, message, loader = self.resolve_loader(data, save_embedding, load_embeddings, verbose, step_size)
        if not passed:
            raise Exception("Sanity checks not passed")
        if verbose:
            print(message)
            
        num_batches = len(loader)
        batch_magnitude = len(str(num_batches))

        for batch_idx, embeddings in loader:
            if verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx, num_batches))
            if save_embeddings:
                filename = "{}_batch_{}".format(self.embeddings_name, str(batch_idx).zfill(batch_magnitude))
                self.save_batch(embeddings, filename)
            self.update_index(embeddings)
            
        if verbose:
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
            np.save(path, batch)
                
    def load_batch(self, filename):
        '''
        Load batch from a filename
        Unpacks bits if binarized
        
        Called by SearchEngine.load_embeddings()
        
        Parameters:
        filename (string): Path to batch, which should be a .npy file
        
        Returns:
        arraylike: loaded batch
        '''
        path = "{}/{}".format(self.save_directory, filename)
        if self.is_binarized:
            batch = np.unpackbits(np.load(path)).astype('float32')
        else:
            batch = np.load(path).astype('float32')
        dims, rows = self.embedding_dimension, len(batch) // self.embedding_dimension
        return batch.reshape(rows, dims)

    def image_to_tensor(self, image, transform):
        '''
        Turns image into a tensor
        
        Called by User
        
        Parameters:
        image (PIL.Image): Image to turn into 
        transform (torchvision.Transform): Transform to perform over image
        
        Returns:
        tensor: The provided image's tensor
        '''
        tensor = transform(image)[None,:,:,:]
        return tensor
   
    def text_to_tensor(self, text, tensor_dict):
        '''
        Turns text into a tensor
        
        Called by User
        
        Parameters:
        text (string): String to turn into tensor
        tensor_dict (dict): dictionary to look up text
        
        Returns:
        tensor: The provided string's tensor
        '''
        if text not in tensor_dict:
            raise Exception("Word not in dictionary")
        return tensor_dict[text]

    def get_embedding(self, data):
        '''
        Featurizes data into an embedding
        
        Called by User
        
        Parameters:
        data (tensor): Data tensor to featurize
        
        Returns:
        arraylike: Embedding of featurized data
        '''
        if not type(data) in (tuple, list):
            data = (data,)
        if self.cuda:
            data = tuple(d.cuda() for d in data)
        embedding = self.embedding_net.get_embedding(*data).detach()
        if self.is_binarized:
            embedding = binarize(embedding, self.threshold)
        return embedding

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
        start_time = time.time()
        distances, idxs = self.index.search(embeddings, n)
        elapsed_time = time.time() - start_time
            
        if verbose:
            print("Median distance: {}".format(np.median(distances)))
            print("Mean distance: {}".format(np.mean(distances)))
            print("Time elapsed: {}".format(round(elapsed_time, 5)))
            
        return distances, idxs
    