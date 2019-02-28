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
    
    By default uses binarized embedding of penultimate layer of pretrained ResNet18

    ResNet 152-D

    '''
    def __init__(self, data = None, threshold = 1, embedding_net = None, embedding_dimension = 512, cuda = None, save_directory = None, embeddings_name = "embeddings"):
        
        self.data = data
        self.threshold = threshold
        self.embedding_net = embedding_net
        self.embedding_dimension = embedding_dimension
        self.cuda = cuda
        self.save_directory = save_directory
        self.embeddings_name = embeddings_name
        
        if self.data is None:
            raise Exception('data not specified')
        if self.embedding_net is None:
             raise Exception('embedding_net not specified')
        
        # Initialize index
        self.index = faiss.IndexFlatL2(embedding_dimension)

        # GPU acceleration of net and index
        if self.cuda:
            self.embedding_net.cuda()
#             res = faiss.StandardGpuResources()
#             self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


    def featurize_and_binarize_data(self, data, threshold):
        for batch_idx, (data, target) in enumerate(data):
            if not type(data) in (tuple, list):
                data = (data,)
            if self.cuda:
                data = tuple(d.cuda() for d in data)
            embeddings = self.embedding_net.get_embedding(*data)
            embeddings = binarize(embeddings.detach(), threshold=threshold)
            yield batch_idx, embeddings
    
    def update_index(self, embeddings):
        assert self.index.is_trained
        self.index.add(embeddings)
    
    def load_embeddings(self):
        filenames = sorted([filename for filename in os.listdir(self.save_directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx])
            yield batch_idx, embeddings

    def fit(self, data=None, verbose = False, step_size = 100, threshold = None, save_embeddings = False, load_embeddings = False):
        start_time = time.time()
        
        if save_embeddings:
            if not self.save_directory:
                raise Exception('save_directory not specified')
            else:
                if not os.path.isdir(self.save_directory):
                    if input('save_directory does not exist. Create directory? Y/N \n') == 'Y':
                        os.makedirs(self.save_directory)
                    else:
                        raise Exception('cannot save embeddings')
                
        if threshold == None:
            threshold = self.threshold
        
        if load_embeddings:
            save_embeddings = False
            loader = self.load_embeddings()
        else:
            if not data:
                raise Exception('data not specified')
            loader = self.featurize_and_binarize_data(data, threshold)
            
        num_batches = len(data)
        batch_magnitude = len(str(num_batches))

        for batch_idx, embeddings in loader:
            if verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx, num_batches))
            if save_embeddings:
                filename = "{}_batch_{}.npy".format(self.embeddings_name, str(batch_idx).zfill(batch_magnitude))
                self.save_batch(embeddings, filename)
            self.update_index(embeddings)
        if verbose:
            time_elapsed = time.time() - start_time
            print("Finished building index in {} seconds.".format(round(time_elapsed, 4)))
        
    def save_batch(self, batch, filename):
        path = "{}/{}".format(self.save_directory, filename)
        np.save(path, np.packbits(batch.astype(bool)))
                
    def load_batch(self, filename):
        path = "{}/{}".format(self.save_directory, filename)
        batch = np.unpackbits(np.load(path)).astype('float32')
        dims, rows = self.embedding_dimension, len(batch) // self.embedding_dimension
        return batch.reshape(rows, dims)

    def get_binarized_embedding(self, tensor, threshold = None):
        if threshold is None:
            threshold = self.threshold
        embedding = self.get_embedding(tensor)
        embedding = binarize(embedding.detach(), threshold)
        return embedding

    def image_to_tensor(self, filename, transform):
        image = PIL.Image.open(filename).convert('RGB')
        tensor = transform(image)[None,:,:,:]
        return tensor
   
    def word_to_tensor(self, word, word2vec_dict):
        if word not in word2vec_dict:
            raise Exception("Word not in dictionary")
        return word2vec_dict[word]

    def get_embedding(self, data):
        if not type(data) in (tuple, list):
            data = (data,)
        if self.cuda:
            data = tuple(d.cuda() for d in data)
        embedding = self.embedding_net.get_embedding(*data)
        return embedding
    
    def query(self, tensor, embedding_net = None, n=10, verbose = False, threshold = 0):
        # function need fixing
        if not embedding_net:
            embedding = self.get_binarized_embedding(tensor, threshold = self.threshold)
        else:
            embedding = embedding_net.get_embedding(tensor)
            embedding = binarize(embedding.detach(), threshold)
        print(embedding)
        distances, idxs = self.search(embedding, n, verbose = verbose)
        return distances, idxs

    def search(self, embedding, n=5, verbose=False):
        start_time = time.time()
        distances, idxs = self.index.search(embedding, n)
        elapsed_time = time.time() - start_time
            
        if verbose:
            print("Median distance: {}".format(np.median(distances)))
            print("Mean distance: {}".format(np.mean(distances)))
            print("Time elapsed: {}".format(round(elapsed_time, 5)))
            
        return distances, idxs
    