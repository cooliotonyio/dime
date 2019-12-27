import numpy as np
from sklearn.preprocessing import binarize

def load_dataset():
    #TODO: write this function
    raise NotImplementedError()

class Dataset():
    """
    Wrapper class around a dataset
    """
    def __init__(self, engine, dataset_params):
        """
        Wrapper class around a dataset

        Parameters: 
        engine (SearchEngine): SearchEngine instance that model is part of
        dataset_params (dict): {
            "name"      (str) Name of dataset
            "data"      (iterable) Data in tensor form
            "targets"   (iterable): Data in target/canonical form
            "modality"  (string): modality of dataset
            "dim":      (int) dimension of tensors of dataset
            "desc":     (str) A description
        }
        """
        self.params = dataset_params
        self.engine = engine

        self.name = dataset_params["name"]
        self.data = dataset_params["data"] 
        self.targets = dataset_params["targets"]
        self.modality = dataset_params["modality"]
        self.dim = dataset_params["dim"]
        self.desc = dataset_params["desc"]

    def idx_to_target(self, indicies):
        """
        Takes either an int or a list of ints and returns corresponding targets of dataset

        Parameters:
        indices (int or list of ints): Indices of interest

        Returns:
        list: list of targets corresponding to provided indicies
        """
        if type(indicies) == int:
            return self.targets[indicies]
        return [self.targets[i] for i in indicies]
    
    def get_data(self, batch_size = 1, start_index = 0):
        """
        Generator function that returns data
        
        Parameters:
        batch_size: the size that data should be batched in
        start_index: the index of the first batch to be yielded (will skip earlier batches)
        
        Yields:
        int: Batch index
        arraylike: data in tensor form
        """
        #TODO: implement with batch_size
        for batch_idx, batch in enumerate(self.data):
            if batch_idx >= start_index:
                if not type(batch) in (tuple, list):
                    batch = (batch,)
                if self.engine.cuda:
                    batch = tuple(d.cuda() for d in batch)
                yield batch_idx, batch

    def __len__(self):
        """Number of datapoints"""
        return len(self.data)
    
    def save(self):
        """Save dataset information"""
        #TODO write this function
        raise NotImplementedError()

class ImageDataset(Dataset):
    def __init__(self, engine, dataset_params):
        """Dataset class specific to images"""
        super(ImageDataset, self).__init__(engine, dataset_params)
        self.modality = "image"
