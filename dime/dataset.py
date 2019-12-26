import numpy as np
from sklearn.preprocessing import binarize


class Dataset():
    """
    Wrapper class around a dataset
    """
    def __init__(self, engine, dataset_params):
        """
        Wrapper class around a dataset

        Parameters: {
            name (string): Name of dataset
            data (iterable): Data in tensor form i.e. transformed PIL.Images 
            targets (iterable): Data in canonical form i.e. filenames for image datasets
            modality (string): modality of dataset
            dim (int): dimension of tensors of dataset
            desc (str): description of dataset
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
        """Takes either an int or a list of ints and returns corresponding targets of dataset

        Parameters:
        indices (int or list of ints): Indices of interest
        """
        if type(indicies) == int:
            return self.targets[indicies]
        return [self.targets[i] for i in indicies]
    
    def get_data(self, batch_size = 1, start_index = 0):
        """
        Generator function that returns data
        
        Parameters:
        model (Model): Model used to extract features
        binarized (bool): Whether resulting feature vectors should be binarized
        offset (int): Which data index to start yielding feature vectors from
        cuda (bool): Whether CUDA is being used
        threshold (float): Threshold for binarization, used only for binarization
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
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
        return len(self.data)

class ImageDataset(Dataset):
    def __init__(self, engine, dataset_params):
        super(ImageDataset, self).__init__(engine, dataset_params)
        self.modality = "image"
