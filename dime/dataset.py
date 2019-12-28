import numpy as np
import os
from sklearn.preprocessing import binarize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_dataset(engine, dataset_name):
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
            "name":     (str) Name of dataset
            "data":     (iterable) Data in tensor form
            "targets":  (iterable): Data in target/canonical form
            "modality": (string): modality of dataset
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
        self.desc = dataset_params["desc"] if "desc" in dataset_params else dataset_params["name"]

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
        data_loader = DataLoader(self.data, batch_size = batch_size)
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= start_index:
                if not type(batch) in (tuple, list):
                    batch = (batch,)
                if self.engine.cuda:
                    batch = tuple(d.cuda() for d in batch)
                yield batch_idx, batch

    def __len__(self):
        """Number of datapoints"""
        return len(self.data)
    
    def save(self, save_data = False):
        """Save dataset information"""
        #TODO write this function
        raise NotImplementedError()

class ImageDataset(Dataset):
    def __init__(self, engine, dataset_params):
        """Dataset class specific to images
        
        Parameters:
        engine (SearchEngine): SearchEngine instance that model is part of
        dataset_params (dict): {
            "name":     (str) Name of dataset
            "data_dir": (str) Path to directory of images
            "transform":(callable) transforms to apply to raw images
            "modality": (str) modality, should always be "image"
            "dim":      (int) dimension of tensors of dataset
            "desc":     (str) A description
        }
        """
        self.engine = engine
        self.params = dataset_params

        assert dataset_params["modality"] == "image", "ImageDataset received unexpected modality"

        self.name = dataset_params["name"]
        self.data_dir = os.path.normpath(dataset_params["data_dir"])
        self.transform = dataset_params["transform"]
        self.modality = dataset_params["modality"]
        self.dim = dataset_params["dim"]
        self.desc = dataset_params["desc"]

        self.data = ImageFolder(
            f"{self.engine.dataset_dir}/{self.data_dir}", 
            transform=self.transform)
        self.filenames = []
        self.labels = []
        for filename, label in self.data.samples:
            self.filenames.append(os.path.normpath(filename))
            self.labels.append(label)
        
    def idx_to_target(self, indicies):
        """
        Takes either an int or a list of ints and returns corresponding filenames of images

        Parameters:
        indices (int or list of ints): Indices of interest

        Returns:
        list: list of filenames corresponding to provided indicies
        """
        #TODO: rename this function?
        if type(indicies) == int:
            return self.filenames[indicies]
        return [self.filenames[i] for i in indicies]

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
        data_loader = DataLoader(self.data, batch_size = batch_size)
        for batch_idx, bunch in enumerate(data_loader):
            if batch_idx >= start_index:
                batch, _ = bunch
                if self.engine.cuda:
                    batch = batch.cuda()
                yield batch_idx, batch

