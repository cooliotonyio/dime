import numpy as np
import os
import pickle
import PIL
import warnings
from sklearn.preprocessing import binarize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dime.utils import BatchKeySampler

def load_dataset(engine, dataset_name):
    with open(f"{engine.dataset_dir}/{dataset_name}.dataset.pkl", "rb") as f:
        dataset_params = pickle.load(f)

    if "image" == dataset_params["modality"]:
        return ImageDataset(engine, dataset_params)
    elif "text" == dataset_params["modality"]:
        assert "data" in dataset_params or "data_file" in dataset_params, "Dataset parameters needs to specify data"
        if "data" not in dataset_params:
            with open(f"{engine.dataset_dir}/{dataset_params['data_file']}", "rb") as f:
                dataset_params["data"] = pickle.load(f)
        return TextDataset(engine, dataset_params)
    else:
        raise NotImplementedError()
    
class Dataset():
    def __init__(self, engine, dataset_params):
        """
        Abstract base wrapper class around a dataset

        Parameters: 
        """
        self.params = dataset_params
        self.engine = engine

        self.data = NotImplementedError()

    def idx_to_target(self, indicies):
        raise NotImplementedError
    
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

    def __len__(self):
        """Number of datapoints"""
        return len(self.data)
    
    def save(self, save_data = False):
        """Save dataset information"""
        raise NotImplementedError()

    def target_to_tensor(self, target):
        """Create tensor from target as if it came from self.data"""
        raise NotImplementedError()

class ImageDataset(Dataset):
    def __init__(self, engine, dataset_params):
        """Dataset class specific to images
        
        Parameters:
        engine (SearchEngine): SearchEngine instance that model is part of
        dataset_params (dict): {
            "name":     (str) Name of dataset
            "data_dir": (str) Path to directory of images
            "transform":(callable) transforms to apply to images
            "modality": (str) modality, should always be "image"
            "dim":      (tuple) dimension of tensors of dataset
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
        self.dim = tuple(dataset_params["dim"])
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
        if type(indicies) == int:
            return self.filenames[indicies]
        return [self.filenames[i] for i in indicies]

    def target_to_tensor(self, target):
        """Create tensor from target as if it came from self.data"""
        image = PIL.Image.open(target)
        return self.transform(image)

    def save(self, save_data = False):
        """Save the dataset"""
        info = self.params
        with open(f"{self.engine.dataset_dir}/{self.name}.dataset.pkl", "wb+") as f:
            pickle.dump(info, f)

class TextDataset(Dataset):
    #TODO: memory optimuize this class
    def __init__(self, engine, dataset_params):
        """Dataset class specific to images
        
        Parameters:
        engine (SearchEngine): SearchEngine instance that model is part of
        dataset_params (dict): {
            "name":     (str) Name of dataset
            "data":     (dict) Mapping of strings to tensors, can specify data_path instead
            "modality": (str) modality, should always be "text"
            "dim":      (tuple) dimension of tensors of dataset
            "desc":     (str) A description
        }
        """
        self.engine = engine
        self.params = dataset_params

        assert dataset_params["modality"] == "text", "TextDataset received unexpected modality"

        self.name = dataset_params["name"]
        self.data = dataset_params["data"]
        self.modality = dataset_params["modality"]
        self.dim = dataset_params["dim"]
        self.desc = dataset_params["desc"]

        self.targets = list(self.data.keys())

    def save(self, save_data = False):
        """Save the dataset"""
        info = {
            "name": self.name,
            "modality": self.modality,
            "dim": self.dim,
            "desc": self.desc
        }
        if "data_file" in self.params:
            info["data_file"] = self.params["data_file"]

        if save_data:
            data_file = f"{self.name}.data.pkl"
            with open(f"{self.engine.dataset_dir}/{data_file}", "wb+") as f:
                pickle.dump(self.data, f)
            info["data_file"] = data_file

        if "data_file" not in info:
            warnings.warn(f"TextDataset {self.name} parameters being saved without saved data")

        with open(f"{self.engine.dataset_dir}/{self.name}.dataset.pkl", "wb+") as f:
            pickle.dump(info, f)
    
    def target_to_tensor(self, target):
        """Create tensor from target as if it came from self.data"""
        assert target in self.data, "Target '{target}' does not exist in '{self}'"
        return self.data[target]

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
        """
        sampler = BatchKeySampler(self.data, batch_size)
        data_loader = DataLoader(self.data, batch_sampler = sampler)
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= start_index:
                if self.engine.cuda:
                    batch = batch.cuda()
                yield batch_idx, batch
    