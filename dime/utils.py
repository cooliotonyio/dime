import torch
import warnings
import numpy as np
import os

class BatchKeySampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, drop_last = False):
        """Samples keys of a dictionary sequentially by dict.keys()

        Parameters:
            data_source (Dataset): dataset to sample from
        """
        self.data_source = data_source.keys()
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for k in iter(self.data_source):
            batch.append(k)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.data_source)

def load_batch(filename, embedding_dir, dim, post_processing=""):
    """
    Load batch from a filename, does bit unpacking if embeddings are binarized
    
    Called by SearchEngine.load_embeddings()
    
    Parameters:
    filename (string): Name of batch .npy file
    embedding_dir (str): Path of the directory containing the embeddings
    dim (tuple): The shape of each embedding should be
    post_processing (str): "binarized" if embeddings are binarized
    
    Returns:
    arraylike: loaded batch
    """
    path = os.path.normpath(f"{embedding_dir}/{filename}")
    if post_processing == "binarized":
        #TODO: confirm this works
        batch = np.array(np.unpackbits(np.load(path)), dtype="float32")
        rows = len(batch) // dim
        batch = batch.reshape(rows, dim)
    else:
        batch = np.load(path).astype("float32")

    if tuple(batch.shape[-len(dim):]) != tuple(dim):
        warnings.warn(f"Loaded batch has dimension {batch.shape[-len(dim):]} but was expected to be {dim}")

    return batch

def save_batch(embeddings, filename, embedding_dir, post_processing = ""):
    """
    Saves batch into a filename into .npy file

    Does bitpacking if batches are binarized to drastically reduce size of files
    
    Parameters:
    embeddings (arraylike): The batch of embeddings to be saved
    filename (string): Name of batch .npy file
    embedding_dir (str): Path of the directory containing the embeddings
    post_processing (str): "binarized" if embeddings are binarized
    
    Returns:
    None
    """
    path = os.path.normpath(f"{embedding_dir}/{filename}.npy")
    if post_processing == "binarized":
        np.save(path, np.packbits(embeddings.astype(bool)))
    else:
        np.save(path, embeddings.astype('float32'))

def allowed_file(filename, extensions):
    """Returns true if filename extension is in extensions, otherwise raise RuntimeError"""
    if "." in filename and filename.rsplit(".", 1)[-1].lower() in extensions:
        return True
    raise RuntimeError(f"Filename '{filename}' not allowed with allowed extensions '{extensions}'")

def in_and_true(k, d):
    """Returns true if key is in dictionary AND corresponding value evaluates to True"""
    return k in d and d[k]

def sanitize_dict(d):
    """Returns a copy of a JSON-safe version of a dictionary"""

    primitives = (str, dict, list, int, float, bool)
    keys = list(d.keys())
    safe_d = {}
    for k in keys:
        if type(d[k]) not in primitives:
            safe_d[k] = str(d[k])
        elif type(d[k]) == dict:
            safe_d[k] = sanitize_dict(d)
        elif type(d[k]) == list:
            safe_d[k] = [str(v) for v in d[k]]
        else:
            safe_d[k] = d[k]
    return safe_d

