from sklearn.preprocessing import binarize

class Dataset():
    """
    Wrapper class around a dataset
    """
    def __init__(self, name, data, targets, modality, dimension):
        """
        Initializes dataset object

        Parameters:
        name (string): Name of dataset
        data (iterable): Data in tensor form i.e. transformed PIL.Images 
        targets (iterable): Data in canonical form i.e. filenames for image datasets
        modality (string): modality of dataset
        dimension (int): dimension of each element of dataset
        """
        self.data = data
        self.name = name
        self.targets = targets
        self.modality = modality
        self.dimension = dimension

    def create_loader(self, model, load_embeddings, save_directory, binarized, threshold, cuda):
        """
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
        """
        
        #TODO: enable loading saved_embeddings midway
        
        if load_embeddings:
            directory = "{}/{}/{}/{}".format(
                save_directory, self.name, model.name, "binarized" if binarized else "unbinarized")
            loader = self.load_embeddings(directory, model, binarized)
        else:
            loader = self.process_data(model, binarized, threshold, 0, cuda)
        return loader
    
    def save_batch(self, batch, filename, binarized, save_directory):
        """
        Saves batch into a filename into .npy file
        Does bitpacking if batches are binarized to drastically reduce size of files
        
        Parameters:
        batch (arraylike): Batch to save
        filename (string): Path to save batch to
        binarized (bool): True if batch is binarized
        save_directory (string): Directory to save .npy files
        
        Returns:
        None
        """
        path = "{}/{}.npy".format(save_directory, filename)
        if binarized:
            np.save(path, np.packbits(batch.astype(bool)))
        else:
            np.save(path, batch.astype('float32'))
                
    def load_batch(self, filename, model, binarized):
        """
        Load batch from a filename, does bit unpacking if embeddings are binarized
        
        Called by SearchEngine.load_embeddings()
        
        Parameters:
        filename (string): Path to batch, which should be a .npy file
        model (EmbeddingModel): Model that created the batches, need to correctly format binarized arrays
        binarized (bool): True if arrays are binarized
        
        Returns:
        arraylike: loaded batch
        """
        if binarized:
            batch = np.unpackbits(np.load(filename)).astype('float32')
            dims, rows = model.output_dimension, len(batch) // model.output_dimension
            batch = batch.reshape(rows, dims)
        else:
            batch = np.load(filename).astype('float32')
        return batch
    
    def load_embeddings(self, directory, model, binarized):
        """
        Loads previously saved embeddings from save_directory
        
        Parameters:
        directory (string): Directory of embeddings
        model (EmbeddingModel): Model object that outputted the saved embeddings
        binarized (bool): True if the saved embedding is binarized. False otherwise.
        
        Yields:
        int: Batch index
        arraylike: Embeddings received from passing data through net
        """
        filenames = sorted(["{}/{}".format(directory, filename) for filename in os.listdir(directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx], model, binarized)
            yield batch_idx, embeddings
    
    def process_data(self, model, binarized, threshold, offset, cuda):
        """
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
        """
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