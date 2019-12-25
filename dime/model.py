class Model():
    
    def __init__(
        self, name, modalities, embedding_nets, input_dimensions, output_dimension, desc, cuda = False):
        """
        Initializes Model obkect
        
        Parameters:
        name (string): Name of network, used as dictionary key in SearchEngine object
        modalities (list of strings): Ordered list of modalities
        embedding_nets (list of callables): Ordered list of either functions or callable networks
        input_dimensions (list of ints): Ordered list of input dimensions for each embedding_net
        output dimension (int): Output dimension of the model
        desc (string): Description of model
        
        Returns:
        Model: Model object
        """
        assert len(modalities) == len(embedding_nets) == len(input_dimensions)
        self.modalities = {}           
        for i in range(len(modalities)):
            self.modalities[modalities[i]] = {
                'embedding_net': embedding_nets[i],
                'input_dimension': tuple(input_dimensions[i]),
                'preprocessing': None}
        self.name = name
        self.output_dimension = output_dimension
        self.cuda = cuda
        self.desc = desc
        if cuda:
            for embedding_net in embedding_nets:
                try:
                    embedding_net.cuda()
                except:
                    continue
        
    def add_preprocessing(self, modality, preprocessor):
        """
        Adds a preprocessing method to a specific embedding_net
        #TODO: change to use a preprocessing model
        
        Paramaters:
        modality (string): Modality of corresponding embedding_net
        preprocessor (callable): Preprocessor that is called
        """
        self.modalities[modality]['preprocessing'] = preprocessor
        if self.cuda:
            try:
                self.modalities[modality]['preprocessing'] = preprocessor.cuda()
            except:
                pass
    
    def batch_embedding(self, batch, modality, preprocessing = False):
        """Get embeedding of a batach"""
        modality = self.modalities[modality]
        if preprocessing:
            batch = modality['preprocessing'](*batch)
        return modality['embedding_net'](*batch)
    
    def get_embedding(self, tensor, modality, preprocessing = False):
        """
        Transforms tensor into an embedding based on modality
        
        Parameters:
        tensor (arraylike): Tensor to be transformed
        modality (string): Modality of tensor
        preprocessing (bool): True if preprocessing method of modality should be used
        
        Returns:
        arraylike: Embedding produced by model based on tensor and the modality
        """
        if self.cuda:
            tensor = tensor.cuda()
        modality = self.modalities[modality]
        if preprocessing and modality['preprocessing']:
            tensor = modality['preprocessing'](tensor)
        return modality['embedding_net'](tensor)
    
    def get_info(self):
        """
        Returns a dictionary summarizing basic information about the model

        Parameters:
        None

        Returns
        dictionary: Dictionary with basic info
        """
        info = {
            "name": self.name,
            "modalities": list(self.modalities.keys()),
            "output_dimension": self.output_dimension,
            "desc": self.desc
        }
        return info
    
    def to_cpu(self):
        """Change model setting to use cpu"""
        self.cuda = False
        for modality in self.modalities:
            try:
                modality['embedding_net'] = modality['embedding_net'].cpu()
                if modality['preprocessing']:
                    modality['preprocessing'] = modality['preprocessing'].cpu()
            except:
                continue
     
    def to_cuda(self):
        """Change model to use cuda (GPU)"""
        self.cuda = True
        for modality in self.modalities:
            try:
                modality['embedding_net'] = modality['embedding_net'].cuda()
                if modality['preprocessing']:
                    modality['preprocessing'] = modality['preprocessing'].cuda()
            except:
                continue