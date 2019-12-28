import torch

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