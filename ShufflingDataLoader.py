import torch
from torch.utils.data import DataLoader

class ShufflingDataLoader(DataLoader):
    def __iter__(self):
        batches = super().__iter__()
        for batch in batches:
            data, target = batch
            indices = torch.randperm(data.size(0))
            shuffled_data = data[indices]
            shuffled_target = target[indices]
            yield shuffled_data, shuffled_target
