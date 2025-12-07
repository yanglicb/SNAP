import torch
from torch.utils.data import Dataset, DataLoader

class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.cumsum_lens = [sum(self.lens[:i+1]) for i in range(len(self.lens))]
    
    def __len__(self):
        return sum(self.lens)
    
    def __getitem__(self, index):
        dataset_index = next(i for i, v in enumerate(self.cumsum_lens) if v > index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumsum_lens[dataset_index-1]

        
        data = self.datasets[dataset_index][sample_index]
        return data


class RoundRobinCycleDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.max_len = max(len(d) for d in datasets)
        
        # We'll cycle each dataset so that you can cover the largest dataset fully
        self.total_length = self.max_len * self.num_datasets

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        dataset_idx = index % self.num_datasets
        sample_idx = index // self.num_datasets
        
        # Wrap around if the dataset is shorter
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        
        return self.datasets[dataset_idx][sample_idx]