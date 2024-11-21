import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

    
class STARDataset(Dataset):
    """
    A PyTorch Dataset class for training with preprocessed trajectory and distance data.
    """
    def __init__(self, data, max_num_agents):
        self.data=data
        self.max_num_agents = max_num_agents

    def __len__(self):
        """
        Initialize the dataset.

        Args:
            data (dict): A dictionary containing:
                - 'src': Source trajectory tensor (shape: [num_samples, src_length, num_features]).
                - 'tgt': Target trajectory tensor (shape: [num_samples, tgt_length, num_features]).
                - 'distance': Distances to agents and objects (shape: [num_samples, src_length, num_distances]).
                - 'type': Types of agents and objects (shape: [num_samples, src_length, num_types]).
        """
        return self.data["src"].shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            data (dict): A dictionary containing:
                - 'src': Source trajectory tensor.
                - 'tgt': Target trajectory tensor.
                - 'distance': Distance tensor.
                - 'type': Type tensor.
        """
        src, tgt, distance, type = self.data['src'], self.data['tgt'], self.data['distance'], self.data['type']
        data = {
          "src": src[idx, :, :].type(torch.float32),
          "tgt": tgt[idx, :, :].type(torch.float32),
          "distance": distance[idx, :, :self.max_num_agents].type(torch.float32),
          "type": type[idx, :, :self.max_num_agents].type(torch.long)
        }
        return data