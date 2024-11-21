import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class SAESTARDataset(Dataset):
    """
    A PyTorch Dataset class for training with preprocessed trajectory and distance data.
    """

    def __init__(self, data : dict):
        """
        Initialize the dataset.

        Args:
            data (dict): A dictionary containing:
                - 'src': Source trajectory tensor (shape: [num_samples, src_length, num_features]).
                - 'tgt': Target trajectory tensor (shape: [num_samples, tgt_length, num_features]).
                - 'distance': Distances to agents and objects (shape: [num_samples, src_length, num_distances]).
                - 'type': Types of agents and objects (shape: [num_samples, src_length, num_types]).
        """
        self.data = data

    def __len__(self):
        """
        Return the number of samples in the dataset.
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
        src, tgt, distance, dist_type = self.data['src'], self.data['tgt'], self.data['distance'], self.data['type']
        data = {
            "src": src[idx, :, :].type(torch.float32),
            "tgt": tgt[idx, :, :].type(torch.float32),
            "distance": distance[idx, :, :].type(torch.float32),
            "type": dist_type[idx, :, :].type(torch.long)
        }
        return data
    