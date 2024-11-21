import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    """
    A PyTorch Dataset class for Transformer models.
    This dataset provides pre-processed source and target sequences for training.
    """

    def __init__(self, data):
        """
        Initialize the dataset.

        Args:
            data (dict): A dictionary containing:
                - 'src': Source sequence tensor (shape: [num_samples, src_length, num_features]).
                - 'tgt': Target sequence tensor (shape: [num_samples, tgt_length, num_features]).
        """
        self.data = data  # Store the preprocessed source and target sequences.

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
                - 'src': Source sequence tensor (shape: [src_length, num_features]).
                - 'tgt': Target sequence tensor (shape: [tgt_length, num_features]).
        """
        src, tgt = self.data['src'], self.data['tgt']
        data = {
            "src": src[idx, :, :].type(torch.float16),  # Source sequence as half-precision tensor.
            "tgt": tgt[idx, :, :].type(torch.float16)   # Target sequence as half-precision tensor.
        }
        return data
    