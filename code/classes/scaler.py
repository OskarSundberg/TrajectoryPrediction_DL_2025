import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class Scaler:
    """
    A class to scale and unscale data using Min-Max scaling. It supports scaling of source (src),
    target (tgt), and distance (distance) data, with optional spatial scaling.
    """

    def __init__(self, train_data : dict, model_name : str=None, spatial=False, size : int=None):
        """
        Initializes the Scaler class, fitting separate scalers for different data types (source, target, distance).
        
        Parameters:
        train_data (dict): A dictionary containing training data with keys 'src', 'tgt', and 'distance'.
        spatial (bool): If True, a scaler for distance data is also created (default: False).
        """
        self.spatial = False  # Default spatial scaling is turned off
        # Create and fit the scaler for the source data (first 3 columns of 'src')
        self.src_scaler = self.create_scaler(train_data['src'][:, :, :3])
        # Create and fit the scaler for the target data (first 2 columns of 'tgt')
        self.tgt_scaler = self.create_scaler(train_data['tgt'][:, :, :2])
        # If spatial scaling is enabled, create and fit the scaler for distance data
        if spatial:
            if model_name == "STAR":
                self.dist_scaler = self.create_scaler(train_data['distance'][:, :, :size])
            else:
                self.dist_scaler = self.create_scaler(train_data['distance'])
    def create_scaler(self, data):
        """
        Creates and fits a MinMaxScaler to the provided data.
        
        Parameters:
        data (numpy.ndarray): The data to fit the scaler on.
        
        Returns:
        MinMaxScaler: A fitted MinMaxScaler instance.
        """
        # Get the shape of the input data
        shape = data.shape
        # Instantiate a MinMaxScaler
        scaler = MinMaxScaler()
        # Convert data to numpy if it's a tensor
        np_data = np.array(data)
        # Flatten the data to 2D for scaling
        flat_data = np_data.reshape(shape[0] * shape[1], shape[2])
        # Clip the data to avoid extreme values that could distort scaling
        clipped_data = np.clip(np.array(flat_data, dtype=np.float32), -1e6, 1e6)
        # Fit and return the scaler on the clipped data
        return scaler.fit(clipped_data)

    def scale(self, data, scaler_type: str):
        """
        Scales the input data using the specified scaler type (src, tgt, or distance).
        
        Parameters:
        data (torch.Tensor): The data to be scaled (tensor format).
        scaler_type (str): The type of scaler to use ('src', 'tgt', or 'distance').
        
        Returns:
        torch.Tensor: Scaled data in the original shape.
        """
        # Select the appropriate scaler based on the input type
        if scaler_type == "src":
            scaler = self.src_scaler
        elif scaler_type == "tgt":
            scaler = self.tgt_scaler
        elif scaler_type == "dist":
            scaler = self.dist_scaler
        
        # Get the shape of the input data
        shape = data.shape
        # Flatten the data into 2D for scaling
        data_flat = data.view(data.shape[0] * data.shape[1], data.shape[2])
        # Clip the data to avoid extreme values that could distort scaling
        data_clipped = np.clip(np.array(data_flat.cpu(), dtype=np.float32), -1e6, 1e6)
        # Transform the data using the fitted scaler
        data_scaled = torch.tensor(scaler.transform(data_clipped), dtype=torch.float32).cuda()
        # Reshape the data back to its original shape
        reshaped_data = data_scaled.reshape(shape)
        return reshaped_data

    def unscale(self, data, scaler_type: str, original_shape):
        """
        Reverses the scaling transformation applied to the data, returning the data to its original scale.
        
        Parameters:
        data (torch.Tensor): The data to be unscaled (scaled data in tensor format).
        scaler_type (str): The type of scaler to use for unscaling ('src', 'tgt', or 'distance').
        original_shape (tuple): The original shape of the data to reshape after unscaling.
        
        Returns:
        torch.Tensor: Unscaled data in its original shape.
        """
        # Select the appropriate scaler based on the input type
        if scaler_type == "src":
            scaler = self.src_scaler
        elif scaler_type == "tgt":
            scaler = self.tgt_scaler
        else:
            scaler = self.dist_scaler
        # Flatten the data for inverse transformation
        data_flatten = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        # Inverse transform the data to the original scale
        data_unscaled = torch.tensor(scaler.inverse_transform(np.array(data_flatten.cpu()))).cuda()
        # Reshape the unscaled data back to the original shape
        return data_unscaled.reshape(original_shape)
        
    

    
    
    