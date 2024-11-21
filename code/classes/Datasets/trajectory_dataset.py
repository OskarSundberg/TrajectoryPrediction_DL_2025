import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from classes.distances import Distances

class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset class for trajectory prediction with spatial and environmental context.
    This dataset includes distances and types of surrounding agents and environmental objects.
    """

    def __init__(self, data, df_vector=None, env_df=None, distance_objects=None, dataset_len=None, max_num_agents=30, src_length=10, tgt_length=40, info=False):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): The trajectory data containing positional and agent information.
            df_vector (pd.DataFrame): Additional data for environmental objects (e.g., vectorized features).
            env_df (pd.DataFrame): Environmental object data.
            distance_objects (list): Pre-computed distance information for environmental objects.
            dataset_len (int): The total number of samples in the dataset.
            max_num_agents (int): Maximum number of agents considered in the environment (default: 30).
            src_length (int): Number of timesteps in the source sequence (default: 10).
            tgt_length (int): Number of timesteps in the target sequence (default: 40).
        """
        self.data = data  # Main trajectory data.
        self.start_idx = 0  # Starting index for sampling sequences.
        self.df_vector = df_vector  # Vectorized features of environmental objects.
        self.env = np.array(env_df)  # Environmental object data as a NumPy array.
        self.max_num_agents = max_num_agents  # Maximum number of agents to include.
        self.distance_objects = distance_objects  # Precomputed distance data for static objects.
        self.max_src_grp_size = 1  # Maximum group size in source sequence.
        self.max_tgt_group_size = 1  # Maximum group size in target sequence.
        self.tgt_sequence_length = tgt_length  # Length of target sequence.
        self.src_sequence_length = src_length  # Length of source sequence.
        self.number_of_features = 5  # Number of features per timestep (e.g., X, Y, speed, etc.).
        self.dataset_length = dataset_len  # Total dataset length.
        self.calculations = Distances()  # Utility for distance calculations.
        self.info = info
    def __len__(self):
        """
        Return the length of the dataset.
        """
        return self.dataset_length

    def __getitem__(self, idx):
        """
        Generate a single sample including trajectory, distances, and types.

        Args:
            idx (int): The index of the sample (ignored; sequential sampling is used).
        
        Returns:
            sample (dict): A dictionary containing:
                - 'src': Source trajectory tensor (shape: [src_length, number_of_features]).
                - 'tgt': Target trajectory tensor (shape: [tgt_length, number_of_features]).
                - 'distance': Distances to agents and environmental objects (shape: [src_length, max_num_agents + len(env)]).
                - 'dist_type': Types of surrounding agents and environmental objects (shape: [src_length, max_num_agents + len(env)]).
                - 'timestamp': Timestamps associated with the source sequence.
        """

        while True:
            
            src_end_idx = self.start_idx + self.src_sequence_length
            tgt_end_idx = src_end_idx + self.tgt_sequence_length
            
            # Ensure indices are within bounds
            if tgt_end_idx > len(self.data) - 1:
                return {
                    'src': None,
                    'tgt': None,
                    'dist': None,
                    'dist_type': None,
                    'timestamp': None
                }
            
            # Slice the dataframe once
            data_slice = self.data.iloc[self.start_idx:tgt_end_idx]
            
            # Split into source and target groups
            src_time_groups = data_slice.iloc[:self.src_sequence_length]
            tgt_time_groups = data_slice.iloc[self.src_sequence_length:]
            
            if src_time_groups['ID'].nunique() > 1 or tgt_time_groups['ID'].nunique() > 1:
                self.start_idx = self.start_idx + 1
                continue

            # Extract features and timestamps
            src_features = src_time_groups[['X', 'Y', 'Speed', 'ID', 'Type']].values.astype(np.float32)
            tgt_features = tgt_time_groups[['X', 'Y', 'Speed', 'ID', 'Type']].values.astype(np.float32)
            timestamps = src_time_groups['Time'].values

            # Convert to tensors
            src = torch.tensor(src_features, dtype=torch.float32)
            tgt = torch.tensor(tgt_features, dtype=torch.float32)
            
            # for getting the metadat information regarding the dataset length
            if self.info == True:
                # Update the starting index for the next sample.
                self.start_idx = self.start_idx + self.src_sequence_length
                # Return the sample.
                sample = {
                    'src': src,
                    'tgt': tgt,
                    'dist': None,
                    'dist_type': None,
                    'timestamp': timestamps
                }
                return sample
            
            # Initialize agent tensors
            agents_at_each_timestep = [None] * len(timestamps)
            agent_types = torch.zeros(len(timestamps), self.max_num_agents + len(self.env), dtype=torch.float32)
            distances = torch.zeros(len(timestamps), self.max_num_agents + len(self.env), dtype=torch.float32)

            id = src_time_groups['ID'].unique()[0]
            # Pre-filter DataFrame once to optimize lookup
            df_filtered = self.data[self.data['ID'] != id]  # Exclude the current agent's ID

            # Iterate over timestamps
            for idx, time in enumerate(timestamps):
                # Get all agents at the current timestamp
                agents_at_time = df_filtered[df_filtered['Time'] == time][['X', 'Y', 'Type']].values
                num_agents = min(agents_at_time.shape[0], self.max_num_agents)

                if num_agents > 0:
                    # Extract agent positions and types
                    agents_at_each_timestep[idx] = agents_at_time[:num_agents, :2]
                    agent_types[idx, :num_agents] = torch.tensor(agents_at_time[:num_agents, 2], dtype=torch.float32)

                    # Calculate distances
                    distances[idx, :num_agents] = self.calculations.calculate_distances(
                        src[idx, :2],
                        agents_at_each_timestep[idx]
                    )
            
            # Precompute the object types to avoid checking them in every iteration.
            special_object_types = {9, 10, 11}

            # Precompute the length of the environment and the relevant slices of df_vector
            env_length = len(self.env)
            df_vector_length = len(self.df_vector)

            # Compute distances to environmental objects.
            for i in range(src.shape[0]):
                agent = src[i, :2]  # Current agent position.
                
                # Iterate through environmental objects
                for j in range(env_length):
                    obj = self.env[j, :]

                    # Check if it's a special object type (types 9 or 10)
                    if obj[3] in special_object_types:
                        dist = self.calculations.euclidean_distance(agent[0], agent[1], obj[0], obj[1])
                    else:
                        # Calculate the distance using the appropriate method for other objects
                        index = j - (env_length - df_vector_length)
                        dist = self.calculations.check_coordinate_in_squares(agent[0], agent[1], index, self.df_vector)
                        if dist is None:
                            dist = self.calculations.calculate_distance_to_nearest_vector(agent[0], agent[1], index, self.df_vector)
                    
                    # Efficiently store the distance and object type
                    distances[i, j + self.max_num_agents] = dist
                    agent_types[i, j + self.max_num_agents] = obj[4]
                    
            # Extract timestamps for the source sequence.
            timestamps = data_slice[['Time']].values
            break

        # Update the starting index for the next sample.
        self.start_idx = self.start_idx + self.src_sequence_length

        # Return the sample.
        sample = {
            'src': src,
            'tgt': tgt,
            'dist': distances,
            'dist_type': agent_types,
            'timestamp': timestamps
        }
        return sample