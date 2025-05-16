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
                - 'dist': Combined distances tensor (shape: [src_length, max_num_agents + len(env)]).
                - 'dist_type': Types of surrounding agents and environmental objects (shape: [src_length, max_num_agents + len(env)]).
                - 'dist_agents': Agent-to-agent distances (shape: [src_length, max_num_agents]).
                - 'dist_env': Agent-to-environment distances (shape: [src_length, len(env)]).
                - 'timestamp': Timestamps associated with the source sequence.
        """

        while True:
            src_end_idx = self.start_idx + self.src_sequence_length
            tgt_end_idx = src_end_idx + self.tgt_sequence_length

            if tgt_end_idx > len(self.data) - 1:
                return {'src': None, 'tgt': None, 'dist': None, 'dist_type': None,
                        'dist_agents': None, 'dist_env': None, 'timestamp': None}

            data_slice = self.data.iloc[self.start_idx:tgt_end_idx]
            src_group = data_slice.iloc[:self.src_sequence_length]
            tgt_group = data_slice.iloc[self.src_sequence_length:]

            if src_group['ID'].nunique() > 1 or tgt_group['ID'].nunique() > 1:
                self.start_idx += 1
                continue

            src_feat = src_group[['X','Y','Speed','ID','Type']].values.astype(np.float32)
            tgt_feat = tgt_group[['X','Y','Speed','ID','Type']].values.astype(np.float32)
            timestamps = src_group['Time'].values
            src = torch.tensor(src_feat)
            tgt = torch.tensor(tgt_feat)

            if self.info:
                self.start_idx += self.src_sequence_length
                return {'src': src, 'tgt': tgt, 'dist': None, 'dist_type': None,
                        'dist_agents': None, 'dist_env': None, 'timestamp': timestamps}

            T = len(timestamps)
            A = self.max_num_agents
            E = len(self.env)
            combined_dist = torch.zeros(T, A + E)
            combined_type = torch.zeros(T, A + E)

            current_id = src_group['ID'].iloc[0]
            others = self.data[self.data['ID'] != current_id]

            # Agent-agent distances
            for t_idx, t in enumerate(timestamps):
                agents = others[others['Time'] == t][['X','Y','Type']].values
                if agents.shape[0] > 0:
                    n = min(agents.shape[0], A)
                    pos = agents[:n, :2]
                    combined_type[t_idx, :n] = torch.tensor(agents[:n, 2], dtype=torch.float32)
                    combined_dist[t_idx, :n] = self.calculations.calculate_distances(src[t_idx, :2], pos)

            # Agent-env distances
            special = {9, 10, 11}
            dfv_len = len(self.df_vector)
            for t_idx in range(T):
                agent_pos = src[t_idx, :2]
                for j, obj in enumerate(self.env):
                    obj_type = obj[4]
                    if obj[3] in special:
                        d = self.calculations.euclidean_distance(
                            agent_pos[0], agent_pos[1], obj[0], obj[1]
                        )
                    else:
                        idx_vec = j - (E - dfv_len)
                        d = self.calculations.check_coordinate_in_squares(
                            agent_pos[0], agent_pos[1], idx_vec, self.df_vector
                        )
                        if d is None:
                            d = self.calculations.calculate_distance_to_nearest_vector(
                                agent_pos[0], agent_pos[1], idx_vec, self.df_vector
                            )
                    combined_dist[t_idx, A + j] = d
                    combined_type[t_idx, A + j] = obj_type

            # Full combined distances / types
            dist_combined = combined_dist.clone()
            type_combined = combined_type.clone()

            # Agent–agent only (env dims zeroed)
            dist_agents_only = combined_dist.clone()
            dist_agents_only[:, A:] = 0
            type_agents_only = combined_type.clone()
            type_agents_only[:, A:] = 0

            # Agent–env only (agent dims zeroed)
            dist_env_only = combined_dist.clone()
            dist_env_only[:, :A] = 0
            type_env_only = combined_type.clone()
            type_env_only[:, :A] = 0

            # Prepare output
            self.start_idx += self.src_sequence_length
            return {
                'src': src,
                'tgt': tgt,
                'dist': dist_combined,
                'dist_type': type_combined,
                'dist_agents': dist_agents_only,
                'type_agents': type_agents_only,
                'dist_env': dist_env_only,
                'type_env': type_env_only,
                'timestamp': timestamps
            }
