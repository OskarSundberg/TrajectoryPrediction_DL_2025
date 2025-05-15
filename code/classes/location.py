import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from classes.Datasets.trajectory_dataset import TrajectoryDataset


class Location:
    """
    A class to handle spatial data, process trajectories, and generate embeddings dims for a specific location.
    
    Attributes:
        min_x, max_x, min_y, max_y (float): Boundaries of the location in meters.
        location_name (str): Name of the location, used for file paths and image loading.
        env_vectors, env_polygons (list): Environmental features (vectors and polygons) for the location.
        static_objects (list): Static objects within the location.
        roi (list): Region of interest coordinates.
        img (str): Path to the base image of the location.
        metadata (str): Path to save metadata for the processed data.
        size_X, size_Y (float): Dimensions of the location in meters.
        img_size_X, img_size_y (int): Dimensions of the image in pixels.
        origin_X, origin_Y (float): Calculated origin points for aligning the image and data.
        scale_X, scale_y (float): Scaling factors for transforming coordinates from meters to pixels.
        df (DataFrame): Processed trajectory data for the location.
        num_agents (int): Maximum number of agents in any single frame.
        dataset_length (int): Length of the processed dataset.
        embedding_dims (list): Dimensions of the embeddings for the data.
    """

    def __init__(self, min_x, max_x, min_y, max_y, location_name, env_vectors, env_polygons, static_objects, roi):
        """
        Initializes the Location instance and processes necessary metadata for trajectory data.

        Parameters:
            min_x, max_x, min_y, max_y (float): Location boundaries in meters.
            location_name (str): Name of the location.
            env_vectors, env_polygons (list): Environmental features for the location.
            static_objects (list): Static objects in the location.
            roi (list): Region of interest.
        """
        self.location_name = location_name
        self.env_vectors = pd.DataFrame(env_vectors)
        self.env_polygons = env_polygons
        self.static_objects = static_objects
        self.roi = roi
        self.img = f'./data/BaseImages/{location_name}.png'
        self.metadata = f'./data/CombinedData/{self.location_name}/metadata.json'
        
        base = plt.imread(self.img)
        
        # Calculate location dimensions in meters
        self.size_X = max_x - min_x
        self.size_Y = max_y - min_y
        
        print("SIZE: ", self.size_X, self.size_Y)
        
        
        # Get image dimensions in pixels
        self.img_size_X = base.shape[1]
        self.img_size_y = base.shape[0]
        print("IMG SIZE: ", self.img_size_X, self.img_size_y)
        
        del base
        gc.collect()

        # Compute origin and scaling factors for coordinate transformations
        self.origin_X = self.img_size_X * (-min_x) / self.size_X
        self.origin_Y = self.img_size_y * (-min_y) / self.size_Y
        print("ORIGIN: ", self.origin_X, self.origin_Y)
        
        self.scale_X = self.img_size_X / self.size_X
        self.scale_y = self.img_size_y / self.size_Y
        
        print("SCALE: ", self.scale_X, self.scale_y) 
        
        file = Path(f'./data/CombinedData/{location_name}/data.csv')
        
        if file.exists():
            # Retrieve trajectory data
            self.df = pd.read_csv(file, sep=',')
            self.num_agents = self.max_num_agents()
        else:
            # Process trajectory data
            self.df = self.create_dataset()
            self.num_agents = self.max_num_agents()
            
        file = Path(self.metadata)
        if file.exists():
            # Retrieve Metadata
            self.dataset_length, self.embedding_dims = self.get_data_info()
        else:
            # Process Metadata
            self.dataset_length, self.embedding_dims = self.dataset_info()
            
        self.df = self.df.sort_values(by=['ID', 'Time'])
        
        

    def combine_datasets(self):
        """
        Combines multiple days of trajectory data into a single DataFrame.

        Returns:
            DataFrame: Concatenated DataFrame containing data from all days.
        """
        # Read and combine data files
        df1 = pd.read_csv(f'./data/Original/{self.location_name}/2023-05-23.csv', sep=';')
        df2 = pd.read_csv(f'./data/Original/{self.location_name}/2023-05-24.csv', sep=';')
        df3 = pd.read_csv(f'./data/Original/{self.location_name}/2023-05-25.csv', sep=';')
        df4 = pd.read_csv(f'./data/Original/{self.location_name}/2023-05-26.csv', sep=';')
        
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        del df1, df2, df3, df4
        gc.collect()
        return df.head(500000)

    def create_dataset(self):
        """
        Processes and transforms raw trajectory data into a usable format.

        Returns:
            DataFrame: Processed DataFrame containing trajectory data with scaled coordinates and floored timestamps.
        """
        # Combine datasets
        data = self.combine_datasets()

        # Preprocess column names and drop unnecessary ones
        data = data.rename(columns={'Speed, km/h': 'Speed'})
        data = data.drop(columns=['ID in database', 'Station', 'Estimated', 'Vx, m/s', 'Vy, m/s'])

        print("Preprocessing Data")

        # Convert Time to datetime and floor to 100ms in a vectorized way
        data['Time'] = pd.to_datetime(data['Time'], format='mixed', utc=True).dt.tz_localize(None).dt.floor('100ms')

        # Scale coordinates directly
        data['X'] = self.scale_X * data['X'] + self.origin_X
        data['Y'] = self.scale_y * data['Y'] + self.origin_Y

        # Group by ID and Time, calculate mean for numeric columns
        grouped = data.groupby(['ID', 'Time'], as_index=False).agg({
            'X': 'mean',
            'Y': 'mean',
            'Speed': 'mean',
            'Type': 'first'
        })

        # Sort values by Time
        grouped = grouped.sort_values(by='Time')

        # Optimize memory by converting to lower-precision types
        grouped['X'] = grouped['X'].astype('float32')
        grouped['Y'] = grouped['Y'].astype('float32')
        grouped['Speed'] = grouped['Speed'].astype('float32')
        grouped['ID'] = grouped['ID'].astype('int')
        grouped['Type'] = grouped['Type'].astype('int')

        # Save the processed data
        output_path = f'./data/CombinedData/{self.location_name}/data.csv'
        grouped.to_csv(output_path, index=False)

        return grouped

    def max_num_agents(self):
        """
        Finds the maximum number of agents present in a single frame.

        Returns:
            int: Maximum number of agents in any given timestamp.
        """
        timestamp_counts = self.df.groupby('Time').size()
        max_count = timestamp_counts.max()
        del timestamp_counts
        gc.collect()
        return max_count

    def dataset_info(self):
        """
        Calculates metadata about the dataset, including its length and embedding dimensions.

        Returns:
            tuple: (Dataset length, Embedding dimensions as a list).
        """
        self.df = self.df.sort_values(by=['ID', 'Time'])
        trajectory_dataset = TrajectoryDataset(self.df, dataset_len=len(self.df), info=True)
        print("Creating metadata json file")
        # Calculate embedding dimensions
        x = self.df['X'].max() - self.df['X'].min()
        y = self.df['Y'].max() - self.df['Y'].min()
        speed_max = self.df['Speed'].max()
        embedding_dims = [x, y, speed_max, self.num_agents + 1]
        
        dataset_length = 0
        
        for i in tqdm(range(len(trajectory_dataset)), total=len(trajectory_dataset), ncols=100):
            sample = trajectory_dataset[i]
            if sample['src'] is None:
                print("Finished")
                dataset_length = i
                # Save metadata
                metadata = {
                    "Dataset Length": int(dataset_length),
                    "Embedding Dim X": float(embedding_dims[0]),
                    "Embedding Dim Y": float(embedding_dims[1]),
                    "Embedding Dim Speed": float(embedding_dims[2]),
                    "Embedding Dim Num Agents": int(self.num_agents)
                }
                break
        
        
        with open(self.metadata, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        embedding_dims = np.array(embedding_dims)
        
        return dataset_length, embedding_dims

    def get_data_info(self):
        """
        Reads and returns metadata information about the dataset.

        Returns:
            tuple: (Dataset length, Embedding dimensions as a list).
        """
        with open(self.metadata, 'r') as json_file:
            metadata = json.load(json_file)
            
        dataset_length = metadata.get("Dataset Length")
        embedding_dims = [
            metadata.get("Embedding Dim X"),
            metadata.get("Embedding Dim Y"),
            metadata.get("Embedding Dim Speed"),
            metadata.get("Embedding Dim Num Agents")
        ]
        
        embedding_dims = np.array(embedding_dims)
        return dataset_length, embedding_dims
