import gc
import json

import torch
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Import custom classes for data handling, visualization, distance calculations, and scaling
from classes.location import Location
from classes.visualization import Visualization
from classes.distances import Distances
from classes.scaler import Scaler

# Import custom dataset and experiment-related classes
from classes.Datasets.trajectory_dataset import TrajectoryDataset
from classes.Datasets.transformer_dataset import TransformerDataset
from classes.Datasets.star_dataset import STARDataset
from classes.Datasets.saestar_dataset import SAESTARDataset
from classes.Datasets.seastar_dataset import SEASTARDataset
from classes.Experiments.experiment import Experiment

class ExperimentManager:
    """
    Manages the lifecycle of experiments, including data preparation, model training, and evaluation.
    """

    def __init__(
        self, location: Location, 
        visualization: Visualization, 
        epochs: int, 
        learning_rate: float, 
        num_layers: int, 
        num_heads: int, 
        dropout: float, 
        src_len: int, 
        tgt_len: int, 
        batch_size: int,
        hidden_size: int,
        earlystopping : int,
        seed
    ):
        """
        Initialize the ExperimentManager with necessary configurations and perform initial visualizations.
        
        Parameters:
        - location: Location object with dataset details.
        - visualization: Visualization object for data insights.
        - epochs: Number of training epochs.
        - learning_rate: Learning rate for optimization.
        - num_layers: Number of layers in the model.
        - num_heads: Number of attention heads in the model.
        - dropout: Dropout rate for regularization.
        - src_len: Source sequence length for the model.
        - tgt_len: Target sequence length for the model.
        - batch_size: Batch size for training.
        - hidden_size: Hidden layer size in the model.
        """
        self.location = location
        self.viz = visualization
        self.epochs = epochs
        self.lr = learning_rate
        self.layers = num_layers
        self.heads = num_heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.earlystopping = earlystopping
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.seed = seed
        
        # Perform initial visualizations for trajectory, time distribution, and counts
        self.viz.visualize_all_trajectories()
        self.viz.visualize_time()
        self.viz.visualize_counts()
        self.viz.visualize_vectors()

        # Process and prepare the dataset for experiments
        self.process_data()
            
    def process_data(self):
        """
        Check if preprocessed data exists; if not, create datasets and save them.
        """
        if self.check_data():
            return  # Data is already processed, no further action required.
        
        print(f"Creating train, val, and test splits. This may take a while.")
        train, val, test = self.create_trajectory_dataset()
        self.create_timestamp_df(test)  # Generate additional test set metadata if needed.
        self.save_tensor(train, "Train")  # Save training data tensors.
        self.save_tensor(val, "Val")  # Save validation data tensors.
        self.save_tensor(test, "Test")  # Save testing data tensors.
        
        # Free up memory after data preparation
        del train, val, test
        gc.collect()
    
    def check_data(self) -> bool:
        """
        Check if preprocessed data tensors already exist.

        Returns:
        - True if all required data tensors are found, False otherwise.
        """
        split_types = ["Train", "Val", "Test"]
        data_types = ['src', 'tgt', 'dist', 'dist_type']
            
        for split_type in split_types:
            for data_type in data_types:
                file = Path(f"./data/Datasets/{self.location.location_name}/{split_type}/{data_type}.pt")
                
                if not file.exists():  # If any required file is missing, return False.
                    print(f"{file} does not exist")
                    return False
        
        # Cleanup and memory management
        del split_types, data_types
        gc.collect()
        return True
    
    def create_trajectory_dataset(self):
        """
        Create datasets for trajectory prediction using location and environmental data.

        Returns:
        - Train, validation, and test datasets split from the full trajectory dataset.
        """
        # Ensure chronological sorting of data by time
        self.location.df['Time'] = pd.to_datetime(self.location.df['Time'])
        self.location.df = self.location.df.sort_values(by=['ID', 'Time'])
        
        # Generate environmental data
        env_df = self.create_env_df()
        
        # Compute distance-related objects using environmental data
        distance_objects = self.create_distance_objects(env_df=env_df)
        
        # Construct the trajectory dataset
        trajectory_dataset = TrajectoryDataset(
            data=self.location.df,
            df_vector=self.location.env_vectors,
            env_df=env_df,
            distance_objects=distance_objects,
            dataset_len=self.location.dataset_length, 
            max_num_agents=self.location.num_agents, 
            src_length=self.src_len,
            tgt_length=self.tgt_len
        )
        
        # Split the dataset into training, validation, and testing sets
        train_data, test_data = train_test_split(trajectory_dataset, test_size=0.2, random_state=42, shuffle=False)
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, shuffle=False)
        
        # Memory cleanup
        del env_df, distance_objects
        gc.collect()
        return train_data, val_data, test_data

    def create_env_df(self):
        """
        Create a DataFrame for environmental objects, including intersections, object types, and coordinates.

        Returns:
        - A DataFrame containing environmental features such as object type, position, and speed.
        """
        calculate = Distances()
        
        # Create a DataFrame from environment polygons and compute intersections
        df = pd.DataFrame(self.location.env_polygons)
        df['Intersection'] = df.apply(calculate.calculate_intersection, axis=1)
        
        # Filter valid intersection points and encode object types
        df_filtered = df.dropna(subset=['Intersection'])
        df_filtered['Type'] = pd.Categorical(df_filtered['Type']).codes + 4
        df_filtered['X'] = df_filtered['Intersection'].apply(lambda x: x[0])
        df_filtered['Y'] = df_filtered['Intersection'].apply(lambda x: x[1])
        
        # Compute width to the nearest vector for environmental objects
        widths = [
            calculate.calculate_width_to_nearest_vector(row['X'], row['Y'], df_filtered)
            for _, row in df_filtered.iterrows()
        ]
        df_filtered['Width_to_Nearest_Vector'] = widths
        
        # Add static objects (e.g., signs, trees) to the dataset
        df_static_objects = pd.DataFrame(self.location.static_objects)
        df_objects = [
            [10, row["Top_Left"][0], row["Top_Left"][1]] if row["Type"] == "Sign" else
            [11, row["Top_Left"][0], row["Top_Left"][1]] if row["Type"] == "Tree" else
            [9, row["Top_Left"][0], row["Top_Left"][1]]
            for _, row in df_static_objects.iterrows()
        ]
        df_objects = pd.DataFrame(df_objects, columns=['Type', 'X', 'Y'])
        df_objects['Width_to_Nearest_Vector'] = 0
        df_objects['ID'] = -1

        # Combine filtered environmental data and static object data
        df_filtered['ID'] = -2
        env_df = pd.concat([df_objects, df_filtered], ignore_index=True)
        env_df.rename(columns={'Width_to_Nearest_Vector': 'Speed'}, inplace=True)
        env_df = env_df[['X', 'Y', 'Speed', 'ID', 'Type']]
        
        # Store the number of unique object types
        self.num_objects = len(env_df['Type'].unique())
        env_df.to_csv(f'./data/CombinedData/{self.location.location_name}/env_df.csv', index=False)
        return env_df

    def create_distance_objects(self, env_df):
        """
        Compute distance-related tensors for environmental objects.

        Parameters:
        - env_df: DataFrame containing environmental objects.

        Returns:
        - Tensor representing Euclidean distances between objects.
        """
        calculate = Distances()
        tensor = torch.tensor(env_df.iloc[len(self.location.static_objects['Type']) + 1:, :].values)
        tensor = tensor.view(tensor.shape[0], 1, 5).repeat(1, 10, 1)
        distance_objects = calculate.compute_euclidean_distance_objects(tensor[:, :, :2])
        
        # Cleanup memory after tensor operations
        del tensor
        gc.collect()
        return distance_objects


    def create_timestamp_df(self, data):
        """
        Converts a list of timestamp entries into a Pandas DataFrame, saves it as a CSV file, and cleans up memory.
        
        Args:
            data (list[dict]): A list of dictionaries, each containing a 'timestamp' key with its corresponding value.
        
        Side Effects:
            - Saves the DataFrame as a CSV file in the location-specific directory.
            - Deletes the DataFrame object and triggers garbage collection to free memory.
        """
        # Extracting timestamps and flattening the array for easier handling
        test_times = np.array([sample['timestamp'] for sample in data])
        times = test_times.flatten()
        
        # Creating a DataFrame with a 'Time' column and ensuring it is properly formatted as datetime
        times_df = pd.DataFrame(times, columns=['Time'])
        times_df['Time'] = pd.to_datetime(times_df['Time'])
        
        # Saving the DataFrame to a CSV file in the appropriate directory
        times_df.to_csv(f'./data/CombinedData/{self.location.location_name}/test_times.csv', index=False)
        
        # Freeing up memory
        del times_df
        gc.collect()
    
    def save_tensor(self, data : list[dict], data_type: str):
        """
        Converts data entries into PyTorch tensors and saves them to disk for a specified data type.
        
        Args:
            data (list[dict]): A list of dictionaries containing data for keys 'src', 'tgt', 'dist', and 'dist_type'.
            data_type (str): The type of data (e.g., 'train', 'validation', 'test') to distinguish saved files.
        
        Side Effects:
            - Saves tensors to the disk in the location-specific directory.
            - Deletes tensor objects and triggers garbage collection to free memory.
        """
        print(f"Saving {data_type} for Location: {self.location.location_name}")
        src = np.array([sample['src'] for sample in data], dtype=np.float64)
        tgt = np.array([sample['tgt'] for sample in data], dtype=np.float64)
        dist = np.array([sample['dist'] for sample in data], dtype=np.float64)
        dist_type = np.array([sample['dist_type'] for sample in data], dtype=np.float64)
        dist_agents = np.array([sample['dist_agents'] for sample in data], dtype=np.float64)
        dist_env = np.array([sample['dist_env'] for sample in data], dtype=np.float64)

        # Extracting data components and converting them into tensors
        src = torch.tensor(src, dtype=torch.float64).to(torch.float32)
        tgt = torch.tensor(tgt, dtype=torch.float64).to(torch.float32)
        dist = torch.tensor(dist, dtype=torch.float64).to(torch.float32)
        dist_type = torch.tensor(dist_type, dtype=torch.float64).to(torch.float32)
        dist_agents = torch.tensor(dist_agents, dtype=torch.float64).to(torch.float32)
        dist_env = torch.tensor(dist_env, dtype=torch.float64).to(torch.float32)
        # Saving tensors to disk
        torch.save(src, f"./data/Datasets/{self.location.location_name}/{data_type}/src.pt")
        torch.save(tgt, f"./data/Datasets/{self.location.location_name}/{data_type}/tgt.pt")
        torch.save(dist, f"./data/Datasets/{self.location.location_name}/{data_type}/dist.pt")
        torch.save(dist_type, f"./data/Datasets/{self.location.location_name}/{data_type}/dist_type.pt")
        torch.save(dist_env, f"./data/Datasets/{self.location.location_name}/{data_type}/dist_env.pt")
        torch.save(dist_agents, f"./data/Datasets/{self.location.location_name}/{data_type}/dist_agents.pt")
        
        
        # Freeing up memory
        del src, tgt, dist, dist_type
        gc.collect()
    
    def load_tensors(self, model_name : str, data_type : str, spatial : bool=False, sea_star : bool=False):
        """
        Loads tensors from disk for a given model and data type, optionally including spatial data.

        Args:
            model_name (str): The name of the model (used for logging).
            data_type (str): The type of data to load (e.g., 'train', 'validation', 'test').
            spatial (bool): Whether to include spatial data ('dist' and 'dist_type') in the returned tensors.
        
        Returns:
            dict: A dictionary containing loaded tensors, with keys 'src', 'tgt', and optionally 'distance', 'type'.
        """
        print(f"Loading {data_type} for model {model_name} at Location: {self.location.location_name}")
        
        # Loading mandatory tensors
        src = torch.load(f"./data/Datasets/{self.location.location_name}/{data_type}/src.pt", weights_only=True)
        tgt = torch.load(f"./data/Datasets/{self.location.location_name}/{data_type}/tgt.pt", weights_only=True)
        
        if spatial:
            # Loading additional spatial data if requested
            dist = torch.load(f"./data/Datasets/{self.location.location_name}/{data_type}/dist.pt", weights_only=True)
            d_type = torch.load(f"./data/Datasets/{self.location.location_name}/{data_type}/dist_type.pt", weights_only=True)
            if sea_star:
                dist_env = torch.load(f"./data/Datasets/{self.location.location_name}/{data_type}/dist_env.pt", weights_only=True)
                dist_agents = torch.load(f"./data/Datasets/{self.location.location_name}/{data_type}/dist_agents.pt", weights_only=True)
                return {'src': src, 'tgt': tgt, 'distance': dist, 'type': d_type, 'dist_env': dist_env, 'dist_agents': dist_agents}
            return {'src': src, 'tgt': tgt, 'distance': dist, 'type': d_type}
        else:
            return {'src': src, 'tgt': tgt}
    
    def experiment_base(self):
        """
        Runs the base experiment, handling data preparation and experiment execution.
        """
        model_name = 'Base'
        print("Base Experiment Starting")
        print("-" * 30)
        
        # Loading and preparing data
        train = self.load_tensors(model_name=model_name, data_type="Train", spatial=False)
        val = self.load_tensors(model_name=model_name, data_type="Val", spatial=False)
        test = self.load_tensors(model_name=model_name, data_type="Test", spatial=False)
        train_dataset = TransformerDataset(train)
        val_dataset = TransformerDataset(val)
        test_dataset = TransformerDataset(test)
        
        # Scaling data and creating DataLoaders
        scaler = Scaler(train, False)
        train_dataloader, val_dataloader, test_dataloader = self.get_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Running the experiment
        self.run_experiment(model_name, scaler, train_dataloader, val_dataloader, test_dataloader, num_types=None, graph_dims=None)
        
        # Cleaning up memory
        del scaler, train, val, test, train_dataloader, test_dataloader, val_dataloader
        gc.collect()
    
    def experiment_transformer(self):
        """
        Runs the Transformer experiment, handling data preparation and experiment execution.
        """
        model_name = 'Transformer'
        print("Transformer Experiment Starting")
        print("-" * 30)
        
        # Loading and preparing data
        train = self.load_tensors(model_name=model_name, data_type="Train", spatial=False)
        val = self.load_tensors(model_name=model_name, data_type="Val", spatial=False)
        test = self.load_tensors(model_name=model_name, data_type="Test", spatial=False)
        train_dataset = TransformerDataset(train)
        val_dataset = TransformerDataset(val)
        test_dataset = TransformerDataset(test)
        
        # Scaling data and creating DataLoaders
        scaler = Scaler(train, False)
        train_dataloader, val_dataloader, test_dataloader = self.get_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Running the experiment
        self.run_experiment(model_name, scaler, train_dataloader, val_dataloader, test_dataloader, num_types=None, graph_dims=None)
        
        # Cleaning up memory
        del scaler, train, val, test, train_dataloader, test_dataloader, val_dataloader
        gc.collect()
    
    def experiment_star(self):
        """
        Runs the STAR experiment, handling data preparation and experiment execution.
        """
        model_name = 'STAR'
        print("STAR Experiment Starting")
        print("-" * 30)
        
        # Loading and preparing data
        train = self.load_tensors(model_name=model_name, data_type="Train", spatial=True)
        val = self.load_tensors(model_name=model_name, data_type="Val", spatial=True)
        test = self.load_tensors(model_name=model_name, data_type="Test", spatial=True)
        graph_dims = len(train['distance'][0, 0, :self.location.num_agents])
        train_dataset = STARDataset(train, self.location.num_agents)
        val_dataset = STARDataset(val, self.location.num_agents)
        test_dataset = STARDataset(test, self.location.num_agents)
        
        #print(train_dataset[0])
        # Scaling data and creating DataLoaders
        scaler = Scaler(train, model_name, True, graph_dims)
        train_dataloader, val_dataloader, test_dataloader = self.get_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Running the experiment
        self.run_experiment(model_name, scaler, train_dataloader, val_dataloader, test_dataloader, num_types=None, graph_dims=graph_dims)
        
        # Cleaning up memory
        del scaler, train, val, test, train_dataloader, test_dataloader, val_dataloader
        gc.collect()
    
    def experiment_saestar(self):
        """
        Runs the SAESTAR experiment, handling data preparation and experiment execution.
        """
        model_name = 'SAESTAR'
        print("SAESTAR Experiment Starting")
        print("-" * 30)
        
        # Loading and preparing data
        train = self.load_tensors(model_name=model_name, data_type="Train", spatial=True)
        val = self.load_tensors(model_name=model_name, data_type="Val", spatial=True)
        test = self.load_tensors(model_name=model_name, data_type="Test", spatial=True)
        df_env = pd.read_csv(f"./data/CombinedData/{self.location.location_name}/env_df.csv", sep=',')
        num_types = df_env['Type'].max()
        graph_dims = len(train['distance'][0, 0, :])
        train_dataset = SAESTARDataset(train)
        ### remove later
        #np.savetxt("train_data.txt", train_dataset[0], fmt='%s')
        #print(train_dataset[1])
        val_dataset = SAESTARDataset(val)
        test_dataset = SAESTARDataset(test)
        
        # Scaling data and creating DataLoaders
        scaler = Scaler(train, model_name, True)
        train_dataloader, val_dataloader, test_dataloader = self.get_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Running the experiment
        self.run_experiment(model_name, scaler, train_dataloader, val_dataloader, test_dataloader, num_types=num_types, graph_dims=graph_dims)
        
        # Cleaning up memory
        del scaler, train, val, test, train_dataloader, test_dataloader, val_dataloader
        gc.collect()
            
    def experiment_seastar(self, device):
        """
        Runs the SEASTAR experiment, handling data preparation and experiment execution.
        """
        model_name = 'SEASTAR'
        print("SEASTAR Experiment Starting")
        print("-" * 30)
        
        # Loading and preparing data
        train = self.load_tensors(model_name=model_name, data_type="Train", spatial=True, sea_star=True)
        val = self.load_tensors(model_name=model_name, data_type="Val", spatial=True, sea_star=True)
        test = self.load_tensors(model_name=model_name, data_type="Test", spatial=True, sea_star=True)
        df_env = pd.read_csv(f"./data/CombinedData/{self.location.location_name}/env_df.csv", sep=',')
        max_label = int(df_env['Type'].max())
        num_types = max_label + 1   # ensure embedding can index up to max_label
        # print(num_types)
        # print("blalbla")
        #num_types = 4 + len(df_env['ID'].unique())
        #print(num_types)
        graph_dims = len(train['distance'][0, 0, :])
        train_dataset = SEASTARDataset(train)
        val_dataset = SEASTARDataset(val)
        test_dataset = SEASTARDataset(test)
        
        # Scaling data and creating DataLoaders
        scaler = Scaler(train, model_name, True)
        train_dataloader, val_dataloader, test_dataloader = self.get_data_loaders(train_dataset, val_dataset, test_dataset)
        #print("blalbl")
        # Running the experiment
        self.run_experiment(model_name, scaler, train_dataloader, val_dataloader, test_dataloader, num_types=num_types, graph_dims=graph_dims)
        
        # Cleaning up memory
        del scaler, train, val, test, train_dataloader, test_dataloader, val_dataloader
        gc.collect()
    
    
    def run_experiment(self, model_name, scaler, train, val, test, num_types, graph_dims):
        """
        Executes the training and testing of the specified model.
        
        Args:
            model_name (str): The name of the model being tested.
            scaler (Scaler): An object to scale and normalize the data.
            train, val, test (DataLoader): DataLoaders for the respective datasets.
            num_types (int): The number of types or classes (if applicable).
            graph_dims (int): The dimensionality of graph data (if applicable).
        """
        print(f"{model_name} Experiment Initiating")
        
        # Initializing the experiment with the provided configuration
        experiment = Experiment(
            scaler=scaler,
            model_name=model_name,
            src_len=self.src_len,
            tgt_len=self.tgt_len,
            num_types=num_types,
            graph_dims=graph_dims,
            layers=self.layers,
            heads=self.heads,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            lr=self.lr,
            epochs=self.epochs,
            location_name=self.location.location_name,
            earlystopping=self.earlystopping,
            seed=self.seed
        )
        
        # Training and testing the model
        experiment.train_model(train, val)
        experiment.test_model(test)
        
        # Cleaning up memory
        del experiment
        gc.collect()
    
    def get_data_loaders(self, train : Dataset, val : Dataset, test : Dataset):
        """
        Creates DataLoader objects for training, validation, and testing datasets.
        
        Args:
            train, val, test (Dataset): Datasets for training, validation, and testing.
        
        Returns:
            tuple[DataLoader]: DataLoader objects for training, validation, and testing.
        """
        train_dataloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        
        return train_dataloader, val_dataloader, test_dataloader
    

