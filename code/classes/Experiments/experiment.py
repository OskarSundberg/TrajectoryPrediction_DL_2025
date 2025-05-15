import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

from classes.scaler import Scaler
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from classes.Models.base_model import TransformerBase
from classes.Models.transformer_model import Transformer
from classes.Models.star_model import STAR
from classes.Models.saestar_model import SAESTAR
from classes.Models.seastar_model import SEASTAR


import torch
from torchviz import make_dot

class Experiment:
    """
    A class to handle the training and evaluation of various models for time series prediction.

    Args:
        scaler (Scaler): A Scaler object to normalize input data.
        model_name (str): The model architecture to use. Choices: 'Base', 'Transformer', 'STAR', 'SAESTAR'.
        src_len (int): The length of the source sequence.
        tgt_len (int): The length of the target sequence.
        num_types (int, optional): Number of types (used in SAESTAR model). Defaults to None.
        graph_dims (int, optional): Dimension of the graph (used in STAR and SAESTAR models). Defaults to None.
        layers (int, optional): Number of layers in the model. Defaults to 16.
        heads (int, optional): Number of attention heads. Defaults to 8.
        hidden_size (int, optional): The hidden layer size. Defaults to 256.
        dropout (float, optional): The dropout rate for regularization. Defaults to 0.1.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.000015.

    Attributes:
        scaler (Scaler): The Scaler object for input normalization.
        model_type (str): The type of model to use ('Base', 'Transformer', 'STAR', 'SAESTAR').
        model (nn.Module): The model object (e.g., Transformer, STAR, etc.).
        optimizer (Adam): The optimizer used for training.
        criterion (nn.Module): The loss function (MSELoss).
    """

    def __init__(
        self,
        scaler: Scaler,
        model_name: str, 
        src_len: int, 
        tgt_len: int, 
        num_types: int = None, 
        graph_dims: int = None, 
        layers: int = 16, 
        heads: int = 8, 
        hidden_size: int = 256, 
        dropout: float = 0.1, 
        lr: float = 0.000015,
        epochs : int = 10,
        location_name : str='',
        earlystopping : int=30
    ):
        """
        Initializes the Experiment class, selects the appropriate model, and prepares the optimizer.

        Args are as described in the class docstring.
        """
        self.scaler = scaler
        self.epochs = epochs
        self.model_type = model_name
        self.heads = heads
        self.criterion = nn.MSELoss()
        self.location_name = location_name
        self.early_stopping_patience = earlystopping
        self.optimizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Select the model architecture based on the model_name
        if model_name == "Base":
            self.model = self.base_model(src_len, tgt_len, hidden_size, layers, heads, dropout)
        if model_name == "Transformer":
            self.model = self.transformer_model(src_len, tgt_len, hidden_size, layers, heads, dropout)
        if model_name == "STAR":
            self.model = self.star_model(src_len, tgt_len, graph_dims, hidden_size, layers, heads, dropout)
        if model_name == "SAESTAR":
            self.model = self.saestar_model(src_len, tgt_len, graph_dims, num_types, hidden_size, layers, heads, dropout)
        if model_name == "SEASTAR":
            self.model = self.seastar_model(src_len, tgt_len, graph_dims, num_types, hidden_size, layers, heads, dropout) 
                
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        #self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)  # You may adjust momentum as needed

        # Initialize the scheduler: decays the LR by 5% every epoch
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

    def create_mask(self, batch_size, sequence_length):
        """
        Creates a look-ahead mask for the sequence to prevent attending to future tokens.
        
        Args:
            batch_size (int): The size of the batch (number of sequences).
            sequence_length (int): The length of the sequence to create the mask for.
        
        Returns:
            torch.Tensor: The look-ahead mask with shape (batch_size, sequence_length, sequence_length).
        """
        
        # Create a look-ahead mask (upper triangular matrix with 1s above the diagonal)
        look_ahead_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
        
        # Expand the mask to match the batch size
        look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(batch_size * self.heads, -1, -1)  # Shape: (batch_size, sequence_length, sequence_length)
        return look_ahead_mask
    
   
    def generate_embedded_dim(self, total, array_length):
        """
        Generates a list of dimensions that sum up to a total value, ensuring the sum of the list equals the total.
        
        Args:
            total (int): The total value to split into an array.
            array_length (int): The length of the array to generate.
        
        Returns:
            list: A list of dimensions.
        """
        initial_value = total // array_length
        remainder = total % array_length
        array = [initial_value] * array_length
        array[-1] += remainder
        
        return torch.tensor(array, dtype=torch.int).to(self.device)
    
    def base_model(self, src, tgt, hidden, layers, heads, dropout) -> nn.Module:
        """
        Creates and returns a Base model.
        
        Args:
            src (int): The source sequence length.
            tgt (int): The target sequence length.
            hidden (int): The hidden layer size.
            layers (int): The number of layers in the model.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        
        Returns:
            nn.Module: The instantiated Base model.
        """
        embedding_dims = [128, 128]
        return TransformerBase(
            embedding_dims, 
            src_len=src, 
            tgt_len=tgt, 
            hidden=hidden, 
            num_layers=layers, 
            num_heads=heads, 
            dropout=dropout
        )
    
    def transformer_model(self, src, tgt, hidden, layers, heads, dropout) -> nn.Module:
        """
        Creates and returns a Transformer model.
        
        Args:
            src (int): The source sequence length.
            tgt (int): The target sequence length.
            hidden (int): The hidden layer size.
            layers (int): The number of layers in the model.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        
        Returns:
            nn.Module: The instantiated Transformer model.
        """
        embedding_dims_src = [94, 94, 64, 4]
        embedding_dims_tgt = [128, 128]   
        return Transformer(
            embedding_dims_src, 
            embedding_dims_tgt, 
            num_agents=4, 
            num_layers=layers, 
            num_heads=heads, 
            hidden=hidden, 
            src_len=src, 
            tgt_len=tgt, 
            dropout=dropout
        )

    def star_model(self, src, tgt, graph_dims, hidden, layers, heads, dropout) -> nn.Module:
        """
        Creates and returns a STAR model.
        
        Args:
            src (int): The source sequence length.
            tgt (int): The target sequence length.
            graph_dims (int): The graph dimensions.
            hidden (int): The hidden layer size.
            layers (int): The number of layers in the model.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        
        Returns:
            nn.Module: The instantiated STAR model.
        """
        src_dims = [94, 94, 64, 4]
        total = sum(src_dims)
        dist_dims = self.generate_embedded_dim(total / 2, graph_dims) 
        type_dims = self.generate_embedded_dim(total / 2, graph_dims)
        return STAR(src_dims, dist_dims, type_dims, 4, hidden, layers, heads, src, tgt, dropout)

    def saestar_model(self, src, tgt, graph_dims, num_types, hidden, layers, heads, dropout) -> nn.Module:
        """
        Creates and returns a SAESTAR model.
        
        Args:
            src (int): The source sequence length.
            tgt (int): The target sequence length.
            graph_dims (int): The graph dimensions.
            num_types (int): The number of types in the model.
            hidden (int): The hidden layer size.
            layers (int): The number of layers in the model.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        
        Returns:
            nn.Module: The instantiated SAESTAR model.
        """
        src_dims = [94, 94, 64, 4]
        total = sum(src_dims)
        dist_dims = self.generate_embedded_dim(total / 2, graph_dims) 
        type_dims = self.generate_embedded_dim(total / 2, graph_dims) 
        return SAESTAR(src_dims, dist_dims, type_dims, 4, num_types, hidden, layers, heads, src, tgt, dropout)
    

    def seastar_model(self, src, tgt, graph_dims, num_types, hidden, layers, heads, dropout) -> nn.Module:
        """
        Creates and returns a SEASTAR model.
        
        Args:
            src (int): The source sequence length.
            tgt (int): The target sequence length.
            graph_dims (int): The graph dimensions.
            num_types (int): The number of types in the model.
            hidden (int): The hidden layer size.
            layers (int): The number of layers in the model.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        
        Returns:
            nn.Module: The instantiated SAESTAR model.
        """
        src_dims = [94, 94, 64, 4]
        total = sum(src_dims)
        dist_dims = self.generate_embedded_dim(total / 2, graph_dims) 
        type_dims = self.generate_embedded_dim(total / 2, graph_dims) 
        return SEASTAR(src_dims, dist_dims, type_dims, 4, num_types, hidden, layers, heads, src, tgt, dropout)
    def train(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Trains the model for one epoch using the provided DataLoader.

        Args:
            train_loader (DataLoader): The DataLoader for the training dataset.
            epoch (int): The current epoch number.

        Returns:
            float: The total training loss for the epoch.
        """
        # Set the model to training mode
        self.model.train()
        train_loss = 0.0  # Initialize training loss
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")


        # Iterate over batches in the training data
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()  # Zero gradients before backward pass
            
            # Handle different model types
            if self.model_type == 'Base':
                inputs, targets = batch['src'].to(self.device), batch['tgt'].to(self.device)
                src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                tgt_mask = self.create_mask(targets.shape[0], targets.shape[1])
                inputs = self.scaler.scale(inputs[:, :, :3], "src")
                targets = self.scaler.scale(targets[:, :, :2], "tgt")
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs[:, :, :2], targets, src_mask=src_mask.to(self.device), tgt_mask=tgt_mask.to(self.device))
            
            elif self.model_type == 'Transformer':
                inputs, targets = batch['src'].to(self.device), batch['tgt'].to(self.device)
                src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                tgt_mask = self.create_mask(targets.shape[0], targets.shape[1])
                scaled_inputs = self.scaler.scale(inputs[:, :, :3], "src")
                targets = self.scaler.scale(targets[:, :, :2], "tgt")
                inputs = torch.cat((scaled_inputs, inputs[:, :, 4:].to(self.device)), dim=2)
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs, targets, src_mask=src_mask.to(self.device), tgt_mask=tgt_mask.to(self.device))
            
            elif self.model_type in ['STAR', 'SAESTAR']:
                inputs, targets, distances, distance_types = batch['src'].to(self.device), batch['tgt'].to(self.device), batch['distance'].to(self.device).long(), batch['type'].to(self.device).long() 
                src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                scaled_inputs = self.scaler.scale(inputs[:, :, :3], "src")
                targets = self.scaler.scale(targets[:, :, :2], "tgt")
                distances = self.scaler.scale(distances, "dist")
                inputs = torch.cat((scaled_inputs, inputs[:, :, 4:].to(self.device)), dim=2)
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs.type(torch.float32), distances.type(torch.float32), distance_types, src_mask=src_mask.to(self.device))
            elif self.model_type in ['SEASTAR']:
                inputs, targets, distances, distance_types, dist_agents, dist_env = batch['src'].to(self.device), batch['tgt'].to(self.device), batch['distance'].to(self.device).long(), batch['type'].to(self.device).long(),  batch['dist_agents'].to(self.device).long(), batch['dist_env'].to(self.device).long()  
                src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                scaled_inputs = self.scaler.scale(inputs[:, :, :3], "src")
                targets = self.scaler.scale(targets[:, :, :2], "tgt")
                distances = self.scaler.scale(dist_agents, "agent_dist")
                dist_env = self.scaler.scale(dist_env, "env_dist")
                inputs = torch.cat((scaled_inputs, inputs[:, :, 4:].to(self.device)), dim=2)
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs.type(torch.float32), distances.type(torch.float32), distance_types, env_dist=dist_env, src_mask=src_mask.to(self.device))


            # Compute loss, backpropagate, and optimize
            print(outputs.shape)
            print(targets.shape)
            print(outputs)
            print(targets)
            loss = self.criterion(outputs.type(torch.float32), targets.type(torch.float32))
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Optimization step
            
            # Accumulate loss and update progress bar
            train_loss += loss.item()
            progress_bar.set_postfix({'Training Loss': train_loss / ((batch_idx + 1) * len(inputs))})


        return train_loss

    
    def eval(self, valid_loader: DataLoader) -> float:
        """
        Evaluates the model on a validation set.

        Args:
            valid_loader (DataLoader): The DataLoader for the validation dataset.

        Returns:
            float: The average validation loss for the validation set.
        """
        # Set the model to evaluation mode
        self.model.eval()
        valid_loss = 0.0

        # No gradient computation during validation
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                # Handle different model types and prepare inputs/targets accordingly
                if self.model_type == 'Base':
                    inputs, targets = batch['src'].to(self.device), batch['tgt'].to(self.device)
                    src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                    tgt_mask = self.create_mask(targets.shape[0], targets.shape[1])
                    inputs = self.scaler.scale(inputs[:, :, :3], "src")
                    targets = self.scaler.scale(targets[:, :, :2], "tgt")
                    outputs = self.model(inputs[:, :, :2], targets, src_mask=src_mask.to(self.device), tgt_mask=tgt_mask.to(self.device))
                
                elif self.model_type == 'Transformer':
                    inputs, targets = batch['src'].to(self.device), batch['tgt'].to(self.device)
                    src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                    tgt_mask = self.create_mask(targets.shape[0], targets.shape[1])
                    scaled_inputs = self.scaler.scale(inputs[:, :, :3], "src")
                    targets = self.scaler.scale(targets[:, :, :2], "tgt")
                    inputs = torch.cat((scaled_inputs, inputs[:, :, 4:].to(self.device)), dim=2)
                    outputs = self.model(inputs, targets, src_mask=src_mask.to(self.device), tgt_mask=tgt_mask.to(self.device))
                
                elif self.model_type in ['STAR', 'SAESTAR']:
                    inputs, targets, distances, distance_types = batch['src'].to(self.device), batch['tgt'].to(self.device), batch['distance'].to(self.device), batch['type'].to(self.device)
                    src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                    scaled_inputs = self.scaler.scale(inputs[:, :, :3], "src")
                    targets = self.scaler.scale(targets[:, :, :2], "tgt")
                    distances = self.scaler.scale(distances, "dist")
                    inputs = torch.cat((scaled_inputs, inputs[:, :, 4:].to(self.device)), dim=2)
                    outputs = self.model(inputs.type(torch.float32), distances.type(torch.float32), distance_types, src_mask=src_mask.to(self.device))
                elif self.model_type in ['SEASTAR']:
                    inputs, targets, distances, distance_types, dist_agents, dist_env = batch['src'].to(self.device), batch['tgt'].to(self.device), batch['distance'].to(self.device).long(), batch['type'].to(self.device).long(),  batch['dist_agents'].to(self.device).long(), batch['dist_env'].to(self.device).long()  
                    src_mask = self.create_mask(inputs.shape[0], inputs.shape[1])
                    scaled_inputs = self.scaler.scale(inputs[:, :, :3], "src")
                    targets = self.scaler.scale(targets[:, :, :2], "tgt")
                    distances = self.scaler.scale(dist_agents, "agent_dist")
                    dist_env = self.scaler.scale(dist_env, "env_dist")
                    inputs = torch.cat((scaled_inputs, inputs[:, :, 4:].to(self.device)), dim=2)
                    with autocast(dtype=torch.float16):
                        outputs = self.model(inputs.type(torch.float32), distances.type(torch.float32), distance_types, env_dist=dist_env, src_mask=src_mask.to(self.device))


                # Compute loss
                loss = self.criterion(outputs.type(torch.float32), targets.type(torch.float32))
                valid_loss += loss.item()

        # Calculate average validation loss
        valid_loss /= len(valid_loader.dataset)
        return valid_loss

        
    def train_model(self, train_loader, valid_loader) -> None:
        """
        Trains the model over multiple epochs, with early stopping and TensorBoard logging.

        Args:
            train_loader (DataLoader): The DataLoader for the training dataset.
            valid_loader (DataLoader): The DataLoader for the validation dataset.
        """
        self.model.to(self.device)  # Move model to GPU
        best_valid_loss = float('inf')
        early_stopping_counter = 0

        # Set up TensorBoard writers for logging training and validation losses
        log_dir = f'./data/Results/{self.model_type}/{self.location_name}/logs'
        weights_dir = f'./data/Weights/{self.model_type}/{self.location_name}/'
        train_writer = SummaryWriter(log_dir= log_dir + '/Train')
        valid_writer = SummaryWriter(log_dir= log_dir + f'/Valid')

        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self.train(train_loader, epoch)
            torch.save(self.model.state_dict(), weights_dir + 'current/checkpoint.pth')

            # Evaluate on the validation set
            valid_loss = self.eval(valid_loader)

            # Log losses
            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            train_writer.add_scalar('Loss', train_loss, epoch)
            valid_writer.add_scalar('Loss', valid_loss, epoch)

            # Save the model checkpoint if validation loss improves
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), weights_dir + '/best_chkp/checkpoint.pth')
                print("Checkpoint saved.")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Early stopping if validation loss does not improve for a number of epochs
            if early_stopping_counter >= self.early_stopping_patience:
                print(f"Validation loss hasn't decreased for {self.early_stopping_patience} epochs. Stopping training.")
                break

        # Close TensorBoard writers
        train_writer.close()
        valid_writer.close()


    def calculate_ADE(self, predicted_trajectories, true_trajectories):
        """
        Calculates the Average Displacement Error (ADE) between predicted and true trajectories.

        Args:
            predicted_trajectories (np.array): The predicted trajectory points.
            true_trajectories (np.array): The true trajectory points.

        Returns:
            float: The Average Displacement Error (ADE).
        """
        errors = np.linalg.norm(predicted_trajectories - true_trajectories, axis=-1)
        ADE = np.mean(errors, axis=-1)
        return ADE


    def calculate_FDE(self, predicted_trajectories, true_trajectories):
        """
        Calculates the Final Displacement Error (FDE) between predicted and true trajectories.

        Args:
            predicted_trajectories (np.array): The predicted trajectory points.
            true_trajectories (np.array): The true trajectory points.

        Returns:
            np.array: The Final Displacement Error (FDE).
        """
        errors = np.linalg.norm(predicted_trajectories[:, -1] - true_trajectories[:, -1], axis=-1)
        return errors


    def calculate_FDE(self, predicted_trajectories, true_trajectories):
        """
        Calculates the Final Displacement Error (FDE) between predicted and true trajectories.
        The FDE measures the Euclidean distance between the predicted and actual positions 
        at the final time step of a trajectory.

        Args:
            predicted_trajectories (np.array): The predicted trajectory points.
            true_trajectories (np.array): The true trajectory points.

        Returns:
            np.array: The Final Displacement Error (FDE) for each prediction.
        """
        # Calculate the Euclidean distance between predicted and true trajectories at the final timestep
        errors = np.linalg.norm(predicted_trajectories[:, -1] - true_trajectories[:, -1], axis=-1)
        return errors


    def validation_metrics(self, predicted_trajectories, true_trajectories):
        """
        Computes validation metrics, including the Average Displacement Error (ADE) and 
        Final Displacement Error (FDE), and identifies the minimum ADE and FDE.

        Args:
            predicted_trajectories (np.array): The predicted trajectory points.
            true_trajectories (np.array): The true trajectory points.

        Returns:
            tuple: A tuple containing the following metrics:
                - ADE (np.array): The Average Displacement Error for each prediction.
                - FDE (np.array): The Final Displacement Error for each prediction.
                - minADE (float): The minimum value of the ADE.
                - minFDE (float): The minimum value of the FDE.
        """
        # Calculate ADE and FDE using the helper functions
        ADE = self.calculate_ADE(predicted_trajectories, true_trajectories)
        FDE = self.calculate_FDE(predicted_trajectories, true_trajectories)
        
        # Find indices of the minimum values for ADE and FDE
        minADE_idx = np.argmin(ADE)
        minFDE_idx = np.argmin(FDE)
        
        # Get the minimum ADE and FDE values
        minADE = ADE[minADE_idx]
        minFDE = FDE[minFDE_idx]
        
        return ADE, FDE, minADE, minFDE


    def test_model(self, test_loader):
        """
        Evaluates the trained model on the test dataset, calculating prediction accuracy
        using ADE and FDE. Logs results to TensorBoard and saves the predictions and 
        true values to disk for later analysis.

        Args:
            test_loader (DataLoader): The DataLoader for the test dataset.
        """
        # Set model to evaluation mode and move it to GPU
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize lists to store results and predictions
        all_ADE = np.array([])
        all_FDE = np.array([])
        all_predictions = np.array([])
        all_actuals = np.array([])

        # Set up TensorBoard writer for logging
        writer = SummaryWriter()

        # Evaluate the model on the test dataset without gradient computation
        with torch.no_grad():
            src_tensors = np.array([])
            tgt_tensor = np.array([])
            pred_tensor = np.array([])

            # Iterate over the test data batches
            for batch_idx, batch in enumerate(test_loader):
                # Retrieve inputs and targets for each model type
                if self.model_type == 'Base':
                    src, tgt = batch['src'].to(self.device), batch['tgt'].to(self.device)
                    src_mask = self.create_mask(src.shape[0], src.shape[1])
                    tgt_mask = self.create_mask(tgt.shape[0], tgt.shape[1])
                    inputs = self.scaler.scale(src[:, :, :3], "src")
                    targets = self.scaler.scale(tgt[:, :, :2], "tgt")
                    outputs = self.model(inputs[:, :, :2], targets, src_mask=src_mask.to(self.device), tgt_mask=tgt_mask.to(self.device))

                elif self.model_type == 'Transformer':
                    src, tgt = batch['src'].to(self.device), batch['tgt'].to(self.device)
                    src_mask = self.create_mask(src.shape[0], src.shape[1])
                    tgt_mask = self.create_mask(tgt.shape[0], tgt.shape[1])
                    scaled_inputs = self.scaler.scale(src[:, :, :3], "src")
                    targets = self.scaler.scale(tgt[:, :, :2], "tgt")
                    inputs = torch.cat((scaled_inputs, src[:, :, 4:].to(self.device)), dim=2)
                    outputs = self.model(inputs, targets, src_mask=src_mask.to(self.device), tgt_mask=tgt_mask.to(self.device))

                elif self.model_type == 'STAR' or self.model_type == 'SAESTAR':
                    src, tgt, distances, distance_types = batch['src'].to(self.device), batch['tgt'].to(self.device), batch['distance'].to(self.device), batch['type'].to(self.device)
                    src_mask = self.create_mask(src.shape[0], src.shape[1])
                    scaled_inputs = self.scaler.scale(src[:, :, :3], "src")
                    targets = self.scaler.scale(tgt[:, :, :2], "tgt")
                    distances = self.scaler.scale(distances, "dist")
                    inputs = torch.cat((scaled_inputs, src[:, :, 4:].to(self.device)), dim=2)
                    outputs = self.model(inputs.type(torch.float32), distances.type(torch.float32), distance_types, src_mask=src_mask.to(self.device))
                elif self.model_type == 'SEASTAR':
                    src, tgt, distances, distance_types, dist_agents, dist_env = batch['src'].to(self.device), batch['tgt'].to(self.device), batch['distance'].to(self.device).long(), batch['type'].to(self.device).long(),  batch['dist_agents'].to(self.device).long(), batch['dist_env'].to(self.device).long()
                    src_mask = self.create_mask(src.shape[0], src.shape[1])
                    scaled_inputs = self.scaler.scale(src[:, :, :3], "src")
                    targets = self.scaler.scale(tgt[:, :, :2], "tgt")
                    distances = self.scaler.scale(dist_agents, "agents_dist")
                    dist_env = self.scaler.scale(dist_env, "env_dist")
                    inputs = torch.cat((scaled_inputs, src[:, :, 4:].to(self.device)), dim=2)
                    outputs = self.model(inputs.type(torch.float32), distances.type(torch.float32), distance_types, env_dist=dist_env, src_mask=src_mask.to(self.device))

    
                # Unscale the model outputs to original values
                new_outputs = self.scaler.unscale(outputs, "tgt", tgt[:, :, :2].shape)
                
                # Calculate validation metrics (ADE, FDE)
                ADE, FDE, minADE, minFDE = self.validation_metrics(new_outputs.cpu().numpy(), tgt[:, :, :2].cpu().numpy())
                
                if pred_tensor.shape[0] == 0:
                    # Collect predictions, targets, and sources for later processing
                    pred_tensor =  new_outputs.cpu().numpy()
                    tgt_tensor = tgt.cpu().numpy()
                    src_tensors = src.cpu().numpy()
                                        
                    all_ADE = ADE
                    all_FDE = FDE

                    # Save predictions and true values
                    all_predictions = new_outputs.cpu().numpy()
                    all_actuals = targets[:, :, :2].cpu().numpy()
                
                # Collect predictions, targets, and sources for later processing
                pred_tensor = np.append(pred_tensor, new_outputs.cpu().numpy(), axis=0)
                tgt_tensor = np.append(tgt_tensor, tgt.cpu().numpy(), axis=0)
                src_tensors = np.append(src_tensors, src.cpu().numpy(), axis=0)
                
                all_ADE = np.append(all_ADE, ADE, axis=0)
                all_FDE = np.append(all_FDE, FDE, axis=0)

                # Save predictions and true values
                all_predictions = np.append(all_predictions, new_outputs.cpu().numpy(), axis=0)
                all_actuals = np.append(all_actuals, targets[:, :, :2].cpu().numpy(), axis=0)
                
                # Log results to TensorBoard
                writer.add_scalar('ADE/mean', ADE.mean())
                writer.add_scalar('FDE/mean', FDE.mean())
                writer.add_scalar('ADE/Min_mean', minADE.mean())
                writer.add_scalar('FDE/Min_mean', minFDE.mean())

            # Close the TensorBoard writer
            writer.close()
            
            # Save the results to CSV
            file_path = f"./data/Results/{self.model_type}/{self.location_name}/src_pred_tgt_results.csv"
            with open(file_path, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Source", "Prediction", "Target"])

                for i in range(src_tensors.shape[0]):
                    for j in range(src_tensors.shape[1]):
                        src_values = src_tensors[i, j, :]
                        prediction_values = pred_tensor[i, j, :]
                        target_values = tgt_tensor[i, j, :]
                        csv_writer.writerow([src_values, prediction_values, target_values])

        # Save tensors for later analysis
        src_tensors = torch.tensor(src_tensors)
        tgt_tensor = torch.tensor(tgt_tensor)
        pred_tensor = torch.tensor(pred_tensor)

        torch.save(src_tensors, f"./data/Results/{self.model_type}/{self.location_name}/src.pt")
        torch.save(tgt_tensor, f"./data/Results/{self.model_type}/{self.location_name}/tgt.pt")
        torch.save(pred_tensor, f"./data/Results/{self.model_type}/{self.location_name}/pred.pt")

        # Calculate overall metrics across all batches
        mean_ADE = np.mean(all_ADE)
        mean_FDE = np.mean(all_FDE)

        # Find the minimum ADE and FDE values
        min_ADE_idx = np.argmin(all_ADE)
        min_FDE_idx = np.argmin(all_FDE)
        min_ADE = all_ADE[min_ADE_idx]
        min_FDE = all_FDE[min_FDE_idx]

        # Save the overall metrics to a file
        save_file = f"./data/Results/{self.model_type}/{self.location_name}/metrics.txt"
        with open(save_file, "w") as file:
            file.write(f"Mean ADE: {mean_ADE}\n")
            file.write(f"Mean FDE: {mean_FDE}\n")
            file.write(f"Min ADE: {min_ADE}\n")
            file.write(f"Min FDE: {min_FDE}\n")

        print("Validation results saved to", save_file)