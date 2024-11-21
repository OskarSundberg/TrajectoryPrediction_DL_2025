import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) decoder for transforming input features
    into desired output representations. The architecture consists of two 
    fully connected layers with ReLU activation and dropout for regularization.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of units in the hidden layer.
        output_size (int): The number of output features.
        dropout (float): Dropout probability for regularization.

    Methods:
        forward(x): Forward pass through the network. Processes the input 
                    tensor `x` through dropout, activation, and fully connected layers.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout):
        """
        Initialize the MLPDecoder model.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
            dropout (float): Dropout probability for regularization.
        """
        super(MLPDecoder, self).__init__()

        # First fully connected layer: input_size -> hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size).cuda()
        
        # Second fully connected layer: hidden_size -> output_size
        self.fc2 = nn.Linear(hidden_size, output_size).cuda()
        
        # Dropout layer for regularization after input
        self.dropout = nn.Dropout(p=dropout)
        
        # ReLU activation after dropout and first FC layer
        self.relu = nn.ReLU()
        
        # Additional dropout and ReLU activation after the first FC layer
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass through the MLP decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Apply dropout to the input
        x = self.dropout(x).cuda()
        
        # Apply ReLU activation
        x = self.relu(x).cuda()
        
        # First fully connected layer
        x = self.fc1(x).cuda()
        
        # Additional dropout after the first FC layer
        x = self.dropout1(x).cuda()
        
        # Additional ReLU activation
        x = self.relu1(x).cuda()
        
        # Second fully connected layer (output layer)
        x = self.fc2(x).cuda()
        
        return x