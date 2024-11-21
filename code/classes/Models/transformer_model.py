import torch
import torch.nn as nn

from classes.Embeddings.transformer_embedding import TransformerEmbedding
from classes.Models.mlp_decoder import MLPDecoder


class Transformer(nn.Module):
    """
    Transformer Model: Implements a transformer-based architecture for sequence-to-sequence 
    prediction tasks, particularly for spatial-temporal data processing.

    This model uses an encoder-decoder structure with transformer layers and MLP decoders 
    for output projection.

    Args:
        src_dim (list): Dimensions of the source input features.
        tgt_dim (list): Dimensions of the target output features.
        num_agents (int): Number of agents in the dataset.
        num_layers (int): Number of transformer layers in the encoder and decoder.
        num_heads (int): Number of attention heads in the transformer layers.
        hidden (int): Hidden layer size for the MLP decoder.
        src_len (int): Length of the source sequence (default=10).
        tgt_len (int): Length of the target sequence (default=40).
        dropout (float): Dropout probability for regularization (default=0.1).

    Methods:
        forward(src, tgt=None, src_mask=None, tgt_mask=None, training=True):
            Performs a forward pass through the model.
    """
    def __init__(self, src_dim, tgt_dim, num_agents, num_layers, num_heads, hidden, src_len=10, tgt_len=40, dropout=0.1):
        """
        Initialize the Transformer model.

        Args:
            src_dim (list): Source input feature dimensions.
            tgt_dim (list): Target output feature dimensions.
            num_agents (int): Number of agents in the dataset.
            num_layers (int): Number of encoder and decoder layers.
            num_heads (int): Number of attention heads in transformer layers.
            hidden (int): Hidden size for the MLP decoder.
            src_len (int): Length of the source sequence.
            tgt_len (int): Length of the target sequence.
            dropout (float): Dropout probability.
        """
        super(Transformer, self).__init__()

        # Key parameters for architecture
        self.src_len = src_len  # Source sequence length
        self.tgt_len = tgt_len  # Target sequence length
        self.size = sum(src_dim)  # Total input embedding size
        self.hidden = hidden  # Hidden size for MLP decoder
        self.output_size = len(tgt_dim)  # Output size (e.g., x and y coordinates)

        # Embedding layer for source and target data
        self.embedding_layer = TransformerEmbedding(
            src_dims=src_dim,
            tgt_dims=tgt_dim,
            num_agents=num_agents,
            sequence_length_src=src_len,
            sequence_length_tgt=tgt_len
        ).cuda()

        # Transformer encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.size, nhead=num_heads, batch_first=True).cuda()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.size, nhead=num_heads, batch_first=True).cuda()
        

        # Full encoder and decoder stacks
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).cuda()
        
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers).cuda()

        # Linear projection layer for output
        self.mlpdecoder = MLPDecoder(self.size, self.hidden, self.output_size, dropout=dropout).cuda()

        # Regularization and activation layers
        self.dropout = nn.Dropout(p=dropout).cuda()
        self.relu = nn.ReLU().cuda()

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, training=True):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_len, src_dim).
            tgt (torch.Tensor, optional): Target input tensor of shape (batch_size, tgt_len, tgt_dim).
                Required if training is True.
            src_mask (torch.Tensor, optional): Attention mask for source input (default=None).
            tgt_mask (torch.Tensor, optional): Attention mask for target input (default=None).
            training (bool): Indicates whether the model is in training mode (default=True).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, output_dim).
        """
        # Embed the source sequence.
        src_embedded = self.embedding_layer(src, is_src=True).cuda()
        src_embedded = self.dropout(src_embedded).cuda()  # Apply dropout.
        src_embedded = self.relu(src_embedded).cuda()  # Apply ReLU activation.

        # Pass the source embeddings through the transformer encoder.
        memory = self.encoder(src_embedded, mask=src_mask).cuda()

        if training:
            # Embed the target sequence for training.
            tgt_embedded = self.embedding_layer(tgt, is_src=False).cuda()
            
            # Decode using the target embeddings and encoded memory.
            output = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask).cuda()
        else:
            # For inference, create a start token tensor for decoding.
            start_token = torch.zeros(1, self.tgt_len, src_embedded.shape[2]).cuda()
            start_token = start_token.repeat(src.shape[0], 1, 1)  # Repeat for the batch size.

            # Perform decoding using the start token and encoded memory.
            output = self.decoder(start_token, memory, tgt_mask=tgt_mask).cuda()

        # Pass the output through the MLP decoder.
        output = self.mlpdecoder(output).cuda()

        # Reshape to (batch_size, tgt_len, output_size) for final output.
        output = output.view(src.shape[0], self.tgt_len, self.output_size).cuda()

        return output