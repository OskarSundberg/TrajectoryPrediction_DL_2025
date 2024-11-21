import torch
import numpy as np
import torch.nn as nn

from classes.Embeddings.star_embedding import STAREmbedding
from classes.Models.mlp_decoder import MLPDecoder


class STAR(torch.nn.Module):
    """
    STAR Model: A spatial-temporal attention-based model for sequence prediction tasks.

    This model incorporates spatial and temporal transformer encoders, 
    along with MLP decoders, to predict sequences over a target length.

    Args:
        embedding_dims (np.array): Dimensions of the input embeddings.
        dist_dims (np.array): Dimensions of the distance embeddings.
        type_dims (np.array): Dimensions of the type embeddings.
        num_types (int): Number of unique types in the type embeddings.
        hidden (int): Hidden layer size for the decoders (default=256).
        num_layers (int): Number of transformer encoder layers (default=16).
        num_heads (int): Number of attention heads in the transformer layers (default=8).
        src_len (int): Length of the source sequence.
        tgt_len (int): Length of the target sequence.
        dropout (float): Dropout probability for regularization (default=0.1).

    Methods:
        forward(src, distance, type, src_mask=None, dist_key_padding_mask=None):
            Performs a forward pass through the model, processing embeddings, encoding 
            spatial and temporal features, and decoding predictions.
    """
    def __init__(
        self, 
        embedding_dims: np.array, 
        dist_dims: np.array, 
        type_dims: np.array, 
        num_types: int, 
        hidden: int = 256, 
        num_layers: int = 16, 
        num_heads: int = 8, 
        src_len: int = 10, 
        tgt_len: int = 40, 
        dropout: float = 0.1
    ):
        """
        Initialize the STAR model.

        Args:
            embedding_dims (np.array): Dimensions for input embeddings.
            dist_dims (np.array): Dimensions for distance embeddings.
            type_dims (np.array): Dimensions for type embeddings.
            num_types (int): Number of unique types for type embeddings.
            hidden (int): Hidden layer size for decoders.
            num_layers (int): Number of layers in transformer encoders.
            num_heads (int): Number of attention heads in transformers.
            src_len (int): Length of the source sequence.
            tgt_len (int): Length of the target sequence.
            dropout (float): Dropout probability for regularization.
        """
        super(STAR, self).__init__()

        # Define key architectural parameters
        self.embedding_size = sum(embedding_dims)  # Total embedding size after concatenation.
        self.output_size = 2  # Output size (e.g., x and y coordinates).
        self.dropout_prob = dropout  # Dropout rate.
        self.src_len = src_len  # Source sequence length.
        self.tgt_len = tgt_len  # Target sequence length.
        self.hidden_size = hidden  # Hidden size for decoders.

        # Embedding layer for input, distance, and type data.
        self.embedding_layer = STAREmbedding(embedding_dims, dist_dims, type_dims, num_types, src_len).cuda()

        # Transformer encoder layers for temporal and spatial data.
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads, batch_first=True).cuda()
        self.spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads, batch_first=True).cuda()

        # Stacked transformer encoders for spatial and temporal processing.
        self.spatial_encoder_1 = nn.TransformerEncoder(self.spatial_encoder_layer, num_layers).cuda()
        self.spatial_encoder_2 = nn.TransformerEncoder(self.spatial_encoder_layer, num_layers).cuda()
        self.temporal_encoder_1 = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers).cuda()
        self.temporal_encoder_2 = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers).cuda()

        # Decoders for transforming encoded features to output.
        self.decoder1 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()
        self.decoder2 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()
        self.decoder3 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()
        self.decoder4 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()

        # Fusion layer to combine spatial and temporal features.
        self.fusion_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).cuda()

        # Regularization and activation layers.
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu1 = nn.ReLU()

    def forward(self, src, distance, type, src_mask=None, dist_key_padding_mask=None):
        """
        Forward pass of the STAR model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_len, embedding_dims).
            distance (torch.Tensor): Distance input tensor of shape (batch_size, src_len, dist_dims).
            type (torch.Tensor): Type input tensor of shape (batch_size, src_len, type_dims).
            src_mask (torch.Tensor, optional): Mask for source input (default=None).
            dist_key_padding_mask (torch.Tensor, optional): Mask for distance input (default=None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, output_size).
        """
        # Temporal embedding for source data.
        src_temporal_embedded = self.embedding_layer(src, is_src=True).cuda()
        src_temporal_embedded = self.dropout(src_temporal_embedded).cuda()  # Apply dropout.
        src_temporal_embedded = self.relu(src_temporal_embedded).cuda()  # Apply ReLU activation.

        # Spatial embedding for distance data.
        src_dist_embedding = self.embedding_layer(distance, is_src=False, src_tensor=src, type_tensor=type).cuda()
        src_dist_embedded = self.dropout1(src_dist_embedding).cuda()  # Apply dropout.
        src_dist_embedded = self.relu1(src_dist_embedded).cuda()  # Apply ReLU activation.

        # Process embeddings through spatial and temporal encoders.
        spatial_input_embedded = self.spatial_encoder_1(src_dist_embedded, mask=src_mask).cuda()
        temporal_input_embedded = self.temporal_encoder_1(src_temporal_embedded, mask=src_mask).cuda()

        # Fuse spatial and temporal features.
        fusion_feat = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=2).cuda()
        fusion_feat = self.fusion_layer(fusion_feat).cuda()  # Apply linear fusion layer.

        # Further processing through secondary encoders.
        spatial_output = self.spatial_encoder_2(fusion_feat).cuda()
        temporal_output = self.temporal_encoder_2(spatial_output).cuda()

        # Reshape output for decoding.
        temporal_output = temporal_output.reshape(
            temporal_output.shape[0] * temporal_output.shape[1], temporal_output.shape[2]
        )

        # Decode spatial-temporal features using decoders.
        output1 = self.decoder1(temporal_output).cuda()
        output2 = self.decoder2(temporal_output).cuda()
        output3 = self.decoder3(temporal_output).cuda()
        output4 = self.decoder4(temporal_output).cuda()

        # Concatenate decoder outputs and reshape.
        output = torch.cat([output1, output2, output3, output4], dim=1)
        output = output.reshape(src.shape[0], self.tgt_len, self.output_size).cuda()

        return output.cuda()
