import torch
import torch.nn as nn

from classes.Embeddings.saestar_embedding import SAESTAREmbedding
from classes.Models.mlp_decoder import MLPDecoder

  
class SAESTAR(torch.nn.Module):
    """
    SAESTAR Model: A spatial-temporal transformer-based model for sequence prediction tasks.

    This model integrates spatial and temporal encoding layers with a multi-layer perceptron (MLP) decoder. 
    It fuses spatial and temporal features to produce predictions over a target sequence length.

    Args:
        src_dims (list[int]): Dimensions of source embeddings.
        dist_dims (list[int]): Dimensions of distance embeddings.
        type_dims (list[int]): Dimensions of type embeddings.
        num_types (int): Number of unique types in the type embedding.
        num_types_dist (int): Number of unique types in the distance embedding.
        hidden (int): Hidden layer size for the decoders (default=256).
        num_layers (int): Number of layers in the transformer encoders (default=16).
        num_heads (int): Number of attention heads in the transformer layers (default=8).
        src_len (int): Length of the source sequence.
        tgt_len (int): Length of the target sequence.
        dropout (float): Dropout probability for regularization (default=0.1).

    Methods:
        forward(src, distance, type, src_mask=None, dist_key_padding_mask=None):
            Executes the forward pass of the model, embedding the input data,
            encoding spatial and temporal features, and decoding the output.
    """
    def __init__(self, src_dims, dist_dims, type_dims, num_types, num_types_dist, hidden=256, num_layers=16, num_heads=8, src_len=10, tgt_len=40, dropout=0.1):
        """
        Initialize the SAESTAR model.

        Args:
            src_dims (list[int]): Dimensions for the source embeddings.
            dist_dims (list[int]): Dimensions for the distance embeddings.
            type_dims (list[int]): Dimensions for the type embeddings.
            num_types (int): Total number of unique types for type embeddings.
            num_types_dist (int): Total number of unique types for distance embeddings.
            hidden (int): Hidden size for the decoders.
            num_layers (int): Number of layers for each transformer encoder.
            num_heads (int): Number of attention heads in each transformer layer.
            src_len (int): Length of the source sequence.
            tgt_len (int): Length of the target sequence.
            dropout (float): Dropout probability for regularization.
        """
        super(SAESTAR, self).__init__()

        # Model architecture parameters.
        self.embedding_size = sum(src_dims)  # Total size of the input embeddings.
        self.output_size = 2  # Output dimensions (e.g., x and y coordinates).
        self.dropout_prob = dropout  # Dropout probability.
        self.src_len = src_len  # Source sequence length.
        self.tgt_len = tgt_len  # Target sequence length.
        self.hidden_size = hidden  # Hidden layer size.

        # Embedding layer for source, distance, and type inputs.
        self.embedding_layer = SAESTAREmbedding(src_dims, dist_dims, type_dims, num_types, num_types_dist, src_len).cuda()

        # Transformer encoder layers for spatial and temporal features.
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads, batch_first=True).cuda()
        self.spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads, batch_first=True).cuda()

        # Stacked transformer encoders for spatial and temporal processing.
        self.spatial_encoder_1 = nn.TransformerEncoder(self.spatial_encoder_layer, num_layers).cuda()
        self.spatial_encoder_2 = nn.TransformerEncoder(self.spatial_encoder_layer, num_layers).cuda()
        self.temporal_encoder_1 = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers).cuda()
        self.temporal_encoder_2 = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers).cuda()

        # Decoders for spatial-temporal feature projection.
        self.decoder1 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()
        self.decoder2 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()
        self.decoder3 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()
        self.decoder4 = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()

        # Fusion layer for combining spatial and temporal features.
        self.fusion_layer = nn.Linear(self.embedding_size * 2, self.embedding_size).cuda()

        # Regularization and activation layers.
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu1 = nn.ReLU()

    def forward(self, src, distance, type, src_mask=None, dist_key_padding_mask=None):
        """
        Forward pass through the SAESTAR model.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_len, src_dims).
            distance (torch.Tensor): Distance tensor of shape (batch_size, src_len, dist_dims).
            type (torch.Tensor): Type tensor of shape (batch_size, src_len, type_dims).
            src_mask (torch.Tensor, optional): Mask for the source sequence (default=None).
            dist_key_padding_mask (torch.Tensor, optional): Mask for the distance tensor (default=None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, output_size).
        """
        # Embed the source sequence for temporal encoding.
        src_temporal_embedded = self.embedding_layer(src, is_src=True).cuda()
        src_temporal_embedded = self.dropout(src_temporal_embedded).cuda()  # Apply dropout.
        src_temporal_embedded = self.relu(src_temporal_embedded).cuda()  # Apply ReLU activation.

        # Embed the distance sequence for spatial encoding.
        src_dist_embedding = self.embedding_layer(distance, is_src=False, src_tensor=src, type_tensor=type).cuda()
        src_dist_embedded = self.dropout1(src_dist_embedding).cuda()  # Apply dropout.
        src_dist_embedded = self.relu1(src_dist_embedded).cuda()  # Apply ReLU activation.

        # Process spatial and temporal embeddings through respective encoders.
        spatial_input_embedded = self.spatial_encoder_1(src_dist_embedded, mask=src_mask).cuda()
        temporal_input_embedded = self.temporal_encoder_1(src_temporal_embedded, mask=src_mask).cuda()

        # Fuse spatial and temporal features.
        fusion_feat = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=2).cuda()
        fusion_feat = self.fusion_layer(fusion_feat).cuda()  # Combine features using the fusion layer.

        # Process the fused features through secondary encoders.
        spatial_output = self.spatial_encoder_2(fusion_feat).cuda()
        temporal_output = self.temporal_encoder_2(spatial_output).cuda()

        # Reshape the temporal output for decoding.
        temporal_output = temporal_output.reshape(
            temporal_output.shape[0] * temporal_output.shape[1], temporal_output.shape[2]
        )

        # Decode the temporal output through multiple decoders.
        output1 = self.decoder1(temporal_output).cuda()
        output2 = self.decoder2(temporal_output).cuda()
        output3 = self.decoder3(temporal_output).cuda()
        output4 = self.decoder4(temporal_output).cuda()

        # Concatenate the outputs from all decoders.
        output = torch.cat([output1, output2, output3, output4], dim=1)
        output = output.reshape(src.shape[0], self.tgt_len, self.output_size).cuda()

        return output.cuda()