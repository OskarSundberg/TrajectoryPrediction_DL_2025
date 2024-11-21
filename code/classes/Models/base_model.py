import torch
import torch.nn as nn

from classes.Embeddings.base_embedding import TransformerBaseEmbedding
from classes.Models.mlp_decoder import MLPDecoder


class TransformerBase(nn.Module):
    """
    A Transformer-based neural network model for sequence-to-sequence tasks. 
    This class integrates embedding layers, transformer encoder-decoder blocks, 
    and a projection layer for output.

    Args:
        embedding_dims (list[int]): List of embedding dimensions for each feature.
        src_len (int): Length of the source sequence.
        tgt_len (int): Length of the target sequence.
        hidden (int): Hidden layer size for the MLP decoder.
        num_layers (int): Number of layers in the encoder and decoder.
        num_heads (int): Number of attention heads in the transformer layers.
        dropout (float): Dropout probability for regularization (default is 0.1).

    Methods:
        forward(src, tgt=None, src_mask=None, tgt_mask=None, training=True):
            Executes the forward pass of the model. If `training` is True, 
            the target tensor is used for decoding; otherwise, decoding is done 
            without a target.
    """
    def __init__(self, embedding_dims, src_len, tgt_len, hidden, num_layers, num_heads, dropout=0.1):
        """
        Initialize the TransformerBase model.

        Args:
            embedding_dims (list[int]): Embedding dimensions for each input feature.
            src_len (int): Source sequence length.
            tgt_len (int): Target sequence length.
            hidden (int): Hidden size for the MLP decoder.
            num_layers (int): Number of transformer encoder/decoder layers.
            num_heads (int): Number of attention heads in each transformer layer.
            dropout (float): Dropout rate for regularization.
        """
        super(TransformerBase, self).__init__()
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.embedding_size = sum(embedding_dims)  # Total embedding size.
        self.hidden_size = hidden
        self.output_size = len(embedding_dims)  # Number of output dimensions (e.g., x and y coordinates).

        # Embedding layer for source and target sequences.
        self.embedding_layer = TransformerBaseEmbedding(
            embedding_dims, sequence_length_src=src_len, sequence_length_tgt=tgt_len
        ).cuda()

        # Transformer encoder and decoder layers.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads, batch_first=True).cuda()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_size, nhead=num_heads, batch_first=True).cuda()

        # Full encoder and decoder using the above layers.
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).cuda()
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers).cuda()

        # Linear projection layer for output mapping.
        self.linear_output = nn.Linear(self.embedding_size, self.output_size).cuda()

        # MLP Decoder for final feature transformation.
        self.mlpdecoder = MLPDecoder(self.embedding_size, self.hidden_size, self.output_size, dropout=dropout).cuda()

        # Regularization layers.
        self.dropout = nn.Dropout(p=dropout).cuda()
        self.relu = nn.ReLU().cuda()

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, training=True):
        """
        Forward pass through the TransformerBase model.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_len, embedding_size).
            tgt (torch.Tensor, optional): Target sequence tensor of shape (batch_size, tgt_len, embedding_size). Required if training=True.
            src_mask (torch.Tensor, optional): Mask for the source sequence (default is None).
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (default is None).
            training (bool, optional): If True, the model uses the target tensor for decoding (default is True).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, output_size).
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
