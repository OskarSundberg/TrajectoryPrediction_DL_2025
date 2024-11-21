import torch
import torch.nn as nn


class TransformerBaseEmbedding(nn.Module):
    """
    A PyTorch module for creating input embeddings for a transformer model.
    
    This class embeds input sequences using linear projections for specific features and 
    adds positional encoding to the embeddings. It supports separate embeddings for 
    source (`src`) and target (`tgt`) sequences.

    Attributes:
        embedding_dims (list[int]): List of embedding dimensions for each feature.
        size (int): Total embedding size (sum of `embedding_dims`).
        linear_layers_tgt (nn.ModuleList): Linear layers for projecting the first two features of `tgt` inputs.
        linear_layers_src (nn.ModuleList): Linear layers for projecting the first two features of `src` inputs.
        positional_encoding_src (torch.Tensor): Positional encoding tensor for `src` inputs.
        positional_encoding_tgt (torch.Tensor): Positional encoding tensor for `tgt` inputs.
    """
    
    def __init__(self, embedding_dims, sequence_length_src=10, sequence_length_tgt=40):
        """
        Initializes the embedding class with specified dimensions and sequence lengths.

        Args:
            embedding_dims (list[int]): Embedding dimensions for the features.
            sequence_length_src (int, optional): Length of the source sequence. Defaults to 10.
            sequence_length_tgt (int, optional): Length of the target sequence. Defaults to 40.
        """
        super(TransformerBaseEmbedding, self).__init__()
        self.embedding_dims = embedding_dims
        self.size = sum(embedding_dims)

        # Linear layers for projecting the first two features of `tgt`
        self.linear_layers_tgt = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in embedding_dims[:2]
        ])

        # Linear layers for projecting the first two features of `src`
        self.linear_layers_src = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in embedding_dims[:2]
        ])

        # Positional encodings for `src` and `tgt` sequences
        self.positional_encoding_src = self._generate_positional_encoding(self.size, sequence_length_src).cuda()
        self.positional_encoding_tgt = self._generate_positional_encoding(self.size, sequence_length_tgt).cuda()

    def _generate_positional_encoding(self, embedding_size, sequence_length):
        """
        Generates positional encoding for a sequence of a given length and embedding size.

        Args:
            embedding_size (int): Dimensionality of the embedding space.
            sequence_length (int): Length of the sequence.

        Returns:
            torch.Tensor: A tensor containing positional encodings with shape (1, sequence_length, embedding_size).
        """
        positional_encoding = torch.zeros(sequence_length, embedding_size)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))

        # Apply sine and cosine to alternate dimensions
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, input_tensor, is_src=True):
        """
        Forward pass for embedding input sequences with optional source or target embeddings.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).
            is_src (bool, optional): If `True`, processes as source (`src`) sequence; otherwise as target (`tgt`).
                                     Defaults to `True`.

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_length, total_embedding_size).
        """
        batch_size, seq_length, num_features = input_tensor.size()

        # Initialize an empty tensor for the embedded output
        embedded_tensor = torch.zeros(batch_size, seq_length, self.size).cuda()

        # Process source (`src`) sequences
        if is_src:
            start_index = 0
            for i, linear_layer in enumerate(self.linear_layers_src):
                # Apply the linear layer to the i-th feature
                linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1)).cuda()

                # Determine the embedding dimension for the current feature
                embedding_dim = self.embedding_dims[i]
                end_index = start_index + embedding_dim

                # Place the linearly projected feature into the correct slice of the output tensor
                embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).cuda()
                start_index = end_index

        # Process target (`tgt`) sequences
        else:
            start_index = 0
            for i, linear_layer in enumerate(self.linear_layers_tgt):
                # Apply the linear layer to the i-th feature
                linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1)).cuda()

                # Determine the embedding dimension for the current feature
                embedding_dim = self.embedding_dims[i]
                end_index = start_index + embedding_dim

                # Place the linearly projected feature into the correct slice of the output tensor
                embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).cuda()
                start_index = end_index

        # Add positional encoding to the embeddings
        positional_encoding = self.positional_encoding_src if is_src else self.positional_encoding_tgt
        positional_encoding = positional_encoding.cuda()
        embedded_tensor = embedded_tensor + positional_encoding

        # Return the final embedded tensor
        return embedded_tensor.view(batch_size, seq_length, -1).cuda()
