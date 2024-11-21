import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    """
    A PyTorch module for embedding input sequences with source (`src`) and target (`tgt`) features 
    using linear projections for numerical features and embedding layers for categorical features. 
    Additionally, positional encodings are added to the embeddings.
    
    Attributes:
        size (int): Total size of embeddings (sum of src_dims).
        src_dims (list[int]): List of embedding dimensions for the source sequence.
        tgt_dims (list[int]): List of embedding dimensions for the target sequence.
        src_len (int): Length of the source sequence.
        tgt_len (int): Length of the target sequence.
        linear_layers_src (nn.ModuleList): List of linear layers for the source sequence features.
        linear_layers_tgt (nn.ModuleList): List of linear layers for the target sequence features.
        embedding_layer_src (nn.Embedding): Embedding layer for the last categorical feature of the source sequence.
        positional_encoding_src (torch.Tensor): Positional encoding for the source sequence.
        positional_encoding_tgt (torch.Tensor): Positional encoding for the target sequence.
    """
    
    def __init__(self, src_dims, tgt_dims, num_agents, sequence_length_src=10, sequence_length_tgt=40):
        """
        Initializes the embedding class with specified dimensions and sequence lengths for source 
        and target sequences.
        
        Args:
            src_dims (list[int]): List of embedding dimensions for the features in the source sequence.
            tgt_dims (list[int]): List of embedding dimensions for the features in the target sequence.
            num_agents (int): Number of distinct categories for the last feature in the source sequence.
            sequence_length_src (int, optional): Length of the source sequence. Defaults to 10.
            sequence_length_tgt (int, optional): Length of the target sequence. Defaults to 40.
        """
        super(TransformerEmbedding, self).__init__()
        
        self.size = sum(src_dims)  # Total embedding size (sum of src_dims)
        self.src_dims = src_dims
        self.tgt_dims = tgt_dims
        self.src_len = sequence_length_src
        self.tgt_len = sequence_length_tgt 
        
        # Linear layers for projecting the first 3 features of the source sequence
        self.linear_layers_src = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in src_dims[:3]  # Use linear layers for the first 3 features of src
        ])
        
        # Linear layers for projecting the first 2 features of the target sequence
        self.linear_layers_tgt = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in tgt_dims[:2]  # Use linear layers for the first 2 features of tgt
        ])

        # Embedding layer for the last feature of the source sequence (categorical)
        self.embedding_layer_src = nn.Embedding(num_agents, src_dims[-1]).cuda()

        # Generate positional encodings for both source and target sequences
        self.positional_encoding_src = self._generate_positional_encoding(self.size, sequence_length_src).cuda()
        self.positional_encoding_tgt = self._generate_positional_encoding(self.size, sequence_length_tgt).cuda()

    def _generate_positional_encoding(self, embedding_size, sequence_length):
        """
        Generates a sinusoidal positional encoding for a given sequence length and embedding size.
        
        Args:
            embedding_size (int): Dimensionality of the embedding space.
            sequence_length (int): Length of the sequence for which to generate the encoding.

        Returns:
            torch.Tensor: Positional encoding tensor with shape (1, sequence_length, embedding_size).
        """
        positional_encoding = torch.zeros(sequence_length, embedding_size)  # Initialize the encoding tensor
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)  # Position tensor
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))  # Scaling factor
        
        # Apply sine and cosine functions to alternating dimensions of the encoding
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        return positional_encoding.unsqueeze(0)  # Add batch dimension and return

    def forward(self, input_tensor, is_src=True):
        """
        Forward pass for embedding input sequences, with separate handling for source and target sequences.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).
            is_src (bool, optional): If `True`, processes the source sequence; otherwise processes the target sequence. Defaults to `True`.

        Returns:
            torch.Tensor: The final embedded tensor of shape (batch_size, seq_length, total_embedding_size).
        """
        batch_size, seq_length, num_features = input_tensor.size()
        
        # Select the embedding dimensions based on whether the sequence is source or target
        embedding_dims = self.src_dims if is_src else self.tgt_dims

        # Initialize an empty tensor for the embedded output
        embedded_tensor = torch.zeros(batch_size, seq_length, self.size).cuda()

        # Handle source sequence embedding
        if is_src:
            start_index = 0
            for i, linear_layer in enumerate(self.linear_layers_src):
                # Linearly project each feature of the source sequence
                linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1)).cuda()

                # Get the embedding dimension for the current feature
                embedding_dim = embedding_dims[i]
                end_index = start_index + embedding_dim

                # Place the linearly projected feature at the correct position in the embedded tensor
                embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).cuda()
                start_index = end_index

            # Embed the last feature of the source sequence (categorical)
            embedded_tensor[:, :, start_index:] = self.embedding_layer_src(input_tensor[:, :, 3].long()).cuda()

        # Handle target sequence embedding
        else:
            start_index = 0
            for i, linear_layer in enumerate(self.linear_layers_tgt):
                # Linearly project each feature of the target sequence
                linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1)).cuda()

                # Get the embedding dimension for the current feature
                embedding_dim = embedding_dims[i]
                end_index = start_index + embedding_dim

                # Place the linearly projected feature at the correct position in the embedded tensor
                embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).cuda()
                start_index = end_index

        # Add positional encoding to the embeddings
        positional_encoding = self.positional_encoding_src if is_src else self.positional_encoding_tgt
        embedded_tensor = embedded_tensor + positional_encoding.cuda()

        # Reshape and return the final embedded tensor
        embedded_tensor = embedded_tensor.view(batch_size, seq_length, -1).cuda()
        return embedded_tensor.cuda()