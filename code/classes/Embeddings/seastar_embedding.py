import torch
import torch.nn as nn

class SEASTAREmbedding(nn.Module):
    """
    A PyTorch module for embedding input sequences with source and distance-related features.
    
    This class embeds the source (`src`) and distance-related (`dist`) sequences using 
    separate linear projections for the numerical features, embedding layers for categorical 
    features, and adds positional encoding for both types of inputs.

    Attributes:
        num_features (int): Number of features in the source (`src`) input.
        dist_len (int): Number of distance-related features.
        type_len (int): Number of types in the categorical input.
        size (int): Total embedding size (sum of source feature dimensions).
        src_dims (list[int]): Embedding dimensions for the source sequence.
        dist_dims (list[int]): Embedding dimensions for the distance sequence.
        linear_layers_src (nn.ModuleList): Linear layers for embedding source features.
        linear_layers_distance (nn.ModuleList): Linear layers for embedding distance features.
        type_layers_distance (nn.ModuleList): Embedding layers for categorical features in the distance sequence.
        embedding_layer_src (nn.Embedding): Embedding layer for the last categorical feature of source sequence.
        positional_encoding_src (torch.Tensor): Positional encoding for the source sequence.
        positional_encoding_distance (torch.Tensor): Positional encoding for the distance sequence.
    """
    
    def __init__(self, src_dims, dist_dims, type_dims, num_types, num_types_dist, env_dist, sequence_length_src=10):
        """
        Initializes the embedding class with specified dimensions and sequence lengths.
        
        Args:
            src_dims (list[int]): List of embedding dimensions for the features in the source sequence.
            dist_dims (list[int]): List of embedding dimensions for the features in the distance sequence.
            type_dims (list[int]): List of embedding dimensions for categorical features.
            num_types (int): Number of distinct categories for the last feature of the source sequence.
            num_types_dist (int): Number of distinct categories for the categorical features in the distance sequence.
            sequence_length_src (int, optional): Length of the source sequence. Defaults to 10.
        """
        super(SEASTAREmbedding, self).__init__()
        self.num_features = len(src_dims)
        self.dist_len = len(dist_dims)
        self.type_len = len(type_dims)
        self.size = sum(src_dims)
        self.src_dims = src_dims
        self.dist_dims = dist_dims
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Linear layers for projecting the first features of the source sequence
        self.linear_layers_src = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in src_dims[:self.num_features - 1]  # Exclude the last feature for different treatment
        ])

        self.type_env_distance = nn.ModuleList([
            nn.Embedding(env_dist + 11, embedding_dim).to(self.device)
            for embedding_dim in type_dims[:]
        ])

        self.linear_layers_env = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in dist_dims[:]
        ])

        self.type_layers_env = nn.ModuleList([
            nn.Embedding(env_dist + num_types , embedding_dim).to(self.device)
            for embedding_dim in type_dims[:]
        ])
        # Linear layers for projecting distance-related features
        self.linear_layers_distance = nn.ModuleList([
            nn.Linear(1, embedding_dim)
            for embedding_dim in dist_dims[:]
        ])
        
        # Embedding layers for categorical features in the distance sequence
        self.type_layers_distance = nn.ModuleList([
            nn.Embedding(num_types_dist + num_types , embedding_dim).to(self.device)
            for embedding_dim in type_dims[:]
        ])
        # print("_______________")
        # print(num_types_dist)#9
        # print(num_types)#4
        # print(src_dims[-1])#4
        # print(env_dist)#10
        # print("_______________")
        # Embedding layer for the last categorical feature of the source sequence
        self.embedding_layer_src = nn.Embedding(num_types, src_dims[-1]).to(self.device)

        # Generate positional encoding for source and distance sequences
        self.positional_encoding_src = self._generate_positional_encoding(self.size, sequence_length_src).to(self.device)
        self.positional_encoding_distance = self._generate_positional_encoding(self.size, sequence_length_src).to(self.device)

    def _generate_positional_encoding(self, embedding_size, sequence_length):
        """
        Generates a sinusoidal positional encoding for a given sequence length and embedding size.

        Args:
            embedding_size (int): The dimensionality of the embedding space.
            sequence_length (int): The length of the sequence for which to generate the encoding.

        Returns:
            torch.Tensor: A tensor containing the positional encoding for the sequence.
        """
        positional_encoding = torch.zeros(sequence_length, embedding_size).to(self.device)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float().to(self.device) * (-torch.log(torch.tensor(10000.0)) / embedding_size)).to(self.device)
        
        # Apply sine and cosine functions to alternate dimensions of the encoding
        positional_encoding[:, 0::2] = torch.sin(position * div_term).to(self.device)
        positional_encoding[:, 1::2] = torch.cos(position * div_term).to(self.device)

        return positional_encoding.unsqueeze(0).to(self.device)  # Add batch dimension

    def forward(self, input_tensor, is_src=True, src_tensor=None, env_tensor_flag=None, type_tensor=None):
        batch_size, seq_length, num_features = input_tensor.size()

        embedding_dims = self.src_dims if is_src else self.dist_dims
        embedded_tensor = torch.zeros(batch_size, seq_length, self.size).to(self.device)

        if is_src:
            start_index = 0
            for i, linear_layer in enumerate(self.linear_layers_src):
                linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1).float()).to(self.device)
                embedding_dim = embedding_dims[i]
                end_index = start_index + embedding_dim
                embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).to(self.device)
                start_index = end_index
            embedded_tensor[:, :, start_index:] = self.embedding_layer_src(input_tensor[:, :, 3].long()).to(self.device)

        else:
            if type_tensor is None:
                raise ValueError("[ERROR] type_tensor must be provided when is_src is False.")
            elif env_tensor_flag:
                embedded_tensor = torch.zeros(batch_size, seq_length, self.size).to(self.device)
                start_index = 0
                # print(input_tensor.shape)
                # print(len(list(zip(self.linear_layers_distance, self.type_layers_distance))))
                # print(len(self.type_layers_distance))
                # print(len(self.linear_layers_distance))
                # print(len(list(zip(self.linear_layers_env, self.type_layers_env))))
                # print(len(self.type_layers_env))
                # print(len(self.linear_layers_env))
                # print(type_tensor)

                for i, (linear_layer, embedding_layer) in enumerate(zip( self.type_layers_env,self.type_env_distance)):
                    
                    linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1).long()).to(self.device)
                    embedding_dim = embedding_dims[i]

                    embedding_output = embedding_layer(type_tensor[:, :, i].long()).to(self.device)

                    embedded_tensor[:, :, start_index:start_index + embedding_output.shape[-1]] = embedding_output
                    start_index = start_index + embedding_output.shape[-1]

                    end_index = start_index + embedding_dim
                    embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).to(self.device)
                    start_index = end_index
            else:
                embedded_tensor = torch.zeros(batch_size, seq_length, self.size).to(self.device)
                start_index = 0
                # print(input_tensor.shape)
                # print(len(list(zip(self.linear_layers_distance, self.type_layers_distance))))
                # print(len(self.type_layers_distance))
                # print(len(self.linear_layers_distance))
                # print(type_tensor)

                for i, (linear_layer, embedding_layer) in enumerate(zip(self.linear_layers_distance, self.type_layers_distance)):
                    
                    linear_output = linear_layer(input_tensor[:, :, i].view(-1, 1).float()).to(self.device)
                    embedding_dim = embedding_dims[i]

                    embedding_output = embedding_layer(type_tensor[:, :, i].long()).to(self.device)

                    embedded_tensor[:, :, start_index:start_index + embedding_output.shape[-1]] = embedding_output
                    start_index = start_index + embedding_output.shape[-1]

                    end_index = start_index + embedding_dim
                    embedded_tensor[:, :, start_index:end_index] = linear_output.view(batch_size, seq_length, embedding_dim).to(self.device)
                    start_index = end_index
        positional_encoding = self.positional_encoding_src if is_src else self.positional_encoding_distance
        embedded_tensor = embedded_tensor + positional_encoding
        embedded_tensor = embedded_tensor.view(batch_size, seq_length, -1).to(self.device)

        return embedded_tensor.to(self.device)
