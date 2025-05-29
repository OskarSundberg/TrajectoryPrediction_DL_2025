#
# Created by Linus Savinainen and Oskar Sundberg 2025
#


import torch
import torch.nn as nn

class SEASTAREmbedding(nn.Module):
    """    
    Args:
        src_dims (list[int]): List of embedding dimensions for the features in the source sequence.
        dist_dims (list[int]): List of embedding dimensions for the features in the distance sequence.
        type_dims (list[int]): List of embedding dimensions for categorical features.
        num_types (int): Number of distinct categories for the last feature of the source sequence.
        num_types_dist (int): Number of distinct categories for the categorical features in the distance sequence.
        sequence_length_src (int, optional): Length of the source sequence. Defaults to 10.
    """
    def __init__(
        self,
        src_dims: list[int],
        dist_dims: list[int],
        type_dims: list[int],
        env_dims:  list[int],
        num_types_src:  int,
        num_types_dist: int,
        sequence_length_src: int = 10
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save dims
        self.src_dims  = src_dims
        self.dist_dims = dist_dims
        self.type_dims = type_dims
        self.embedding_size = sum(src_dims)

        # === Define source embeddings ===
        self.linear_layers_src = nn.ModuleList([
            nn.Linear(1, d) for d in src_dims[:-1]
        ])
        self.embedding_src = nn.Embedding(num_types_src, src_dims[-1]).to(self.device)

        # === Define distance embeddings (agent–agent) ===
        self.linear_layers_dist = nn.ModuleList([
            nn.Linear(1, d) for d in dist_dims
        ])
        self.embedding_layers_dist = nn.ModuleList([
            nn.Embedding(num_types_dist, d) for d in type_dims
        ]).to(self.device)

         # === Define environment embeddings (agent–env) ===
        self.linear_layers_env = nn.ModuleList([
            nn.Linear(1, d) for d in env_dims
        ])

        # === Define positional encoding ===
        self.positional_encoding = self._generate_positional_encoding(
            self.embedding_size,
            sequence_length_src
        ).to(self.device)

    def _generate_positional_encoding(self, dim: int, length: int) -> torch.Tensor:
        """
        Generates a sinusoidal positional encoding for a given sequence length and embedding size.

        Args:
            embedding_size (int): The dimensionality of the embedding space.
            sequence_length (int): The length of the sequence for which to generate the encoding.

        Returns:
            torch.Tensor: A tensor containing the positional encoding for the sequence.
        """
        pe = torch.zeros(1, length, dim, device=self.device)
        pos = torch.arange(length, device=self.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2, device=self.device).float() *
                        (-torch.log(torch.tensor(10000.0)) / dim))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe


    #Input tensor shapes:
    #B ==> Batch size: number of independent samples in a batch
    #S ==> Sequence length: number time steps per sample
    #G ==> Number of interaction features 

    def forward(
        self,
        src: torch.Tensor,           #(B, S=src_len, F_src)
        dist: torch.Tensor,          #(B, S, G) numeric agent–agent
        type_dist: torch.Tensor,     #(B, S, G) categorical agent–agent
        env_dist: torch.Tensor       #(B, S, G) numeric agent–env
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = src.shape
        D = self.embedding_size

        # --- Source ---
        # Initialize source embedding tensor (B, S, D)
        src_emb = torch.zeros(B, S, D, device=self.device)
        idx = 0

        for i, linear in enumerate(self.linear_layers_src):
            out = linear(src[:, :, i].unsqueeze(-1))       #Project feature i to embedding space
            d = self.src_dims[i]                               #Dimensionality of the current feature's embedding
            src_emb[:, :, idx:idx+d] = out.view(B, S, d)
            idx += d

        
       
        cat = self.embedding_src(src[:, :, -1].long())   #Embedding for the last feature, i.e., the class of the closest agent
        src_emb[:, :, idx:] = cat

        # --- Distance (agent–agent) ---
        dist_emb = torch.zeros(B, S, D, device=self.device)
        idx = 0
        for i, (linear, emb) in enumerate(zip(self.linear_layers_dist, self.embedding_layers_dist)):
            c = emb(type_dist[:, :, i].long())       
            cd = c.size(-1)
            dist_emb[:, :, idx:idx+cd] = c
            idx += cd

            n = linear(dist[:, :, i].unsqueeze(-1))    #Project each numeric env feature to embedding space
            nd = n.size(-1)
            nd = n.size(-1)
            dist_emb[:, :, idx:idx+nd] = n.view(B, S, nd)
            idx += nd

        # --- Environment---
        #Initialize A2E distance embedding tensor (B, S, D)
        env_emb = torch.zeros(B, S, D, device=self.device)
        idx = 0
        for i, linear in enumerate(self.linear_layers_env):
            n = linear(env_dist[:, :, i].unsqueeze(-1))
            nd = n.size(-1)
            env_emb[:, :, idx:idx+nd] = n.view(B, S, nd)
            idx += nd

        # --- Add positional encoding ---
        #Positional encoding to all three modalities to retain order information
        pe = self.positional_encoding[:, :S, :]
        src_emb  += pe
        dist_emb += pe
        env_emb  += pe

        return src_emb, dist_emb, env_emb
