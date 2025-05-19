import torch
import torch.nn as nn
from classes.Embeddings.seastar_embedding import SEASTAREmbedding

class SEASTAR(nn.Module):
    def __init__(
        self,
        src_dims,
        dist_dims,
        type_dims,
        env_dims,
        num_types,
        num_types_dist,
        hidden=256,
        num_layers=1,
        num_heads=8,
        src_len=10,
        tgt_len=40,
        dropout=0.1
    ):
        super().__init__()
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.embedding_size = sum(src_dims)
        self.output_size    = 2

        # 1) shared embedding for all three streams
        self.embedding = SEASTAREmbedding(
            src_dims, dist_dims, type_dims, env_dims,
            num_types, num_types_dist, sequence_length_src=src_len
        )

        # 2) First‑pass transformers (one each)
        self.spatial_encoder_1      = nn.TransformerEncoderLayer(
            d_model=self.embedding_size, nhead=num_heads, batch_first=True
        )
        self.temporal_encoder_1     = nn.TransformerEncoderLayer(
            d_model=self.embedding_size, nhead=num_heads, batch_first=True
        )
        self.environmental_encoder_1= nn.TransformerEncoderLayer(
            d_model=self.embedding_size, nhead=num_heads, batch_first=True
        )

        # 3) Fusion FC
        self.fusion = nn.Linear(self.embedding_size * 3, self.embedding_size)

        # 4) Second‑pass transformers
        self.spatial_encoder_2      = nn.TransformerEncoderLayer(
            d_model=self.embedding_size, nhead=num_heads, batch_first=True
        )
        self.environmental_encoder_2= nn.TransformerEncoderLayer(
            d_model=self.embedding_size, nhead=num_heads, batch_first=True
        )
        self.temporal_encoder_2     = nn.TransformerEncoderLayer(
            d_model=self.embedding_size, nhead=num_heads, batch_first=True
        )

        # 5) Final decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.tgt_len * self.output_size)
        )

    def forward(self, src, dist, type_dist, env_dist, src_mask=None):
        """
        src:         (B, S=src_len, 4)
        dist:        (B, S, G)
        type_dist:   (B, S, G)
        env_dist:    (B, S, G)
        """
        # --- Embedding ---
        src_e, dist_e, env_e = self.embedding(src, dist, type_dist, env_dist)  
        # each is (B, S, embed_size)

        # --- Encoder 1 (parallel) ---
        sp1 = self.spatial_encoder_1(src_e, src_mask)      # Spatial from src_e
        tp1 = self.temporal_encoder_1(src_e, src_mask)     # Temporal from src_e
        en1 = self.environmental_encoder_1(src_e, src_mask)# Environmental from src_e

        # --- Fusion ---
        # concat along last dim: (B, S, 3*embed)
        fused = torch.cat([sp1, tp1, en1], dim=-1)
        fused = self.fusion(fused)                         # (B, S, embed)

        # --- Encoder 2 (sequence) ---
        x = self.spatial_encoder_2(fused, src_mask)
        x = self.environmental_encoder_2(x, src_mask)
        x = self.temporal_encoder_2(x, src_mask)            # (B, S, embed)

        # pick only the last source position to decode future?
        # or flatten all S steps?
        # Here, we decode from the final timestep:
        last = x[:, -1, :]      # (B, embed)

        # --- Decoder ---
        out = self.decoder(last) # (B, tgt_len * 2)
        pred = out.view(-1, self.tgt_len, self.output_size)  # (B, tgt_len, 2)

        return pred
