import torch
import torch.nn as nn
from classes.Embeddings.seastar_embedding import SEASTAREmbedding

class SEASTAR(nn.Module):
    def __init__(
        self,
        src_dims: list[int],
        dist_dims: list[int],
        type_dims: list[int],
        num_types_src: int,
        num_types_dist: int,
        hidden: int = 256,
        num_layers: int = 16,
        num_heads: int = 8,
        src_len: int = 10,
        tgt_len: int = 40,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_size = sum(src_dims)
        self.tgt_len        = tgt_len
        self.output_size    = 2
        self.hidden_size    = hidden
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding
        self.embedding_layer = SEASTAREmbedding(
            src_dims, dist_dims, type_dims,
            num_types_src, num_types_dist,
            sequence_length_src=src_len
        ).to(self.device)

        # Transformer layer factory
        def make_enc():
            return nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=num_heads,
                batch_first=True
            ).to(self.device)

        # First-pass encoders
        self.temporal_encoder_1      = nn.TransformerEncoder(make_enc(), num_layers)
        self.spatial_encoder_1       = nn.TransformerEncoder(make_enc(), num_layers)
        self.environmental_encoder_1 = nn.TransformerEncoder(make_enc(), num_layers)

        # Fusion
        self.fusion_layer = nn.Linear(self.embedding_size * 3, self.embedding_size)

        # Second-pass encoders
        self.spatial_encoder_2       = nn.TransformerEncoder(make_enc(), num_layers)
        self.environmental_encoder_2 = nn.TransformerEncoder(make_enc(), num_layers)
        self.temporal_encoder_2      = nn.TransformerEncoder(make_enc(), num_layers)

        # Single-shot decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.tgt_len * self.output_size)
        ).to(self.device)

        # Helpers
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu     = nn.ReLU()
        self.relu1    = nn.ReLU()
        self.relu2    = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        dist: torch.Tensor,
        type_dist: torch.Tensor,
        env_dist: torch.Tensor,
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, S, _ = src.shape

        # 1) Embedding
        src_e, spat_e, env_e = self.embedding_layer(
            src, dist, type_dist,
            env_dist
        )
        src_e  = self.relu(self.dropout(src_e))
        spat_e = self.relu1(self.dropout1(spat_e))
        env_e  = self.relu2(self.dropout2(env_e))

        # 2) First-pass encoding
        t1 = self.temporal_encoder_1(src_e,   mask=src_mask)
        s1 = self.spatial_encoder_1(spat_e,  mask=src_mask)
        e1 = self.environmental_encoder_1(env_e, mask=src_mask)

        # 3) Fusion
        fused = torch.cat([t1, e1, s1], dim=-1)
        fused = self.fusion_layer(fused)

        # 4) Second-pass encoding
        x = self.spatial_encoder_2(fused)
        x = self.environmental_encoder_2(x)
        x = self.temporal_encoder_2(x)  # → (B, S, embed)

        # === 5) Decode from the *last* source step ===
        last = x[:, -1, :]               # (B, embed)
        out  = self.decoder(last)        # (B, tgt_len * 2)
        pred = out.view(B, self.tgt_len, self.output_size)  # (B, 40, 2)
        return pred
