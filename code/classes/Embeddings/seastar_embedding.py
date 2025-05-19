import torch
import torch.nn as nn

class SEASTAREmbedding(nn.Module):
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

        # === Source embeddings ===
        self.linear_layers_src = nn.ModuleList([
            nn.Linear(1, d) for d in src_dims[:-1]
        ])
        self.embedding_src = nn.Embedding(num_types_src, src_dims[-1]).to(self.device)

        # === Distance embeddings (agent–agent) ===
        self.linear_layers_dist = nn.ModuleList([
            nn.Linear(1, d) for d in dist_dims
        ])
        self.embedding_layers_dist = nn.ModuleList([
            nn.Embedding(num_types_dist, d) for d in type_dims
        ]).to(self.device)

         # === Environment embeddings (agent–env) ===
        self.linear_layers_env = nn.ModuleList([
            nn.Linear(1, d) for d in env_dims
        ])

        # === Positional encoding ===
        self.positional_encoding = self._generate_positional_encoding(
            self.embedding_size,
            sequence_length_src
        ).to(self.device)

    def _generate_positional_encoding(self, dim: int, length: int) -> torch.Tensor:
        pe = torch.zeros(1, length, dim, device=self.device)
        pos = torch.arange(length, device=self.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2, device=self.device).float() *
                        (-torch.log(torch.tensor(10000.0)) / dim))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        src: torch.Tensor,           # (B, S=src_len, F_src)
        dist: torch.Tensor,          # (B, S, G) numeric agent–agent
        type_dist: torch.Tensor,     # (B, S, G) categorical agent–agent
        env_dist: torch.Tensor       # (B, S, G) numeric agent–env
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = src.shape
        D = self.embedding_size

        # --- Source ---
        src_emb = torch.zeros(B, S, D, device=self.device)
        idx = 0
        for i, linear in enumerate(self.linear_layers_src):
            out = linear(src[:, :, i].unsqueeze(-1))        # (B*S, d)
            d = self.src_dims[i]
            src_emb[:, :, idx:idx+d] = out.view(B, S, d)
            idx += d
        cat = self.embedding_src(src[:, :, -1].long())      # (B, S, last_dim)
        src_emb[:, :, idx:] = cat

        # --- Distance (agent–agent) ---
        dist_emb = torch.zeros(B, S, D, device=self.device)
        idx = 0
        for i, (linear, emb) in enumerate(zip(self.linear_layers_dist, self.embedding_layers_dist)):
            c = emb(type_dist[:, :, i].long())              # (B, S, c_dim)
            cd = c.size(-1)
            dist_emb[:, :, idx:idx+cd] = c
            idx += cd

            n = linear(dist[:, :, i].unsqueeze(-1))         # (B*S, n_dim)
            nd = n.size(-1)
            dist_emb[:, :, idx:idx+nd] = n.view(B, S, nd)
            idx += nd

        # --- Environment (numeric only) ---
        env_emb = torch.zeros(B, S, D, device=self.device)
        idx = 0
        for i, linear in enumerate(self.linear_layers_env):
            n = linear(env_dist[:, :, i].unsqueeze(-1))
            nd = n.size(-1)
            env_emb[:, :, idx:idx+nd] = n.view(B, S, nd)
            idx += nd

        # --- Add positional encoding ---
        pe = self.positional_encoding[:, :S, :]
        src_emb  += pe
        dist_emb += pe
        env_emb  += pe

        return src_emb, dist_emb, env_emb
