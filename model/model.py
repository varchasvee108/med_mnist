import torch
import torch.nn as nn
from core.config import Config
from model.layers import PatchEmbedding, TransformerBlock


class ViT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.embd_dim))
        num_patches = self.patch_embedding.num_patches

        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches + 1, config.model.embd_dim)
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.model.num_layers)]
        )
        self.norm = nn.LayerNorm(config.model.embd_dim)
        self.head = nn.Linear(config.model.embd_dim, config.data.num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embedding(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_emb

        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)

        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits
