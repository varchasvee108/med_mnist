import torch.nn as nn
from core.config import Config


class PatchEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_patches = (config.model.img_size[0] // config.model.patch_size[0]) * (
            config.model.img_size[1] // config.model.patch_size[1]
        )

        self.proj = nn.Conv2d(
            in_channels=config.model.in_channels,
            out_channels=config.model.embd_dim,
            kernel_size=config.model.patch_size,
            stride=config.model.patch_size,
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.proj(x)
        # x shape: [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(
                config.model.embd_dim,
                int(config.model.mlp_ratio * config.model.embd_dim),
            ),
            nn.GELU(),
            nn.Linear(
                int(config.model.mlp_ratio * config.model.embd_dim),
                config.model.embd_dim,
            ),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert config.model.embd_dim % config.model.num_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim=config.model.embd_dim,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.model.dropout)

    def forward(self, x):
        # x shape: [B, T, C]
        x, _ = self.attn(x, x, x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.model.embd_dim)
        self.norm2 = nn.LayerNorm(config.model.embd_dim)
        self.mlp = MLP(config)
        self.attn = Attention(config)

    def forward(self, x):
        # x shape: [B, T, C]
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        x = x + self.mlp(self.norm2(x))
        return x
