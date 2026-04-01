import torch
import torch.nn as nn
from torch.nn import functional as F
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
