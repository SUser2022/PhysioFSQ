"""Transformer components used by the paired frequency-FSQ Teacher."""

from __future__ import annotations

import math

import torch
from torch import nn
from einops import rearrange


class TemporalEncoder(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches=10, used_channel=[0, 1]):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.used_channel = tuple(int(channel) for channel in used_channel)
        self._identity_channel_order = self.used_channel == tuple(
            range(len(self.used_channel))
        )
        # A Python-list CUDA advanced index constructs an index tensor on every
        # forward and is not CUDA-graph safe.  This non-persistent constant
        # preserves checkpoint state while making channel selection reusable.
        self.register_buffer(
            "used_channel_index",
            torch.tensor(self.used_channel, dtype=torch.long),
            persistent=False,
        )
        num_channel = len(used_channel)

        self.linear = nn.Linear(num_channel * patch_size, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        if not (
            self._identity_channel_order
            and x.shape[2] == len(self.used_channel)
        ):
            x = x.index_select(2, self.used_channel_index)
        x = rearrange(x, 'b n_t c t -> (b n_t) (c t)')
        x = self.linear(x)
        x = self.gelu(x)
        x = rearrange(x, '(b n_t) d -> b n_t d', n_t=self.num_patches)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=1250):
        super().__init__()
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class JointEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=8, num_layers=4, drop_rate=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            activation=nn.GELU(),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_channels, patch_size, nhead=8, num_layers=4, drop_rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            activation=nn.GELU(),
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, num_channels * patch_size)
        )

    def forward(self, x_quantized):
        x = self.decoder(x_quantized)
        x = self.mlp(x)
        x_re = rearrange(x, 'b N_t (c t) -> b N_t c t', c=self.num_channels, t=self.patch_size)
        return x_re
