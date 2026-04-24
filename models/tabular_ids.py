"""Efficient tabular neural IDS models."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Class-weighted focal loss for imbalanced multiclass IDS training."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TabularTransformerIDS(nn.Module):
    """
    Small FT-Transformer-style IDS for Colab T4.
    Each numeric feature is embedded as a token, contextualized with Transformer
    layers, pooled, then passed through residual MLP layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        token_dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_scale = nn.Parameter(torch.randn(input_dim, token_dim) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, mlp_dim),
            nn.GELU(),
            ResidualBlock(mlp_dim, dropout=dropout),
            ResidualBlock(mlp_dim, dropout=dropout),
            nn.LayerNorm(mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_scale.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0])
