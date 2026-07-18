"""Transformer-based EEG artifact classifier for NeuroQA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for temporal sequence order."""

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize sinusoidal positional encoding.

        Args:
            d_model: Model embedding dimension.
            max_len: Maximum supported sequence length including the CLS token.
            dropout: Dropout probability applied after encoding addition.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape ``(batch_size, sequence_length, d_model)``.

        Returns:
            Encoded tensor with the same shape as ``x``.
        """
        encoded = x + self.encoding[:, : x.size(1)]
        return self.dropout(encoded)


class EEGTransformer(nn.Module):
    """Transformer encoder for multichannel EEG artifact classification."""

    def __init__(
        self,
        n_channels: int = 10,
        window_samples: int = 512,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        n_classes: int = 2,
    ) -> None:
        """Initialize the EEG transformer classifier.

        Args:
            n_channels: Number of EEG input channels.
            window_samples: Number of time samples per input window.
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Feedforward hidden dimension.
            dropout: Dropout probability.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_classes = n_classes

        self.spatial_projection = nn.Linear(n_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=window_samples + 1,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify EEG windows into artifact categories.

        Args:
            x: Input tensor of shape ``(batch_size, n_channels, window_samples)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        batch_size = x.size(0)

        sequence = x.permute(0, 2, 1)
        sequence = self.spatial_projection(sequence)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, sequence], dim=1)
        sequence = self.positional_encoding(sequence)
        encoded = self.transformer_encoder(sequence)

        cls_output = encoded[:, 0, :]
        return self.classifier(cls_output)
