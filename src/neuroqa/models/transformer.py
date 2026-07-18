"""Lightweight 1D Vision Transformer for EEG artifact classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Project multichannel EEG patches into a shared embedding space."""

    def __init__(self, n_channels: int, patch_size: int, d_model: int) -> None:
        """Initialize patch embedding layers.

        Args:
            n_channels: Number of EEG channels per sample.
            patch_size: Number of time samples in each non-overlapping patch.
            d_model: Output embedding dimension.
        """
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.projection = nn.Linear(n_channels * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed non-overlapping EEG patches.

        Args:
            x: Input tensor of shape ``(batch, n_channels, sequence_length)``.

        Returns:
            Patch embeddings of shape ``(batch, n_patches, d_model)``.
        """
        batch_size, _, sequence_length = x.shape
        n_patches = sequence_length // self.patch_size

        patches = x.view(batch_size, self.n_channels, n_patches, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, n_patches, self.n_channels * self.patch_size)

        return self.projection(patches)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for transformer inputs."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        """Initialize sinusoidal positional encoding.

        Args:
            d_model: Model embedding dimension.
            max_len: Maximum supported sequence length.
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
            x: Input tensor of shape ``(batch, sequence_length, d_model)``.

        Returns:
            Encoded tensor with the same shape as ``x``.
        """
        encoded = x + self.encoding[:, : x.size(1)]
        return self.dropout(encoded)


class EEGTransformer(nn.Module):
    """Lightweight Vision Transformer for EEG artifact classification."""

    def __init__(
        self,
        n_channels: int = 19,
        sequence_length: int = 512,
        patch_size: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the EEG transformer classifier.

        Args:
            n_channels: Number of EEG input channels.
            sequence_length: Number of time samples per input window.
            patch_size: Patch length along the time axis.
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Feedforward hidden dimension.
            n_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.n_classes = n_classes
        self.dropout = dropout

        self.n_patches = sequence_length // patch_size

        self.patch_embedding = PatchEmbedding(n_channels, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(
            d_model,
            max_len=self.n_patches + 1,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify EEG windows into artifact categories.

        Args:
            x: Input tensor of shape ``(batch, n_channels, sequence_length)``.

        Returns:
            Logits of shape ``(batch, n_classes)``.
        """
        batch_size = x.size(0)

        patch_embeddings = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, patch_embeddings], dim=1)
        sequence = self.positional_encoding(sequence)
        encoded = self.transformer_encoder(sequence)
        cls_output = encoded[:, 0, :]

        return self.classifier(cls_output)

    def get_param_count(self) -> int:
        """Return the total number of trainable model parameters.

        Returns:
            Count of trainable parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def __repr__(self) -> str:
        """Return a string representation including parameter count."""
        return f"EEGTransformer(params={self.get_param_count()})"
