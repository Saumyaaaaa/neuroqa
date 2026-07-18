"""PyTorch Dataset utilities for EEG artifact classification."""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from neuroqa.preprocessing.filters import apply_standard_filters
from neuroqa.preprocessing.loader import ChannelNotFoundError, load_edf
from neuroqa.preprocessing.segmenter import segment_signal

logger = logging.getLogger(__name__)


class EEGArtifactDataset(Dataset):
    """PyTorch dataset of segmented EEG windows labeled for artifact detection."""

    def __init__(
        self,
        edf_paths: list[str],
        labels: list[int],
        target_channels: list[str],
        target_sfreq: int = 256,
        window_seconds: float = 2.0,
        overlap: float = 0.5,
        apply_filters: bool = True,
    ) -> None:
        """Load, filter, segment, and label EEG recordings from EDF files.

        Each successfully processed EDF file may yield multiple segments. Every
        segment inherits the label assigned to its source file. Files that raise
        :class:`ChannelNotFoundError` or :class:`ValueError` are skipped.

        Args:
            edf_paths: Paths to EDF files.
            labels: Integer class label per file (0 or 1).
            target_channels: Channel names to extract from each recording.
            target_sfreq: Target sampling rate in Hz. Defaults to 256.
            window_seconds: Segment window length in seconds. Defaults to 2.0.
            overlap: Fractional overlap between consecutive windows. Defaults to 0.5.
            apply_filters: Whether to apply standard bandpass filtering. Defaults to True.
        """
        if len(edf_paths) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(edf_paths)} edf_paths vs {len(labels)} labels"
            )

        self._segments: list[np.ndarray] = []
        self._labels: list[int] = []
        loaded_files = 0

        for edf_path, label in zip(edf_paths, labels, strict=True):
            try:
                data, metadata = load_edf(
                    edf_path,
                    target_channels,
                    target_sfreq=target_sfreq,
                )
                if apply_filters:
                    data = apply_standard_filters(
                        data,
                        float(metadata["sampling_rate"]),
                    )

                segments = segment_signal(
                    data,
                    float(target_sfreq),
                    window_seconds=window_seconds,
                    overlap=overlap,
                )
                for segment in segments:
                    self._segments.append(segment)
                    self._labels.append(label)

                loaded_files += 1
            except (ChannelNotFoundError, ValueError) as error:
                logger.warning("Skipping file %s: %s", edf_path, error)

        logger.info(
            "Loaded %d segments from %d files",
            len(self._segments),
            loaded_files,
        )

    def __len__(self) -> int:
        """Return the total number of segments across all loaded files."""
        return len(self._segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a segment and its class label.

        Args:
            idx: Index of the segment to retrieve.

        Returns:
            A tuple containing:
                - Segment tensor of shape ``(n_channels, window_samples)``.
                - Scalar class label as a long tensor.
        """
        segment = torch.as_tensor(self._segments[idx], dtype=torch.float32)
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return segment, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training.

        Returns:
            Float tensor of shape ``(2,)`` where
            ``weight[i] = total_samples / (n_classes * count_of_class_i)``.
        """
        n_classes = 2
        counts = np.zeros(n_classes, dtype=np.float64)
        for label in self._labels:
            counts[label] += 1

        total_samples = len(self._labels)
        weights = np.zeros(n_classes, dtype=np.float64)
        for class_index in range(n_classes):
            if counts[class_index] > 0:
                weights[class_index] = total_samples / (
                    n_classes * counts[class_index]
                )

        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    edf_paths: list[str],
    labels: list[int],
    target_channels: list[str],
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create stratified train and validation DataLoaders with balanced sampling.

    Args:
        edf_paths: Paths to EDF files.
        labels: Integer class label per file.
        target_channels: Channel names to extract from each recording.
        batch_size: Batch size for both loaders. Defaults to 32.
        val_split: Fraction of files held out for validation. Defaults to 0.2.
        seed: Random seed for the train/validation split. Defaults to 42.
        num_workers: Number of DataLoader worker processes. Defaults to 0.

    Returns:
        A tuple of ``(train_loader, val_loader)``. The training loader uses a
        :class:`WeightedRandomSampler` derived from inverse class frequencies.
    """
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        edf_paths,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    train_dataset = EEGArtifactDataset(
        train_paths,
        train_labels,
        target_channels,
    )
    val_dataset = EEGArtifactDataset(
        val_paths,
        val_labels,
        target_channels,
    )

    class_weights = train_dataset.get_class_weights()
    sample_weights = torch.DoubleTensor(
        [class_weights[label].item() for label in train_dataset._labels]
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
