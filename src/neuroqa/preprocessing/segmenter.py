"""EEG signal segmentation utilities for NeuroQA preprocessing."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def segment_signal(
    data: np.ndarray,
    sfreq: int,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """Segment multichannel EEG data into overlapping fixed-length windows.

    Args:
        data: EEG data array of shape ``(n_channels, n_timepoints)``.
        sfreq: Sampling frequency in Hz.
        window_seconds: Duration of each window in seconds. Defaults to 2.0.
        overlap: Fractional overlap between consecutive windows. Must be in
            ``(0.0, 1.0)``. Defaults to 0.5.

    Returns:
        Segmented data as a float32 array of shape
        ``(n_segments, n_channels, window_samples)``.

    Raises:
        ValueError: If ``sfreq``, ``window_seconds``, or ``overlap`` are invalid,
            or if the signal is shorter than one window.
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap <= 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be between 0.0 and 1.0 exclusive")

    n_timepoints = data.shape[1]
    window_samples = int(window_seconds * sfreq)
    step_size = int(window_samples * (1 - overlap))

    if n_timepoints < window_samples:
        raise ValueError("Signal too short for one window")

    segments: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_timepoints:
        segments.append(data[:, start : start + window_samples])
        start += step_size

    segmented = np.stack(segments, axis=0).astype(np.float32)
    n_segments = segmented.shape[0]

    logger.debug("Extracted %d segments", n_segments)

    return segmented


def reconstruct_info(
    n_segments: int,
    sfreq: int,
    window_seconds: float,
    overlap: float,
) -> dict[str, int | float]:
    """Compute segmentation metadata for logging and reproducibility reports.

    Args:
        n_segments: Number of extracted segments.
        sfreq: Sampling frequency in Hz.
        window_seconds: Duration of each window in seconds.
        overlap: Fractional overlap between consecutive windows.

    Returns:
        Dictionary containing ``n_segments``, ``window_samples``, ``step_size``,
        and ``total_seconds_covered``.
    """
    window_samples = int(window_seconds * sfreq)
    step_size = int(window_samples * (1 - overlap))
    total_seconds_covered = (step_size * (n_segments - 1) + window_samples) / sfreq

    return {
        "n_segments": n_segments,
        "window_samples": window_samples,
        "step_size": step_size,
        "total_seconds_covered": total_seconds_covered,
    }
