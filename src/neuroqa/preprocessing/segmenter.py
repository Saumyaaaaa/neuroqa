"""Signal segmentation utilities for sliding window extraction."""

from __future__ import annotations
import numpy as np

def segment_signal(
    data: np.ndarray, 
    sfreq: float, 
    window_seconds: float = 2.0, 
    overlap: float = 0.5
) -> list[np.ndarray]:
    """Slices a continuous multi-channel EEG signal into overlapping time windows.

    Args:
        data: Processed EEG array of shape (n_channels, n_timepoints).
        sfreq: The sampling frequency of the recording in Hz.
        window_seconds: Duration of each chunk in seconds. Defaults to 2.0.
        overlap: Fraction of overlap between consecutive windows (0.0 to 1.0). Defaults to 0.5.

    Returns:
        list[np.ndarray]: A list of segmented window matrices, each of shape 
            (n_channels, window_samples).
    """
    n_channels, n_timepoints = data.shape
    window_samples = int(window_seconds * sfreq)
    step_samples = int(window_samples * (1.0 - overlap))
    
    segments = []
    start = 0
    
    while start + window_samples <= n_timepoints:
        end = start + window_samples
        chunk = data[:, start:end]
        segments.append(chunk)
        start += step_samples
        
    return segments