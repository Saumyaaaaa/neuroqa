"""Signal filtering utilities for raw EEG preprocessing."""

from __future__ import annotations
import mne
import numpy as np

def apply_standard_filters(
    data: np.ndarray, 
    sfreq: float, 
    l_freq: float = 1.0, 
    h_freq: float = 40.0
) -> np.ndarray:
    """Applies a bandpass filter to remove low-frequency drift and high-frequency noise.

    Args:
        data: Raw EEG array of shape (n_channels, n_timepoints).
        sfreq: The sampling rate of the signal in Hz.
        l_freq: Cut-off frequency for the high-pass filter. Defaults to 1.0 Hz.
        h_freq: Cut-off frequency for the low-pass filter. Defaults to 40.0 Hz.

    Returns:
        np.ndarray: The bandpass filtered EEG data array cast as float32.
    """
    # Wrap array in temporary MNE structure to leverage its robust digital filtering algorithms
    info = mne.create_info(ch_names=data.shape[0], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose=False)
    
    # Apply standard zero-phase FIR bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    
    return raw.get_data().astype(np.float32)