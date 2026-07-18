"""EEG filtering utilities for NeuroQA preprocessing."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

logger = logging.getLogger(__name__)


def bandpass_filter(
    data: np.ndarray,
    sfreq: int,
    low_freq: float = 1.0,
    high_freq: float = 40.0,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to multichannel EEG data.

    Each channel is filtered independently using a 4th-order Butterworth filter
    applied with forward-backward ``filtfilt`` for zero phase distortion.

    Args:
        data: EEG data array of shape ``(n_channels, n_timepoints)``.
        sfreq: Sampling frequency in Hz.
        low_freq: Lower cutoff frequency in Hz. Defaults to 1.0.
        high_freq: Upper cutoff frequency in Hz. Defaults to 40.0.

    Returns:
        Bandpass-filtered data with the same shape as ``data``, as float32.

    Raises:
        ValueError: If cutoff frequencies are invalid relative to each other,
            to zero, or to the Nyquist frequency.
    """
    if low_freq <= 0:
        raise ValueError("low_freq must be positive")
    if low_freq >= high_freq:
        raise ValueError("low_freq must be less than high_freq")

    nyquist = sfreq / 2
    if high_freq >= nyquist:
        raise ValueError("high_freq must be less than Nyquist frequency (sfreq/2)")

    normalized_low = low_freq / nyquist
    normalized_high = high_freq / nyquist
    b, a = butter(4, [normalized_low, normalized_high], btype="bandpass")

    filtered = np.empty_like(data, dtype=np.float32)
    for channel_index in range(data.shape[0]):
        filtered[channel_index] = filtfilt(b, a, data[channel_index]).astype(
            np.float32
        )

    return filtered


def notch_filter(
    data: np.ndarray,
    sfreq: int,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a zero-phase IIR notch filter to remove power line noise.

    Each channel is filtered independently using ``iirnotch`` and ``filtfilt``.

    Args:
        data: EEG data array of shape ``(n_channels, n_timepoints)``.
        sfreq: Sampling frequency in Hz.
        notch_freq: Frequency to attenuate in Hz. Use 50.0 for Europe/Asia and
            60.0 for the USA. Defaults to 50.0.
        quality_factor: Quality factor of the notch filter. Defaults to 30.0.

    Returns:
        Notch-filtered data with the same shape as ``data``, as float32.

    Raises:
        ValueError: If ``notch_freq`` is not positive.
    """
    if notch_freq <= 0:
        raise ValueError("notch_freq must be positive")

    b, a = iirnotch(notch_freq, quality_factor, sfreq)

    filtered = np.empty_like(data, dtype=np.float32)
    for channel_index in range(data.shape[0]):
        filtered[channel_index] = filtfilt(b, a, data[channel_index]).astype(
            np.float32
        )

    return filtered


def apply_standard_filters(
    data: np.ndarray,
    sfreq: int,
    low_freq: float = 1.0,
    high_freq: float = 40.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """Apply standard bandpass and notch filters in sequence.

    Args:
        data: EEG data array of shape ``(n_channels, n_timepoints)``.
        sfreq: Sampling frequency in Hz.
        low_freq: Lower bandpass cutoff in Hz. Defaults to 1.0.
        high_freq: Upper bandpass cutoff in Hz. Defaults to 40.0.
        notch_freq: Notch frequency in Hz. Defaults to 50.0.

    Returns:
        Filtered data with the same shape as ``data``, as float32.
    """
    filtered = bandpass_filter(data, sfreq, low_freq=low_freq, high_freq=high_freq)
    filtered = notch_filter(filtered, sfreq, notch_freq=notch_freq)

    logger.info(
        "Applied standard filters: bandpass [%s-%s Hz], notch [%s Hz]",
        low_freq,
        high_freq,
        notch_freq,
    )

    return filtered
