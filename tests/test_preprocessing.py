"""Unit tests for the EDF loader utility."""

from __future__ import annotations
import os
import pytest
import numpy as np
import mne
from neuroqa.preprocessing.loader import load_edf, ChannelNotFoundError


@pytest.fixture
def mock_edf_file(tmp_path) -> str:
    """Fixture that creates a temporary mock EDF file with standard channels."""
    # Define basic metadata for a 1-second synthetic EEG recording
    ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sfreq = 100  # Start with 100 Hz to test the loader's resampling logic later
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    
    # Generate random data: 10 channels, 100 time points
    data = np.random.randn(10, 100).astype(np.float64)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Export to a temporary path as an EDF file
    file_path = tmp_path / "test_subject.edf"
    mne.export.export_raw(str(file_path), raw, fmt="edf", overwrite=True, verbose=False)
    
    return str(file_path)


def test_load_edf_success(mock_edf_file):
    """Verifies successful loading, channel selection, and resampling."""
    target_channels = ["Fp1", "Fp2", "C3", "C4"]
    target_sfreq = 256
    
    data, metadata = load_edf(
        filepath=mock_edf_file,
        target_channels=target_channels,
        target_sfreq=target_sfreq
    )
    
    # Assert data matrix specs are correct
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.shape[0] == len(target_channels)
    
    # Assert metadata integrity
    assert metadata["subject_id"] == "test_subject"
    assert metadata["sampling_rate"] == target_sfreq
    assert metadata["n_channels"] == len(target_channels)
    assert metadata["duration_seconds"] > 0


def test_load_edf_file_not_found():
    """Verifies that a ValueError is raised if the target file path does not exist."""
    with pytest.raises(ValueError, match="EDF file not found"):
        load_edf("non_existent_file.edf", ["Fp1"])


def test_load_edf_channel_not_found(mock_edf_file):
    """Verifies that ChannelNotFoundError is raised if an invalid channel is requested."""
    with pytest.raises(ChannelNotFoundError, match="required channels were not found"):
        load_edf(mock_edf_file, ["Fp1", "INVALID_CHANNEL"])