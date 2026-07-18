"""EDF loading utilities for NeuroQA preprocessing."""

from __future__ import annotations
import logging
import os
from pathlib import Path
import sys
import mne
import numpy as np
import yaml

# Module-level logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ChannelNotFoundError(Exception):
    """Exception raised when requested EEG channels are missing from the EDF file."""
    pass


class SamplingRateError(Exception):
    """Exception raised when there is an issue handling or verifying the sampling rate."""
    pass


def load_edf(
    filepath: str, 
    target_channels: list[str], 
    target_sfreq: int = 256
) -> tuple[np.ndarray, dict]:
    """Loads an EDF file, filters specific channels, validates presence, and resamples.

    Args:
        filepath: Path to the input EDF data file.
        target_channels: List of channel labels expected to be present and extracted.
        target_sfreq: The desired output sampling rate in Hz. Defaults to 256.

    Returns:
        A tuple containing:
            - np.ndarray: Processed EEG data matrix of shape (n_channels, n_timepoints)
              cast explicitly as float32.
            - dict: Extracted file metadata containing 'subject_id', 'sampling_rate',
              'duration_seconds', and 'n_channels'.

    Raises:
        ValueError: If the provided filepath does not point to an existing file.
        ChannelNotFoundError: If one or more channels in target_channels are 
            not found in the recording.
    """
    if not os.path.exists(filepath):
        raise ValueError(f"EDF file not found: {filepath}")

    # Load file with preloading enabled to allow processing operations
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # Validate channel presence
    available_channels = raw.ch_names
    missing_channels = [ch for ch in target_channels if ch not in available_channels]
    if missing_channels:
        raise ChannelNotFoundError(
            f"The following required channels were not found in the recording: {missing_channels}"
        )

    # Isolate strictly to target channels
    raw.pick_channels(target_channels, ordered=True)

    # Check sampling frequency and resample if necessary
    current_sfreq = int(raw.info["sfreq"])
    if current_sfreq != target_sfreq:
        logger.warning(
            f"Sampling rate mismatch for {filepath}. Expected {target_sfreq}Hz, found {current_sfreq}Hz. Resampling."
        )
        raw.resample(sfreq=float(target_sfreq), verbose=False)

    # Extract scientific data array as float32
    data: np.ndarray = raw.get_data().astype(np.float32)

    # Construct metadata footprint
    metadata = {
        "subject_id": str(Path(filepath).stem),
        "sampling_rate": int(raw.info["sfreq"]),
        "duration_seconds": float(raw.times[-1] + (1.0 / raw.info["sfreq"])),
        "n_channels": len(raw.ch_names),
    }

    return data, metadata


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m neuroqa.preprocessing.loader <path_to_edf>")

    input_file = sys.argv[1]
    config_path = Path("configs/default.yaml")

    # Fallback default 10-20 channels if config yaml hasn't been generated
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sfreq_target = 256

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config:
                channels = config.get("target_channels", channels)
                sfreq_target = config.get("sampling_rate", sfreq_target)

    try:
        # Purity check: execution wrapper handling output natively at shell level
        _, meta = load_edf(filepath=input_file, target_channels=channels, target_sfreq=sfreq_target)
        
        # Safe printing strictly inside execution gate
        import json
        print(json.dumps(meta, indent=4))
        
    except Exception as e:
        sys.exit(f"Execution Error: {e}")