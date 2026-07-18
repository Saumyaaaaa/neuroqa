"""EDF loading utilities for NeuroQA preprocessing."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mne
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ChannelNotFoundError(Exception):
    """Raised when one or more requested EEG channels are absent from an EDF file."""


class SamplingRateError(Exception):
    """Raised when an EDF file's sampling rate cannot be reconciled with the target rate."""


def load_edf(
    filepath: str,
    target_channels: list[str],
    target_sfreq: int = 256,
) -> tuple[np.ndarray, dict[str, str | int | float]]:
    """Load and preprocess an EDF file for NeuroQA.

    Loads the file, validates channel availability, selects the requested
    channels, optionally resamples to the target sampling rate, and returns
    the EEG data array with accompanying metadata.

    Args:
        filepath: Path to the EDF file on disk.
        target_channels: Ordered list of channel names to extract.
        target_sfreq: Desired sampling rate in Hz. Defaults to 256.

    Returns:
        A tuple containing:
            - EEG data as a float32 array of shape ``(n_channels, n_timepoints)``.
            - Metadata dictionary with keys ``subject_id``, ``sampling_rate``,
              ``duration_seconds``, and ``n_channels``.

    Raises:
        ValueError: If ``filepath`` does not exist.
        ChannelNotFoundError: If any ``target_channels`` are missing from the file.
    """
    path = Path(filepath)
    if not path.is_file():
        raise ValueError(f"EDF file not found: {filepath}")

    raw = mne.io.read_raw_edf(str(path), preload=True)

    available_channels = set(raw.ch_names)
    missing_channels = [
        channel for channel in target_channels if channel not in available_channels
    ]
    if missing_channels:
        missing_list = ", ".join(missing_channels)
        raise ChannelNotFoundError(
            f"The following channels were not found in the EDF file: {missing_list}"
        )

    raw.pick(target_channels)

    current_sfreq = int(raw.info["sfreq"])
    if current_sfreq != target_sfreq:
        logger.warning(
            "Resampling from %d Hz to %d Hz for file: %s",
            current_sfreq,
            target_sfreq,
            filepath,
        )
        raw.resample(target_sfreq)

    data = raw.get_data().astype(np.float32)
    duration_seconds = float(raw.n_times / raw.info["sfreq"])

    metadata: dict[str, str | int | float] = {
        "subject_id": path.stem,
        "sampling_rate": target_sfreq,
        "duration_seconds": duration_seconds,
        "n_channels": len(target_channels),
    }

    return data, metadata


def _load_default_config() -> dict:
    """Load the project default YAML configuration.

    Returns:
        Parsed configuration dictionary from ``configs/default.yaml``.
    """
    config_path = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main(filepath: str) -> dict[str, str | int | float]:
    """Load an EDF file using standard 10-20 channels from the default config.

    Args:
        filepath: Path to the EDF file on disk.

    Returns:
        Metadata dictionary produced by :func:`load_edf`.
    """
    config = _load_default_config()
    target_channels: list[str] = config["data"]["channels"]
    target_sfreq: int = config["data"]["sampling_rate"]

    _, metadata = load_edf(
        filepath,
        target_channels,
        target_sfreq=target_sfreq,
    )
    return metadata


if __name__ == "__main__":
    metadata = main(sys.argv[1])
    print(metadata)
