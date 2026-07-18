"""Model factory for NeuroQA."""

from __future__ import annotations

from neuroqa.models.transformer import EEGTransformer

_MODEL_KEYS = (
    "n_channels",
    "sequence_length",
    "patch_size",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "n_classes",
    "dropout",
)


def create_model(config: dict) -> EEGTransformer:
    """Instantiate an EEG transformer from a configuration dictionary.

    Args:
        config: Full project configuration containing a ``model`` section.

    Returns:
        Configured :class:`EEGTransformer` instance.

    Raises:
        KeyError: If ``config["model"]`` or any required model key is missing.
    """
    if "model" not in config:
        raise KeyError('Missing required config section: "model"')

    model_config = config["model"]
    missing_keys = [key for key in _MODEL_KEYS if key not in model_config]
    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise KeyError(
            f"Missing required model config keys: {missing_list}. "
            f"Expected keys: {', '.join(_MODEL_KEYS)}"
        )

    return EEGTransformer(
        n_channels=model_config["n_channels"],
        sequence_length=model_config["sequence_length"],
        patch_size=model_config["patch_size"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        n_classes=model_config["n_classes"],
        dropout=model_config["dropout"],
    )
