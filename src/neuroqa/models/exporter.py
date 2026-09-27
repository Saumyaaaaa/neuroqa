"""ONNX export and benchmarking utilities for NeuroQA."""

from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.attention

# Ensure torch.nn.attention.sdp_kernel exists across PyTorch versions
if not hasattr(torch.nn.attention, "sdp_kernel") and hasattr(
    torch.backends.cuda, "sdp_kernel"
):
    torch.nn.attention.sdp_kernel = torch.backends.cuda.sdp_kernel

from neuroqa.models.transformer import EEGTransformer

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: EEGTransformer,
    output_path: str,
    n_channels: int = 19,
    window_samples: int = 512,
    opset_version: int = 14,
) -> dict[str, object]:
    """Export an EEGTransformer model to ONNX format with validation.

    Args:
        model: Trained or untrained EEGTransformer instance.
        output_path: File path where the .onnx file will be saved.
        n_channels: Number of EEG channels. Defaults to 19.
        window_samples: Number of time samples per window. Defaults to 512.
        opset_version: ONNX opset version. Defaults to 14.

    Returns:
        Dictionary with keys: output_path, onnx_output_shape,
        pytorch_onnx_match, model_size_mb, opset_version.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    dummy_input = torch.randn(1, n_channels, window_samples)

    logger.info(
        "Exporting EEGTransformer to ONNX: %s (opset %d)",
        output_path,
        opset_version,
    )

    with torch.no_grad():
        pytorch_output = model(dummy_input)

    if not hasattr(torch.nn.attention, "sdp_kernel") and hasattr(
        torch.backends.cuda, "sdp_kernel"
    ):
        torch.nn.attention.sdp_kernel = torch.backends.cuda.sdp_kernel

    extra_export_args: dict[str, object] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        extra_export_args["dynamo"] = False

    with torch.nn.attention.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    ):
        torch.onnx.export(
            model,
            dummy_input,
            str(output_file),
            input_names=["eeg_input"],
            output_names=["logits"],
            dynamic_axes={
                "eeg_input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            **extra_export_args,
        )

    logger.info("ONNX file written. Validating graph...")

    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph validation passed.")

    session = ort.InferenceSession(
        str(output_file),
        providers=["CPUExecutionProvider"],
    )

    onnx_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = session.run(None, onnx_input)[0]

    pytorch_np = pytorch_output.detach().numpy()
    match = bool(np.allclose(pytorch_np, onnx_output, atol=1e-4))

    if not match:
        max_diff = float(np.max(np.abs(pytorch_np - onnx_output)))
        logger.warning(
            "PyTorch and ONNX outputs differ. Max diff: %.6f", max_diff
        )
    else:
        logger.info("PyTorch and ONNX outputs match within atol=1e-4.")

    model_size_mb = output_file.stat().st_size / (1024 * 1024)

    return {
        "output_path": str(output_file),
        "onnx_output_shape": tuple(onnx_output.shape),
        "pytorch_onnx_match": match,
        "model_size_mb": round(model_size_mb, 3),
        "opset_version": opset_version,
    }


def benchmark_onnx(
    onnx_path: str,
    n_channels: int = 19,
    window_samples: int = 512,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark ONNX model inference speed on CPU.

    Args:
        onnx_path: Path to the .onnx file.
        n_channels: Number of EEG input channels. Defaults to 19.
        window_samples: Samples per window. Defaults to 512.
        n_runs: Number of timed inference passes. Defaults to 100.

    Returns:
        Dictionary with latency statistics and throughput.
    """
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, n_channels, window_samples).astype(np.float32)
    feed = {input_name: dummy_input}

    for _ in range(5):
        session.run(None, feed)

    latencies: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, feed)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    latencies_np = np.array(latencies)
    mean_ms = float(np.mean(latencies_np))

    logger.info(
        "Benchmark: mean=%.2f ms, std=%.2f ms over %d runs",
        mean_ms,
        float(np.std(latencies_np)),
        n_runs,
    )

    return {
        "mean_latency_ms": round(mean_ms, 3),
        "std_latency_ms": round(float(np.std(latencies_np)), 3),
        "min_latency_ms": round(float(np.min(latencies_np)), 3),
        "max_latency_ms": round(float(np.max(latencies_np)), 3),
        "throughput_per_second": round(1000.0 / mean_ms, 2),
    }