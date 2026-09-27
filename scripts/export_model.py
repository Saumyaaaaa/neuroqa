"""CLI script to export EEGTransformer to ONNX and run benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without setting PYTHONPATH manually
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from rich.console import Console
from rich.table import Table

from neuroqa.models import create_model
from neuroqa.models.exporter import benchmark_onnx, export_to_onnx


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export NeuroQA EEGTransformer to ONNX format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt model checkpoint. If omitted, uses random weights.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/model.onnx",
        help="Output path for the .onnx file. Default: checkpoints/model.onnx",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run CPU inference benchmark after export.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of benchmark inference runs. Default: 100",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the ONNX export CLI."""
    args = parse_args()
    console = Console()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build model
    console.print("[cyan]Building EEGTransformer...[/cyan]")
    model = create_model(config)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
            sys.exit(1)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu")
        )
        console.print(f"[green]Loaded checkpoint: {args.checkpoint}[/green]")
    else:
        console.print("[yellow]No checkpoint provided — using random weights.[/yellow]")

    # Export
    console.print(f"[cyan]Exporting to ONNX: {args.output}[/cyan]")
    try:
        result = export_to_onnx(
            model=model,
            output_path=args.output,
            n_channels=config["data"]["n_channels"]
            if "n_channels" in config.get("data", {})
            else 19,
            sequence_length=config["model"].get("sequence_length", 512),
            opset_version=17,
        )
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/red]")
        sys.exit(1)

    # Print export results table
    export_table = Table(title="NeuroQA ONNX Export Results", show_header=False)
    export_table.add_column("Field", style="bold cyan", min_width=22)
    export_table.add_column("Value", style="white")

    match_symbol = "[green]✓ YES[/green]" if result["pytorch_onnx_match"] else "[red]✗ NO[/red]"

    export_table.add_row("Output path", result["output_path"])
    export_table.add_row("Model size", f"{result['model_size_mb']} MB")
    export_table.add_row("PyTorch match", match_symbol)
    export_table.add_row("Output shape", str(result["onnx_output_shape"]))
    export_table.add_row("Opset version", str(result["opset_version"]))

    console.print(export_table)

    # Benchmark
    if args.benchmark:
        console.print(f"\n[cyan]Running CPU benchmark ({args.n_runs} runs)...[/cyan]")
        try:
            bench = benchmark_onnx(
                onnx_path=args.output,
                n_channels=19,
                sequence_length=512,
                n_runs=args.n_runs,
            )
        except Exception as exc:
            console.print(f"[red]Benchmark failed: {exc}[/red]")
            sys.exit(1)

        bench_table = Table(
            title=f"CPU Inference Benchmark ({args.n_runs} runs)",
            show_header=False,
        )
        bench_table.add_column("Metric", style="bold cyan", min_width=22)
        bench_table.add_column("Value", style="white")

        bench_table.add_row("Mean latency", f"{bench['mean_latency_ms']} ms")
        bench_table.add_row("Std deviation", f"{bench['std_latency_ms']} ms")
        bench_table.add_row(
            "Min / Max",
            f"{bench['min_latency_ms']} / {bench['max_latency_ms']} ms",
        )
        bench_table.add_row("Throughput", f"{bench['throughput_per_second']} /sec")

        console.print(bench_table)

    console.print("\n[green]Export complete.[/green]")


if __name__ == "__main__":
    main()