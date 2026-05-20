"""
Benchmark PyTorch vs ONNX Runtime inference performance.

Measures queries-per-second (QPS) for both runtimes and reports speedup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a PyTorch vs ONNX benchmark."""
    pytorch_qps: float
    onnx_qps: float
    speedup: float
    n_runs: int
    pytorch_mean_ms: float
    onnx_mean_ms: float
    pytorch_total_s: float
    onnx_total_s: float
    providers: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"BenchmarkResult(PyTorch={self.pytorch_qps:.0f} qps, "
            f"ONNX={self.onnx_qps:.0f} qps, Speedup={self.speedup:.1f}×)"
        )


def benchmark_model(
    onnx_path: str,
    sample_input: Any = None,
    model: Any = None,
    n_runs: int = 1000,
    warmup: int = 50,
    providers: Optional[List[str]] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark ONNX Runtime vs PyTorch inference.

    Provide either `sample_input` (tensor/numpy) for benchmarking,
    or both `model` and `sample_input` for a head-to-head comparison.

    Args:
        onnx_path:    Path to the ONNX model file.
        sample_input: Example input (torch.Tensor or numpy array).
        model:        PyTorch model for comparison (optional).
        n_runs:       Number of inference iterations (default 1000).
        warmup:       Warmup iterations before timing (default 50).
        providers:    ONNX Runtime execution providers (default: CPUExecutionProvider).
        verbose:      Print benchmark results.

    Returns:
        BenchmarkResult with QPS, timing, and speedup.

    Raises:
        ImportError: If onnxruntime (or torch for head-to-head) is not installed.
        FileNotFoundError: If onnx_path doesn't exist.
    """
    import os
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required: pip install onnxruntime")

    if providers is None:
        providers = ["CPUExecutionProvider"]

    # Prepare numpy input
    np_input = _to_numpy(sample_input)

    # --- ONNX Runtime benchmark ---
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_meta = session.get_inputs()[0]
    feed = {input_meta.name: _cast_input(np_input, input_meta.type)}

    # Warmup
    for _ in range(warmup):
        session.run(None, feed)

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, feed)
    onnx_total = time.perf_counter() - start

    onnx_mean_ms = (onnx_total / n_runs) * 1000
    onnx_qps = n_runs / onnx_total

    # --- PyTorch benchmark (optional) ---
    pytorch_qps = 0.0
    pytorch_total = 0.0
    pytorch_mean_ms = 0.0

    if model is not None:
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for PyTorch benchmark: pip install torch")

        torch_input = sample_input if isinstance(sample_input, torch.Tensor) else torch.from_numpy(np_input)

        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                model(torch_input)
                torch.cpu.synchronize() if hasattr(torch, 'cpu') else None

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    model(torch_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pytorch_total = time.perf_counter() - start

        pytorch_mean_ms = (pytorch_total / n_runs) * 1000
        pytorch_qps = n_runs / pytorch_total

    speedup = onnx_qps / pytorch_qps if pytorch_qps > 0 else 0.0

    result = BenchmarkResult(
        pytorch_qps=round(pytorch_qps, 2),
        onnx_qps=round(onnx_qps, 2),
        speedup=round(speedup, 2),
        n_runs=n_runs,
        pytorch_mean_ms=round(pytorch_mean_ms, 4),
        onnx_mean_ms=round(onnx_mean_ms, 4),
        pytorch_total_s=round(pytorch_total, 4),
        onnx_total_s=round(onnx_total, 4),
        providers=providers,
    )

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  micro-onnx Benchmark ({n_runs} runs)")
        print(f"{'=' * 50}")
        if model is not None:
            print(f"  PyTorch:  {pytorch_qps:>10,.0f} qps  ({pytorch_mean_ms:.3f} ms)")
        print(f"  ONNX RT:  {onnx_qps:>10,.0f} qps  ({onnx_mean_ms:.3f} ms)")
        if model is not None:
            print(f"  Speedup:  {speedup:>10.1f}×")
        print(f"{'=' * 50}\n")

    return result


def _to_numpy(x: Any) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _cast_input(arr: np.ndarray, onnx_type: str) -> np.ndarray:
    """Cast numpy array to match ONNX input type."""
    type_str = str(onnx_type).lower()
    if "int64" in type_str:
        return arr.astype(np.int64)
    elif "float16" in type_str:
        return arr.astype(np.float16)
    elif "float64" in type_str:
        return arr.astype(np.float64)
    return arr.astype(np.float32)
