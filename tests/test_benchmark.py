"""Tests for ONNX vs PyTorch benchmarking."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def exported_model():
    """Create and export a simple model."""
    from micro_onnx import export_model
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    sample = torch.randn(1, 32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    export_model(model, sample_input=sample, output_path=path)
    yield model, sample, path
    os.unlink(path)


def test_benchmark_onnx_only(exported_model):
    """Benchmark ONNX Runtime without PyTorch comparison."""
    from micro_onnx import benchmark_model
    _, sample, onnx_path = exported_model
    result = benchmark_model(onnx_path, sample_input=sample, model=None, n_runs=100, verbose=False)
    assert result.onnx_qps > 0
    assert result.onnx_mean_ms > 0
    assert result.n_runs == 100


def test_benchmark_head_to_head(exported_model):
    """Benchmark ONNX vs PyTorch head-to-head."""
    from micro_onnx import benchmark_model
    model, sample, onnx_path = exported_model
    result = benchmark_model(onnx_path, sample_input=sample, model=model, n_runs=100, verbose=False)
    assert result.pytorch_qps > 0
    assert result.onnx_qps > 0
    assert result.speedup > 0


def test_benchmark_result_str(exported_model):
    """BenchmarkResult should have readable string."""
    from micro_onnx import benchmark_model
    _, sample, onnx_path = exported_model
    result = benchmark_model(onnx_path, sample_input=sample, n_runs=50, verbose=False)
    s = str(result)
    assert "qps" in s


def test_benchmark_cpu_provider(exported_model):
    """CPUExecutionProvider should always be available."""
    from micro_onnx import benchmark_model
    _, sample, onnx_path = exported_model
    result = benchmark_model(
        onnx_path, sample_input=sample, n_runs=50,
        providers=["CPUExecutionProvider"], verbose=False,
    )
    assert "CPUExecutionProvider" in result.providers


def test_benchmark_file_not_found():
    """Should raise FileNotFoundError for missing files."""
    from micro_onnx import benchmark_model
    with pytest.raises(FileNotFoundError):
        benchmark_model("/nonexistent/model.onnx", n_runs=10, verbose=False)


def test_benchmark_numpy_input(exported_model):
    """Benchmark should accept numpy inputs."""
    from micro_onnx import benchmark_model
    _, _, onnx_path = exported_model
    np_input = np.random.randn(1, 32).astype(np.float32)
    result = benchmark_model(onnx_path, sample_input=np_input, n_runs=50, verbose=False)
    assert result.onnx_qps > 0
