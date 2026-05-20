"""Tests for numerical validation."""

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


def test_validate_export_passes(exported_model):
    """Validation should pass for a correctly exported model."""
    from micro_onnx import validate_export
    model, sample, onnx_path = exported_model
    result = validate_export(model, onnx_path, sample_input=sample, tolerance=1e-5)
    assert result.passed
    assert result.max_diff < 1e-5
    assert result.cosine_similarity > 0.99


def test_validate_tight_tolerance(exported_model):
    """Very tight tolerance may fail due to floating point."""
    from micro_onnx import validate_export
    model, sample, onnx_path = exported_model
    result = validate_export(model, onnx_path, sample_input=sample, tolerance=1e-15)
    # Likely fails due to FP differences
    # Just check the result is properly computed
    assert isinstance(result.passed, bool)
    assert result.max_diff >= 0


def test_validate_result_str(exported_model):
    """ValidationResult should have readable string."""
    from micro_onnx import validate_export
    model, sample, onnx_path = exported_model
    result = validate_export(model, onnx_path, sample_input=sample)
    s = str(result)
    assert "PASS" in s or "FAIL" in s


def test_compare_outputs():
    """Test raw array comparison utility."""
    from micro_onnx.validate import compare_outputs
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0, 2.0, 3.0]])
    result = compare_outputs(a, b)
    assert result.passed
    assert result.max_diff == 0.0
    assert result.cosine_similarity > 0.9999


def test_compare_outputs_different():
    """Compare slightly different arrays."""
    from micro_onnx.validate import compare_outputs
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0, 2.001, 3.0]])
    result = compare_outputs(a, b, tolerance=0.01)
    assert result.passed
    assert abs(result.max_diff - 0.001) < 1e-10


def test_validate_batch_input(exported_model):
    """Validation should work with batch inputs."""
    from micro_onnx import validate_export
    model, _, onnx_path = exported_model
    batch = torch.randn(8, 32)
    result = validate_export(model, onnx_path, sample_input=batch)
    assert isinstance(result.passed, bool)
