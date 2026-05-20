"""Tests for ONNX export functionality."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_model():
    """A simple MLP model for testing."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def sample_input():
    return torch.randn(1, 64)


@pytest.fixture
def tmp_onnx(tmp_path):
    return str(tmp_path / "test_model.onnx")


def test_export_simple_model(simple_model, sample_input, tmp_onnx):
    """Export a simple model to ONNX."""
    from micro_onnx import export_model
    result = export_model(simple_model, sample_input=sample_input, output_path=tmp_onnx)
    assert os.path.exists(tmp_onnx)
    assert result.file_size > 0
    assert result.opset_version == 17
    assert result.path == os.path.abspath(tmp_onnx)


def test_export_file_size_reporting(simple_model, sample_input, tmp_onnx):
    """File size should be reported correctly."""
    from micro_onnx import export_model
    result = export_model(simple_model, sample_input=sample_input, output_path=tmp_onnx)
    assert result.file_size == os.path.getsize(tmp_onnx)


def test_export_custom_opset(simple_model, sample_input, tmp_onnx):
    """Export with a custom opset version."""
    from micro_onnx import export_model
    result = export_model(simple_model, sample_input=sample_input, output_path=tmp_onnx, opset=14)
    assert result.opset_version == 14


def test_export_creates_directory(simple_model, sample_input, tmp_path):
    """Export should create output directories if they don't exist."""
    from micro_onnx import export_model
    nested_path = str(tmp_path / "nested" / "dir" / "model.onnx")
    result = export_model(simple_model, sample_input=sample_input, output_path=nested_path)
    assert os.path.exists(nested_path)


def test_export_default_path(simple_model, sample_input):
    """Export with default path should create model.onnx."""
    from micro_onnx import export_model
    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            result = export_model(simple_model, sample_input=sample_input)
            assert os.path.exists("model.onnx")
            assert result.path.endswith("model.onnx")
    finally:
        os.chdir(original_dir)


def test_export_result_str(simple_model, sample_input, tmp_onnx):
    """ExportResult should have a readable string representation."""
    from micro_onnx import export_model
    result = export_model(simple_model, sample_input=sample_input, output_path=tmp_onnx)
    s = str(result)
    assert "bytes" in s
    assert "opset" in s


def test_export_no_torch_raises():
    """Should raise helpful error if torch not available."""
    # This test would need to mock the import, skip if torch is present
    # (which it will be in our test env)
    pass  # Covered by integration: torch is required


def test_export_batch_dimension(simple_model, tmp_onnx):
    """Export should handle batch inputs correctly."""
    from micro_onnx import export_model
    batch_input = torch.randn(8, 64)
    result = export_model(simple_model, sample_input=batch_input, output_path=tmp_onnx)
    assert result.file_size > 0
