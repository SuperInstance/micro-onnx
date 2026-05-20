"""Tests for ONNX optimization and profiles."""

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
    yield path
    os.unlink(path)


def test_optimize_basic(exported_model):
    """Optimization should produce a valid model."""
    from micro_onnx import optimize_model
    result = optimize_model(exported_model)
    assert os.path.exists(result.path)
    assert result.original_size > 0
    assert result.optimized_size > 0


def test_optimize_to_new_path(exported_model, tmp_path):
    """Optimization should support writing to a new path."""
    from micro_onnx import optimize_model
    out_path = str(tmp_path / "optimized.onnx")
    result = optimize_model(exported_model, output_path=out_path)
    assert os.path.exists(out_path)
    assert result.path == os.path.abspath(out_path)


def test_optimize_result_str(exported_model):
    """OptimizeResult should have readable string."""
    from micro_onnx import optimize_model
    result = optimize_model(exported_model)
    s = str(result)
    assert "bytes" in s


def test_profiles_available():
    """All expected profiles should be available."""
    from micro_onnx.profiles import PROFILES, get_profile
    assert "cpu" in PROFILES
    assert "gpu" in PROFILES
    assert "npu" in PROFILES
    assert PROFILES["cpu"].opset == 17
    assert not PROFILES["cpu"].use_quantization


def test_get_profile():
    """get_profile should return the right profile."""
    from micro_onnx.profiles import get_profile
    cpu = get_profile("cpu")
    assert cpu.name == "cpu"
    assert "CPUExecutionProvider" in cpu.providers


def test_get_profile_unknown():
    """get_profile should raise for unknown profiles."""
    from micro_onnx.profiles import get_profile
    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile("tpu")


def test_list_profiles():
    """list_profiles should return all profiles."""
    from micro_onnx.profiles import list_profiles
    profiles = list_profiles()
    assert len(profiles) >= 4
    assert "cpu" in profiles
    assert "gpu" in profiles


def test_mesh_registration():
    """Test that Mesh-style registration works."""
    def register_micro_onnx(registry):
        from micro_onnx import export_model, benchmark_model
        registry.register("compressors", "onnx-export", lambda: export_model)
        registry.register("devices", "onnx-runtime", lambda: benchmark_model)

    class FakeRegistry:
        def __init__(self):
            self.entries = {}
        def register(self, category, name, factory):
            self.entries[(category, name)] = factory

    reg = FakeRegistry()
    register_micro_onnx(reg)
    assert ("compressors", "onnx-export") in reg.entries
    assert ("devices", "onnx-runtime") in reg.entries
    assert reg.entries[("compressors", "onnx-export")]() is not None
