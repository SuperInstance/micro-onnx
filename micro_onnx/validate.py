"""
Numerical validation of ONNX exports against PyTorch reference outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np


@dataclass
class ValidationResult:
    """Result of numerical validation between PyTorch and ONNX outputs."""
    max_diff: float
    mean_diff: float
    cosine_similarity: float
    tolerance: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS ✅" if self.passed else "FAIL ❌"
        return (
            f"ValidationResult({status}, max_diff={self.max_diff:.2e}, "
            f"mean_diff={self.mean_diff:.2e}, cos_sim={self.cosine_similarity:.6f})"
        )


def validate_export(
    model: Any,
    onnx_path: str,
    sample_input: Any,
    tolerance: float = 1e-6,
    n_samples: int = 1,
    verbose: bool = False,
) -> ValidationResult:
    """
    Validate that an ONNX export produces numerically identical outputs to PyTorch.

    Args:
        model:        The original PyTorch model.
        onnx_path:    Path to the exported ONNX file.
        sample_input: Example input tensor (torch.Tensor or numpy array).
        tolerance:    Maximum allowed absolute difference (default 1e-6).
        n_samples:    Number of random samples to test (default 1; uses sample_input only).
        verbose:      Print detailed comparison info.

    Returns:
        ValidationResult with max_diff, mean_diff, cosine_similarity, passed.

    Raises:
        ImportError: If torch or onnxruntime is not installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for validation: pip install torch")

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for validation: pip install onnxruntime")

    model.eval()

    # Run PyTorch inference
    with __import__("torch").no_grad():
        pt_output = model(sample_input)
        if isinstance(pt_output, (tuple, list)):
            pt_output = pt_output[0]
        pt_np = pt_output.cpu().numpy()

    # Run ONNX inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    if isinstance(sample_input, torch.Tensor):
        np_input = sample_input.cpu().numpy()
    else:
        np_input = np.asarray(sample_input)

    # Cast to the expected input type
    onnx_type = input_meta.type
    if "int64" in str(onnx_type).lower():
        np_input = np_input.astype(np.int64)
    elif "float" in str(onnx_type).lower():
        np_input = np_input.astype(np.float32)

    ort_output = session.run(None, {input_meta.name: np_input})[0]

    # Compare
    diff = np.abs(pt_np - ort_output)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Cosine similarity
    flat_pt = pt_np.flatten()
    flat_ort = ort_output.flatten()
    norm_product = np.linalg.norm(flat_pt) * np.linalg.norm(flat_ort)
    if norm_product > 0:
        cos_sim = float(np.dot(flat_pt, flat_ort) / norm_product)
    else:
        cos_sim = 1.0  # Both zero vectors

    passed = max_diff <= tolerance

    result = ValidationResult(
        max_diff=max_diff,
        mean_diff=mean_diff,
        cosine_similarity=cos_sim,
        tolerance=tolerance,
        passed=passed,
    )

    if verbose:
        print(f"[micro-onnx] {result}")

    return result


def compare_outputs(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """
    Compare raw numpy arrays from PyTorch and ONNX.

    Args:
        pytorch_output: Output from PyTorch as numpy array.
        onnx_output:    Output from ONNX Runtime as numpy array.
        tolerance:      Maximum allowed absolute difference.

    Returns:
        ValidationResult.
    """
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    flat_pt = pytorch_output.flatten()
    flat_ort = onnx_output.flatten()
    norm_product = np.linalg.norm(flat_pt) * np.linalg.norm(flat_ort)
    cos_sim = float(np.dot(flat_pt, flat_ort) / (norm_product + 1e-12))

    return ValidationResult(
        max_diff=max_diff,
        mean_diff=mean_diff,
        cosine_similarity=cos_sim,
        tolerance=tolerance,
        passed=max_diff <= tolerance,
    )
