"""
Export PyTorch models to ONNX with validation and sensible defaults.

Uses opset 17 for broad device support (CPU, GPU, mobile NPU, embedded).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_OPSET_DEFAULT = 17


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for export. Install with: pip install torch"
        )


def _check_onnx():
    try:
        import onnx
        return onnx
    except ImportError:
        raise ImportError(
            "onnx is required for export validation. Install with: pip install onnx"
        )


@dataclass
class ExportResult:
    """Result of an ONNX export operation."""
    path: str
    file_size: int
    opset_version: int
    input_names: List[str]
    output_names: List[str]
    dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=dict)
    model_hash: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"ExportResult(path={self.path!r}, size={self.file_size:,} bytes, "
            f"opset={self.opset_version})"
        )


def export_model(
    model: Any,
    sample_input: Any,
    opset: int = _OPSET_DEFAULT,
    output_path: Optional[str] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    do_constant_folding: bool = True,
    validate_graph: bool = True,
    verbose: bool = False,
) -> ExportResult:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model:              Any torch.nn.Module.
        sample_input:       Example input tensor (or tuple/dict of tensors).
        opset:              ONNX opset version (default 17).
        output_path:        Where to save the .onnx file. Defaults to "model.onnx".
        input_names:        Names for inputs in the ONNX graph.
        output_names:       Names for outputs in the ONNX graph.
        dynamic_axes:       Dynamic axes specification.
        do_constant_folding: Whether to fold constants (default True).
        validate_graph:     Whether to validate the exported ONNX graph.
        verbose:            Print detailed export info.

    Returns:
        ExportResult with path, size, opset, and metadata.

    Raises:
        ImportError:  If torch or onnx is not installed.
        RuntimeError: If export or validation fails.
    """
    torch = _check_torch()
    onnx = _check_onnx()

    if output_path is None:
        output_path = "model.onnx"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model.eval()

    # Default names
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Default dynamic axes: batch dimension is dynamic
    if dynamic_axes is None:
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    try:
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            verbose=verbose,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e

    # Validate the exported graph
    if validate_graph:
        try:
            loaded = onnx.load(output_path)
            onnx.checker.check_model(loaded)
        except Exception as e:
            raise RuntimeError(f"Exported ONNX graph is invalid: {e}") from e

    file_size = os.path.getsize(output_path)
    loaded = onnx.load(output_path)

    result = ExportResult(
        path=os.path.abspath(output_path),
        file_size=file_size,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    if verbose:
        print(f"[micro-onnx] Exported → {result}")

    return result
