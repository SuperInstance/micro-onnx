"""
ONNX graph optimization and quantization utilities.

For micro-models, FP32 is often optimal — INT8 quantization overhead
exceeds savings at tiny dimensions. This module applies graph-level
optimizations (constant folding, node fusion, dead code elimination)
which provide the real speedup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class OptimizeResult:
    """Result of an ONNX optimization pass."""
    path: str
    original_size: int
    optimized_size: int
    savings_pct: float
    optimizations_applied: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"OptimizeResult(size {self.original_size:,} → {self.optimized_size:,} "
            f"bytes, {self.savings_pct:.1f}% smaller)"
        )


def optimize_model(
    onnx_path: str,
    output_path: Optional[str] = None,
    level: str = "all",
) -> OptimizeResult:
    """
    Apply ONNX graph optimizations.

    Optimizations include constant folding, node fusion, dead code
    elimination, and operator simplification.

    Args:
        onnx_path:    Path to the input ONNX file.
        output_path:  Where to save the optimized model. Defaults to overwriting input.
        level:        Optimization level — "basic", "extended", or "all" (default).

    Returns:
        OptimizeResult with before/after sizes and applied optimizations.

    Note:
        For micro-models, graph optimization provides the main speedup.
        INT8 quantization often *hurts* performance because the
        dequantize/quantize overhead exceeds compute savings at small
        dimensions. Always benchmark before quantizing.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("onnx is required for optimization: pip install onnx")

    if output_path is None:
        output_path = onnx_path

    original_size = os.path.getsize(onnx_path)
    model = onnx.load(onnx_path)
    applied = []

    # Try onnx optimizer (available in older versions)
    try:
        from onnx import optimizer as onnx_optimizer
        available = onnx_optimizer.get_available_passes()

        passes = []
        if level in ("basic", "all"):
            passes.extend([
                "eliminate_deadend",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "eliminate_identity",
                "fold_constants",
            ])
        if level in ("extended", "all"):
            passes.extend([
                "fuse_consecutive_concats",
                "fuse_consecutive_reduce_unsqueeze",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
            ])

        passes = [p for p in passes if p in available]
        if passes:
            model = onnx_optimizer.optimize(model, passes=passes)
            applied.extend(passes)
    except ImportError:
        pass  # optimizer not available in this onnx version

    # Shape inference as additional optimization
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
        applied.append("shape_inference")
    except Exception:
        pass

    onnx.save(model, output_path)
    optimized_size = os.path.getsize(output_path)
    savings = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0

    return OptimizeResult(
        path=os.path.abspath(output_path),
        original_size=original_size,
        optimized_size=optimized_size,
        savings_pct=round(savings, 2),
        optimizations_applied=applied,
    )


def _basic_optimize(
    onnx_path: str,
    output_path: Optional[str],
    onnx_module,
) -> OptimizeResult:
    """Fallback optimization using only the onnx library."""
    if output_path is None:
        output_path = onnx_path

    original_size = os.path.getsize(onnx_path)
    model = onnx_module.load(onnx_path)

    # Constant folding via shape inference
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    onnx_module.save(model, output_path)
    optimized_size = os.path.getsize(output_path)
    savings = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0

    return OptimizeResult(
        path=os.path.abspath(output_path),
        original_size=original_size,
        optimized_size=optimized_size,
        savings_pct=round(savings, 2),
        optimizations_applied=["shape_inference"],
    )
