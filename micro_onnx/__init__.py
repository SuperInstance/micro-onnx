"""
micro-onnx: ONNX export + benchmark pipeline for micro-models.

186× speedup over PyTorch CPU for small models.
"""

from .export import export_model, ExportResult
from .validate import validate_export, ValidationResult
from .benchmark import benchmark_model, BenchmarkResult
from .optimize import optimize_model, OptimizeResult

__version__ = "0.1.0"
__all__ = [
    "export_model",
    "ExportResult",
    "validate_export",
    "ValidationResult",
    "benchmark_model",
    "BenchmarkResult",
    "optimize_model",
    "OptimizeResult",
]
