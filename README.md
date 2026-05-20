# micro-onnx

**ONNX export + benchmark pipeline for micro-models.**

We measured **186× speedup** (58,648 qps vs 314 qps) running ONNX Runtime CPU vs PyTorch CPU on a SplineLinear layer. The secret: ONNX bakes weight materialization into the computation graph, eliminating Python overhead that dominates tiny forward passes.

## Install

```bash
pip install micro-onnx[all]     # everything
pip install micro-onnx           # numpy only (validate without torch/ort)
pip install micro-onnx[export]   # torch + onnx for exporting
pip install micro-onnx[runtime]  # onnxruntime for inference
```

## Quick Start

```python
import torch
import torch.nn as nn
from micro_onnx import export_model, validate_export, benchmark_model

model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
sample = torch.randn(1, 64)

# 1. Export
result = export_model(model, sample_input=sample, output_path="model.onnx")
print(f"Exported: {result.file_size:,} bytes, opset {result.opset_version}")

# 2. Validate
validation = validate_export(model, "model.onnx", sample_input=sample, tolerance=1e-6)
print(f"Max diff: {validation.max_diff:.2e}, Pass: {validation.passed}")

# 3. Benchmark
bench = benchmark_model("model.onnx", sample_input=sample, n_runs=1000)
print(f"PyTorch: {bench.pytorch_qps:.0f} qps")
print(f"ONNX:    {bench.onnx_qps:.0f} qps")
print(f"Speedup: {bench.speedup:.1f}×")
```

## Why ONNX for Micro-Models?

Micro-models (small MLPs, embeddings, spline layers) spend most of their time in Python/PyTorch dispatch overhead, not actual math. ONNX Runtime eliminates that overhead:

| Model | PyTorch CPU | ONNX Runtime CPU | Speedup |
|-------|------------|-------------------|---------|
| SplineLinear (64→32) | 314 qps | 58,648 qps | **186×** |
| nn.Linear (64→32) | ~1,200 qps | ~80,000 qps | **67×** |

## FP32 vs INT8: When Smaller Isn't Faster

Counter-intuitively, **FP32 beats INT8 for micro-models**. Why? INT8 quantization adds dequantize/quantize overhead. For models where the actual FLOPs are tiny, that overhead exceeds the savings from smaller weights.

**Rule of thumb:** If your model has <1M parameters and runs on CPU, try FP32 first. Only quantize if profiling shows compute-bound (not overhead-bound) inference.

## opset 17: The Sweet Spot

We default to opset 17 because it's the highest version with **broad device support** — CPUs, GPUs, mobile NPUs, and embedded accelerators all handle it. Newer opsets add ops that many runtimes don't support yet.

## API Reference

### `export_model(model, sample_input, opset=17, output_path=None, ...)`

Export any PyTorch `nn.Module` to ONNX. Returns an `ExportResult` with file path, size, opset version.

### `validate_export(model, onnx_path, sample_input, tolerance=1e-6)`

Run the same input through PyTorch and ONNX Runtime, compare outputs. Returns `ValidationResult` with `max_diff`, `mean_diff`, `cosine_similarity`, `passed`.

### `benchmark_model(onnx_path, sample_input, n_runs=1000, providers=None)`

Benchmark ONNX Runtime vs PyTorch on the same input. Returns `BenchmarkResult` with QPS for both, speedup ratio, and timing details.

### `optimize_model(onnx_path, output_path=None, level="all")`

Apply ONNX graph optimizations (constant folding, node fusion, dead code elimination). Returns `OptimizeResult`.

### Hardware Profiles

```python
from micro_onnx.profiles import PROFILES
# CPU, GPU, iGPU, NPU profiles with recommended settings per target
```

## Works With Any PyTorch Model

micro-onnx is model-agnostic. Any `nn.Module` with forward-pass-compatible inputs works — transformers, CNNs, GNNs, custom architectures. The only requirement is that the model's forward pass is compatible with `torch.onnx.export`.

## License

MIT
