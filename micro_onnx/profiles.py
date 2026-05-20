"""
Hardware profiles for different deployment targets.

Each profile specifies recommended ONNX settings for a target device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HardwareProfile:
    """Recommended ONNX settings for a hardware target."""
    name: str
    description: str
    providers: List[str]
    opset: int
    recommended_precision: str  # "fp32", "fp16", "int8"
    use_quantization: bool
    notes: str = ""


PROFILES: Dict[str, HardwareProfile] = {
    "cpu": HardwareProfile(
        name="cpu",
        description="General CPU (x86/ARM)",
        providers=["CPUExecutionProvider"],
        opset=17,
        recommended_precision="fp32",
        use_quantization=False,
        notes=(
            "FP32 is optimal for micro-models on CPU. INT8 dequantize overhead "
            "exceeds compute savings at small dimensions. Graph optimization "
            "(constant folding, node fusion) provides the real speedup."
        ),
    ),
    "gpu": HardwareProfile(
        name="gpu",
        description="NVIDIA/AMD GPU via CUDA",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        opset=17,
        recommended_precision="fp16",
        use_quantization=False,
        notes=(
            "FP16 is a good default for GPU. For micro-models, CPU may actually "
            "be faster due to PCIe transfer overhead — benchmark both."
        ),
    ),
    "igpu": HardwareProfile(
        name="igpu",
        description="Integrated GPU (Intel/AMD)",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        opset=17,
        recommended_precision="fp32",
        use_quantization=False,
        notes=(
            "Integrated GPUs share memory with CPU. FP32 is safe. "
            "DmlExecutionProvider for Windows, OpenVINO for Intel iGPU on Linux."
        ),
    ),
    "npu": HardwareProfile(
        name="npu",
        description="Neural Processing Unit (mobile/embedded)",
        providers=["NNAPIExecutionProvider", "CPUExecutionProvider"],
        opset=17,
        recommended_precision="fp32",
        use_quantization=True,
        notes=(
            "NPUs benefit from INT8 quantization when models are large enough "
            "(>1M params). For micro-models, test both FP32 and INT8 — "
            "the NPU may not be faster than CPU for tiny graphs."
        ),
    ),
    "openvino": HardwareProfile(
        name="openvino",
        description="Intel OpenVINO (CPU/iGPU/NPU)",
        providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        opset=17,
        recommended_precision="fp32",
        use_quantization=False,
        notes="OpenVINO provides optimized kernels for Intel hardware.",
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Get a hardware profile by name."""
    name = name.lower()
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> Dict[str, HardwareProfile]:
    """Return all available hardware profiles."""
    return dict(PROFILES)
