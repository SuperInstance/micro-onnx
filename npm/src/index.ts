// @superinstance/micro-onnx
// ONNX export + benchmark pipeline for micro-models

/** Recommended ONNX opset version for broad runtime compatibility */
export const RECOMMENDED_OPSET = 17;

/**
 * Documents the finding that FP36-style mixed-precision was the first
 * optimization that unlocked significant speedups in the micro-model pipeline.
 */
export const FP36_FIRST = true;

// ── Types & Classes ──────────────────────────────────────────────

export class ExportResult {
  path: string;
  fileSize: number;
  opsetVersion: number;
  success: boolean;

  constructor(opts: { path: string; fileSize: number; opsetVersion: number; success: boolean }) {
    this.path = opts.path;
    this.fileSize = opts.fileSize;
    this.opsetVersion = opts.opsetVersion;
    this.success = opts.success;
  }
}

export class ValidationResult {
  maxDiff: number;
  passed: boolean;
  tolerance: number;

  constructor(opts: { maxDiff: number; passed: boolean; tolerance: number }) {
    this.maxDiff = opts.maxDiff;
    this.passed = opts.passed;
    this.tolerance = opts.tolerance;
  }
}

export class BenchmarkResult {
  pytorchQps: number;
  onnxQps: number;
  speedup: number;
  provider: string;

  constructor(opts: { pytorchQps: number; onnxQps: number; speedup: number; provider: string }) {
    this.pytorchQps = opts.pytorchQps;
    this.onnxQps = opts.onnxQps;
    this.speedup = opts.speedup;
    this.provider = opts.provider;
  }
}

export class OptimizeResult {
  originalSize: number;
  optimizedSize: number;
  passes: string[];

  constructor(opts: { originalSize: number; optimizedSize: number; passes: string[] }) {
    this.originalSize = opts.originalSize;
    this.optimizedSize = opts.optimizedSize;
    this.passes = opts.passes;
  }
}

// ── Hardware Profiles ────────────────────────────────────────────

export type HardwareProfile =
  | { kind: "CPU"; cores: number; arch: string }
  | { kind: "GPU"; vram: number; model: string }
  | { kind: "iGPU"; vram: number; model: string }
  | { kind: "NPU"; tops: number; model: string };

// ── Functions (descriptive — actual export needs Python runtime) ──

/**
 * Export a PyTorch model to ONNX format.
 *
 * In a real pipeline this shells out to Python:
 *   torch.onnx.export(model, dummy_input, path, opset_version=17)
 *
 * Returns an ExportResult describing the outcome.
 */
export function exportModel(
  modelPath: string,
  opts?: { opsetVersion?: number; outputPath?: string }
): ExportResult {
  const opset = opts?.opsetVersion ?? RECOMMENDED_OPSET;
  const outPath = opts?.outputPath ?? modelPath.replace(/\.pt$/, ".onnx");
  return new ExportResult({
    path: outPath,
    fileSize: 0,
    opsetVersion: opset,
    success: false, // JS can't run PyTorch — use the Python CLI
  });
}

/**
 * Benchmark a model comparing PyTorch vs ONNX Runtime throughput.
 *
 * In a real pipeline this runs warmup + timed inference loops via
 * onnxruntime InferenceSession.
 *
 * Returns a BenchmarkResult describing the expected comparison.
 */
export function benchmarkModel(
  onnxPath: string,
  opts?: { provider?: string; warmupRuns?: number; benchmarkRuns?: number }
): BenchmarkResult {
  const provider = opts?.provider ?? "CPU";
  return new BenchmarkResult({
    pytorchQps: 0,
    onnxQps: 0,
    speedup: 0,
    provider,
  });
}
