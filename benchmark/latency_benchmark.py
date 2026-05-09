from pathlib import Path
from time import perf_counter
import json

from core.logger import logger


class BenchmarkError(RuntimeError):
    """Raised when ONNX Runtime benchmarking cannot be completed."""


class OnnxLatencyBenchmark:
    """Benchmark ONNX models with ONNX Runtime inference sessions."""

    def __init__(
        self,
        warmup_iterations=5,
        benchmark_iterations=50,
        providers=None,
    ):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.providers = providers or ["CPUExecutionProvider"]

    def benchmark_model(self, model_path):
        model_path = Path(model_path)
        logger.info("Benchmarking ONNX model latency: %s", model_path)

        try:
            ort = self._import_onnxruntime()
            session = self._create_session(ort, model_path)
            inputs = self._create_dummy_inputs(session)

            for _ in range(self.warmup_iterations):
                session.run(None, inputs)

            latencies_ms = []
            for _ in range(self.benchmark_iterations):
                start = perf_counter()
                session.run(None, inputs)
                latencies_ms.append((perf_counter() - start) * 1000)

            avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
            min_latency_ms = min(latencies_ms)
            max_latency_ms = max(latencies_ms)
            batch_size = self._batch_size(session.get_inputs())
            throughput = batch_size / (avg_latency_ms / 1000)

            stats = {
                "model": str(model_path),
                "provider": session.get_providers()[0],
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations,
                "batch_size": batch_size,
                "average_latency_ms": round(avg_latency_ms, 4),
                "min_latency_ms": round(min_latency_ms, 4),
                "max_latency_ms": round(max_latency_ms, 4),
                "throughput_inferences_per_second": round(throughput, 4),
                "inputs": self._input_summary(session.get_inputs()),
            }

            logger.info(
                "Benchmark completed for %s: avg %.4f ms, throughput %.2f "
                "infer/sec",
                model_path,
                stats["average_latency_ms"],
                stats["throughput_inferences_per_second"],
            )
            return stats
        except Exception as exc:
            logger.exception("ONNX latency benchmark failed: %s", exc)
            if isinstance(exc, BenchmarkError):
                raise
            raise BenchmarkError(f"Failed to benchmark ONNX model '{model_path}'") from exc

    def benchmark_pair(self, original_model, optimized_model, report_path=None):
        original_stats = self.benchmark_model(original_model)
        optimized_stats = self.benchmark_model(optimized_model)
        comparison = self._compare(original_stats, optimized_stats)

        report = {
            "status": "benchmarked",
            "backend": "onnxruntime",
            "original": original_stats,
            "optimized": optimized_stats,
            "comparison": comparison,
        }

        if report_path is not None:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as file:
                json.dump(report, file, indent=2)
            logger.info("Benchmark statistics report written: %s", report_path)

        return report

    def _import_onnxruntime(self):
        try:
            import onnxruntime as ort

            return ort
        except ImportError as exc:
            raise BenchmarkError(
                "onnxruntime is required for ONNX inference benchmarking"
            ) from exc

    def _create_session(self, ort, model_path):
        if not model_path.exists():
            raise BenchmarkError(f"Model not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=self.providers,
        )

    def _create_dummy_inputs(self, session):
        inputs = {}
        for input_meta in session.get_inputs():
            shape = self._concrete_shape(input_meta.shape)
            inputs[input_meta.name] = self._dummy_array(shape, input_meta.type)
        return inputs

    def _concrete_shape(self, shape):
        concrete_shape = []
        for dim in shape:
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            else:
                concrete_shape.append(1)
        return concrete_shape

    def _dummy_array(self, shape, onnx_type):
        try:
            import numpy as np
        except ImportError as exc:
            raise BenchmarkError("numpy is required for benchmark inputs") from exc

        dtype = self._numpy_dtype(np, onnx_type)
        if dtype in (np.float16, np.float32, np.float64):
            return np.random.random_sample(shape).astype(dtype)
        if dtype == np.bool_:
            return np.zeros(shape, dtype=dtype)
        return np.random.randint(low=0, high=10, size=shape).astype(dtype)

    def _numpy_dtype(self, np, onnx_type):
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }

        if onnx_type not in dtype_map:
            raise BenchmarkError(f"Unsupported ONNX input type: {onnx_type}")
        return dtype_map[onnx_type]

    def _batch_size(self, inputs):
        if not inputs:
            return 1

        first_shape = self._concrete_shape(inputs[0].shape)
        if not first_shape:
            return 1
        return first_shape[0]

    def _input_summary(self, inputs):
        return [
            {
                "name": input_meta.name,
                "shape": self._concrete_shape(input_meta.shape),
                "type": input_meta.type,
            }
            for input_meta in inputs
        ]

    def _compare(self, original_stats, optimized_stats):
        original_latency = original_stats["average_latency_ms"]
        optimized_latency = optimized_stats["average_latency_ms"]
        original_throughput = original_stats["throughput_inferences_per_second"]
        optimized_throughput = optimized_stats["throughput_inferences_per_second"]

        latency_improvement = 0.0
        if original_latency > 0:
            latency_improvement = (
                (original_latency - optimized_latency) / original_latency
            ) * 100

        throughput_improvement = 0.0
        if original_throughput > 0:
            throughput_improvement = (
                (optimized_throughput - original_throughput)
                / original_throughput
            ) * 100

        return {
            "average_latency_improvement_percent": round(
                latency_improvement,
                2,
            ),
            "throughput_improvement_percent": round(
                throughput_improvement,
                2,
            ),
            "optimized_is_faster": optimized_latency < original_latency,
        }
