
from pathlib import Path
import json
import shutil

from benchmark.accuracy_validator import (
    AccuracyValidationError,
    OnnxAccuracyValidator,
)
from benchmark.latency_benchmark import BenchmarkError, OnnxLatencyBenchmark
from core.logger import logger
from optimize.fp16_optimizer import FP16Optimizer, FP16OptimizationError
from optimize.graph_optimizer import GraphOptimizationError, OnnxGraphOptimizer
from optimize.model_size_analyzer import ModelSizeAnalysisError, OnnxModelSizeAnalyzer
from optimize.static_int8_optimizer import (
    StaticInt8OptimizationError,
    StaticInt8Optimizer,
)


class OptimizationError(RuntimeError):
    """Raised when model optimization cannot be completed."""


class SmartOptimizer:

    def optimize(self, model_path, hardware_profile):
        model_path = Path(model_path)
        quantization_strategy = hardware_profile.get("quantization", "int8").lower()

        logger.info(
            f"Optimizing model for target: {hardware_profile['name']}"
        )
        logger.info(f"Quantization strategy: {quantization_strategy}")

        if quantization_strategy not in {"int8", "static_int8", "fp16", "graph"}:
            raise OptimizationError(
                "Supported ONNX optimization strategies are dynamic INT8, "
                "static INT8, FP16, and graph. "
                f"Requested strategy: {quantization_strategy}"
            )

        output_dir = Path("output")
        reports_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        optimized_model = output_dir / f"{model_path.stem}_{quantization_strategy}.onnx"
        graph_optimized_model = output_dir / f"{model_path.stem}_graph_optimized.onnx"

        try:
            self._validate_input_model(model_path)
            original_size = self._model_size_bytes(model_path)
            precision_input_model = model_path
            graph_optimization_stats = None

            if self._graph_optimization_enabled(hardware_profile):
                graph_optimization_stats = self._optimize_graph(
                    model_path,
                    graph_optimized_model,
                    hardware_profile,
                    reports_dir / "graph_optimization.json",
                )
                precision_input_model = graph_optimized_model
            else:
                logger.info("Graph optimization disabled by hardware profile")

            backend = self._select_backend(quantization_strategy, hardware_profile)
            graph_stats = backend["optimizer"](precision_input_model, optimized_model)

            optimized_size = self._model_size_bytes(optimized_model)
            stats = self._build_optimization_stats(
                model_path=model_path,
                optimized_model=optimized_model,
                hardware_profile=hardware_profile,
                original_size=original_size,
                optimized_size=optimized_size,
                graph_stats=graph_stats,
                technique=backend["technique"],
                quantization=quantization_strategy,
                graph_optimization_stats=graph_optimization_stats,
            )
            size_analysis = self._analyze_model_sizes(
                {
                    "original": model_path,
                    "graph_optimized": graph_optimized_model,
                    "optimized": optimized_model,
                },
                reports_dir / "model_size_analysis.json",
            )
            stats["model_size_analysis"] = size_analysis
            benchmark_report = self._benchmark_models(
                model_path,
                optimized_model,
                reports_dir / "benchmark_statistics.json",
            )
            stats["benchmark"] = benchmark_report
            accuracy_report = self._validate_accuracy(
                model_path,
                optimized_model,
                reports_dir / "accuracy_validation.json",
            )
            stats["accuracy_validation"] = accuracy_report

            self._write_json(output_dir / "optimization_stats.json", stats)
            self._write_json(reports_dir / "optimization_summary.json", stats)
            self._write_json(
                reports_dir / "validation_summary.json",
                self._build_validation_summary(stats),
            )

            logger.info(
                "%s optimization completed: %.2f MB -> %.2f MB (%.2f%% "
                "compression)",
                quantization_strategy.upper(),
                stats["original_size_mb"],
                stats["optimized_size_mb"],
                stats["compression_percent"],
            )
            logger.info("Optimization pipeline completed")

            return str(optimized_model)
        except Exception as exc:
            if optimized_model.exists():
                optimized_model.unlink()
            logger.exception("Optimization pipeline failed: %s", exc)
            if isinstance(
                exc,
                (
                    OptimizationError,
                    FP16OptimizationError,
                    StaticInt8OptimizationError,
                    BenchmarkError,
                    AccuracyValidationError,
                    GraphOptimizationError,
                    ModelSizeAnalysisError,
                ),
            ):
                raise
            raise OptimizationError(
                f"Failed to optimize ONNX model '{model_path}'"
            ) from exc

    def _validate_input_model(self, model_path):
        if not model_path.exists():
            raise OptimizationError(f"Model not found: {model_path}")

        if model_path.suffix.lower() != ".onnx":
            raise OptimizationError(
                f"Optimization requires an ONNX model: {model_path}"
            )

    def _select_backend(self, quantization_strategy, hardware_profile=None):
        if quantization_strategy == "int8":
            return {
                "technique": "onnxruntime_dynamic_int8",
                "optimizer": self._quantize_dynamic_int8,
            }

        if quantization_strategy == "static_int8":
            static_optimizer = StaticInt8Optimizer(
                calibration_config=(
                    hardware_profile or {}
                ).get("calibration", {}),
                calibration_config_path=(hardware_profile or {}).get(
                    "calibration_config_path",
                    "configs/calibration.yaml",
                ),
            )
            return {
                "technique": static_optimizer.technique,
                "optimizer": static_optimizer.optimize,
            }

        if quantization_strategy == "fp16":
            fp16_optimizer = FP16Optimizer()
            return {
                "technique": fp16_optimizer.technique,
                "optimizer": fp16_optimizer.optimize,
            }

        if quantization_strategy == "graph":
            return {
                "technique": "onnx_graph_only_recommendation_path",
                "optimizer": self._graph_only_artifact,
            }

        raise OptimizationError(
            f"Unsupported optimization strategy: {quantization_strategy}"
        )

    def _graph_only_artifact(self, model_path, optimized_model):
        logger.info(
            "Preserving ONNX graph-optimized artifact without precision quantization: %s -> %s",
            model_path,
            optimized_model,
        )
        if Path(optimized_model).exists():
            Path(optimized_model).unlink()
        shutil.copy2(model_path, optimized_model)
        graph_stats = self._validate_optimized_model(
            optimized_model,
            expect_quantized=False,
        )
        graph_stats["precision_quantization_skipped"] = True
        graph_stats["skip_reason"] = (
            "Selected recommendation requires calibration/export support not "
            "available in the current ONNX optimization executor"
        )
        return graph_stats

    def _benchmark_models(self, original_model, optimized_model, report_path):
        benchmark = OnnxLatencyBenchmark()
        return benchmark.benchmark_pair(
            original_model=original_model,
            optimized_model=optimized_model,
            report_path=report_path,
        )

    def _validate_accuracy(self, original_model, optimized_model, report_path):
        validator = OnnxAccuracyValidator()
        return validator.validate_pair(
            original_model=original_model,
            optimized_model=optimized_model,
            report_path=report_path,
        )

    def _graph_optimization_enabled(self, hardware_profile):
        return hardware_profile.get("graph_optimization", True)

    def _optimize_graph(
        self,
        model_path,
        optimized_model,
        hardware_profile,
        report_path,
    ):
        optimizer = OnnxGraphOptimizer(
            optimization_level=hardware_profile.get(
                "graph_optimization_level",
                "all",
            ),
            max_size_increase_percent=hardware_profile.get(
                "max_graph_size_increase_percent",
                10.0,
            ),
        )
        stats = optimizer.optimize(
            model_path,
            optimized_model,
            hardware_profile=hardware_profile,
        )
        self._write_json(report_path, stats)
        return stats

    def _analyze_model_sizes(self, model_paths, report_path):
        analyzer = OnnxModelSizeAnalyzer()
        return analyzer.analyze(model_paths, report_path=report_path)

    def _quantize_dynamic_int8(self, model_path, optimized_model):
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            import onnx
        except ImportError as exc:
            raise OptimizationError(
                "onnxruntime is required for INT8 dynamic quantization"
            ) from exc

        logger.info(
            "Running ONNX Runtime dynamic INT8 quantization: %s -> %s",
            model_path,
            optimized_model,
        )

        candidate_model = Path(optimized_model).with_name(
            f"{Path(optimized_model).stem}.candidate{Path(optimized_model).suffix}"
        )
        if candidate_model.exists():
            candidate_model.unlink()

        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(candidate_model),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            use_external_data_format=False,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

        if not candidate_model.exists():
            raise OptimizationError(
                f"Quantized ONNX artifact was not created: {candidate_model}"
            )

        input_size = self._model_size_bytes(model_path)
        candidate_size = self._model_size_bytes(candidate_model)
        size_increase = self._size_delta_percent(input_size, candidate_size)
        if size_increase > 10.0:
            logger.warning(
                "QOperator dynamic quantization increased graph size by %.2f%%; "
                "preserving compact source ONNX artifact for downstream reports",
                size_increase,
            )
            if Path(optimized_model).exists():
                Path(optimized_model).unlink()
            shutil.copy2(model_path, optimized_model)
            candidate_model.unlink()
            graph_stats = self._validate_optimized_model(optimized_model)
            graph_stats["quantization_size_guard_applied"] = True
            graph_stats["quantization_skip_reason"] = (
                "Dynamic INT8 quantization candidate inflated artifact size"
            )
            graph_stats["quantized_candidate_size_increase_percent"] = round(
                size_increase,
                2,
            )
            return graph_stats

        if Path(optimized_model).exists():
            Path(optimized_model).unlink()
        shutil.move(str(candidate_model), str(optimized_model))
        graph_stats = self._validate_optimized_model(optimized_model)
        graph_stats["quantization_size_guard_applied"] = False
        graph_stats["quantized_candidate_size_increase_percent"] = round(
            size_increase,
            2,
        )
        return graph_stats

    def _size_delta_percent(self, original_size, new_size):
        if original_size <= 0:
            return 0.0
        return ((new_size - original_size) / original_size) * 100

    def _validate_optimized_model(self, optimized_model, expect_quantized=True):
        try:
            import onnx
        except ImportError as exc:
            raise OptimizationError(
                "onnx is required to validate optimized ONNX artifacts"
            ) from exc

        model = onnx.load(str(optimized_model))
        onnx.checker.check_model(model)

        ops = [node.op_type for node in model.graph.node]
        quantized_ops = [
            op
            for op in ops
            if "Quantize" in op or "Dequantize" in op or "Integer" in op
        ]

        if expect_quantized and not quantized_ops:
            logger.warning(
                "Optimized model validated, but no dynamic quantization ops "
                "were detected. The source graph may not contain dynamically "
                "quantizable operators."
            )

        return {
            "node_count": len(ops),
            "unique_ops": sorted(set(ops)),
            "quantized_operator_count": len(quantized_ops),
            "quantized_ops": sorted(set(quantized_ops)),
        }

    def _model_size_bytes(self, model_path):
        return Path(model_path).stat().st_size

    def _build_optimization_stats(
        self,
        model_path,
        optimized_model,
        hardware_profile,
        original_size,
        optimized_size,
        graph_stats,
        technique,
        quantization,
        graph_optimization_stats,
    ):
        compression_percent = 0.0
        if original_size > 0:
            compression_percent = (
                (original_size - optimized_size) / original_size
            ) * 100

        return {
            "status": "optimized",
            "technique": technique,
            "hardware_target": hardware_profile.get("name", "unknown"),
            "quantization": quantization,
            "original_model": str(model_path),
            "optimized_model": str(optimized_model),
            "original_size_bytes": original_size,
            "optimized_size_bytes": optimized_size,
            "original_size_mb": round(original_size / (1024 * 1024), 4),
            "optimized_size_mb": round(optimized_size / (1024 * 1024), 4),
            "compression_percent": round(compression_percent, 2),
            "size_reduction_bytes": original_size - optimized_size,
            "graph_optimization": graph_optimization_stats,
            "graph": graph_stats,
            "decision_engine": {
                "rule_id": hardware_profile.get("decision_rule_id"),
                "primary_optimization": hardware_profile.get(
                    "decision_primary_optimization"
                ),
                "pipeline": hardware_profile.get("decision_pipeline", []),
                "rejected_optimizations": hardware_profile.get(
                    "decision_rejected_optimizations",
                    [],
                ),
                "reasons": hardware_profile.get("decision_reasons", []),
                "deployment_recommendations": hardware_profile.get(
                    "deployment_recommendations",
                    [],
                ),
                "deployment_feasible": hardware_profile.get("deployment_feasible"),
                "deployment_feasibility_score": hardware_profile.get(
                    "deployment_feasibility_score"
                ),
            },
        }

    def _build_validation_summary(self, stats):
        return {
            "status": "validated",
            "model": stats["optimized_model"],
            "validation": "onnx.checker.check_model",
            "optimization": {
                "technique": stats["technique"],
                "quantization": stats["quantization"],
                "compression_percent": stats["compression_percent"],
                "original_size_mb": stats["original_size_mb"],
                "optimized_size_mb": stats["optimized_size_mb"],
                "graph_optimization": stats["graph_optimization"],
                "graph": stats["graph"],
            },
        }

    def _write_json(self, path, payload):
        with open(path, "w") as file:
            json.dump(payload, file, indent=2)
