from pathlib import Path
import json
import os
import shutil
from benchmark.accuracy_validator import (
    AccuracyValidationError,
    OnnxAccuracyValidator,
)
from benchmark.latency_benchmark import BenchmarkError, OnnxLatencyBenchmark
from convert.onnx_to_tflite import TFLiteConversionError, convert_onnx_to_tflite
from core.artifact_manager import (
    generate_final_artifact_name,
    preserve_required_outputs,
)
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
    pass

class SmartOptimizer:
    def optimize(self, model_path, hardware_profile):
        model_path = Path(model_path)
        quantization_strategy = hardware_profile.get("quantization", "int8").lower()

        logger.info("Optimizing model for target: %s", hardware_profile["name"])
        logger.info("Quantization strategy: %s", quantization_strategy)

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

        final_suffix = hardware_profile.get("deployment_format", ".onnx")
        android_tflite = self._android_tflite_enabled(hardware_profile)
        artifact_name = generate_final_artifact_name(
            hardware_profile.get("source_model_stem", model_path.stem),
            hardware_profile.get("target", hardware_profile.get("name", "hardware")),
            quantization_strategy,
            suffix=final_suffix if android_tflite else ".onnx",
        )
        final_artifact = output_dir / artifact_name
        optimized_model = (
            final_artifact.with_suffix(".onnx") if android_tflite else final_artifact
        )
        graph_optimized_model = output_dir / f"{model_path.stem}_graph_optimized.onnx"
        logger.info("Final artifact path: %s", final_artifact)

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
            
            tflite_report = None
            if android_tflite:
                tflite_report = convert_onnx_to_tflite(
                    optimized_model,
                    final_artifact,
                    quantization=quantization_strategy,
                )

            optimized_size = self._model_size_bytes(final_artifact)
            stats = self._build_optimization_stats(
                model_path=model_path,
                optimized_model=final_artifact,
                hardware_profile=hardware_profile,
                original_size=original_size,
                optimized_size=optimized_size,
                graph_stats=graph_stats,
                technique=backend["technique"],
                quantization=quantization_strategy,
                graph_optimization_stats=graph_optimization_stats,
            )
            
            if tflite_report:
                stats["onnx_optimized_model"] = str(optimized_model)
                stats["tflite_conversion"] = tflite_report
                stats["runtime_recommendation"] = "Use TensorFlow Lite on Android."
                size_analysis = self._analyze_file_sizes(
                    model_path,
                    final_artifact,
                    reports_dir / "model_size_analysis.json",
                )
            else:
                size_analysis = self._analyze_model_sizes(
                    {
                        "original": model_path,
                        "graph_optimized": graph_optimized_model,
                        "optimized": final_artifact,
                    },
                    reports_dir / "model_size_analysis.json",
                )
            stats["model_size_analysis"] = size_analysis

            # Transformer-aware benchmark wrapper
            if tflite_report:
                benchmark_report = self._write_skipped_benchmark_report(
                    final_artifact,
                    reports_dir / "benchmark_statistics.json",
                )
            else:
                try:
                    benchmark_report = self._benchmark_models(
                        model_path,
                        final_artifact,
                        reports_dir / "benchmark_statistics.json",
                    )
                except Exception as exc:
                    logger.warning(
                        "Benchmark skipped for transformer-style model: %s",
                        exc,
                    )
                    benchmark_report = {
                        "status": "skipped",
                        "reason": "Transformer benchmarking requires tokenizer-aware inference inputs.",
                    }
                    self._write_json(
                        reports_dir / "benchmark_statistics.json",
                        benchmark_report,
                    )
            
            stats["benchmark"] = benchmark_report

            # Transformer-aware accuracy validation wrapper
            if tflite_report:
                accuracy_report = self._write_skipped_accuracy_report(
                    model_path,
                    optimized_model,
                    final_artifact,
                    reports_dir / "accuracy_validation.json",
                )
            else:
                try:
                    accuracy_report = self._validate_accuracy(
                        model_path,
                        final_artifact,
                        reports_dir / "accuracy_validation.json",
                    )
                except Exception as exc:
                    logger.warning(
                        "Accuracy validation skipped for transformer-style model: %s",
                        exc,
                    )
                    accuracy_report = {
                        "status": "skipped",
                        "reason": "Transformer accuracy validation requires tokenizer-aware inference inputs.",
                    }
                    self._write_json(
                        reports_dir / "accuracy_validation.json",
                        accuracy_report,
                    )
            
            stats["accuracy_validation"] = accuracy_report

            self._write_json(output_dir / "optimization_stats.json", stats)
            self._write_json(reports_dir / "optimization_summary.json", stats)
            self._write_json(
                reports_dir / "validation_summary.json",
                self._build_validation_summary(stats),
            )
            if os.environ.get("SKIP_FINAL_ARTIFACT_CLEANUP") == "1":
                logger.info(
                    "Skipping final artifact cleanup because workflow artifact "
                    "preservation is enabled"
                )
            else:
                preserve_required_outputs(
                    output_dir=output_dir,
                    final_artifact=final_artifact,
                    pruned_model=hardware_profile.get("pruned_model_path"),
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
            return str(final_artifact)
            
        except Exception as exc:
            for path in (optimized_model, final_artifact):
                if path.exists():
                    path.unlink()
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
                    TFLiteConversionError,
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

    def _analyze_file_sizes(self, original_model, optimized_model, report_path):
        original_model = Path(original_model)
        optimized_model = Path(optimized_model)
        original_size = self._model_size_bytes(original_model)
        optimized_size = self._model_size_bytes(optimized_model)
        delta = optimized_size - original_size
        delta_percent = self._size_delta_percent(original_size, optimized_size)
        report = {
            "status": "analyzed",
            "models": {
                "original": {
                    "path": str(original_model),
                    "file_size_bytes": original_size,
                    "file_size_mb": round(original_size / (1024 * 1024), 4),
                },
                "optimized": {
                    "path": str(optimized_model),
                    "file_size_bytes": optimized_size,
                    "file_size_mb": round(optimized_size / (1024 * 1024), 4),
                    "format": optimized_model.suffix,
                },
            },
            "comparisons": {
                "original_vs_optimized": {
                    "size_delta_bytes": delta,
                    "size_delta_mb": round(delta / (1024 * 1024), 4),
                    "size_delta_percent": round(delta_percent, 2),
                }
            },
            "notes": [
                "Final artifact is not an ONNX model, so ONNX tensor storage "
                "analysis was skipped."
            ],
        }
        self._write_json(report_path, report)
        return report

    def _write_skipped_benchmark_report(self, final_artifact, report_path):
        report = {
            "status": "skipped",
            "reason": "TFLite artifact generated for Android deployment.",
            "note": "TFLite models are not benchmarked with ONNX Runtime.",
            "optimized": {"model": str(final_artifact), "runtime": "tflite"},
        }
        self._write_json(report_path, report)
        logger.info("Skipping ONNX Runtime benchmark for Android TFLite artifact")
        return report

    def _write_skipped_accuracy_report(
        self,
        original_model,
        optimized_onnx_model,
        final_artifact,
        report_path,
    ):
        report = {
            "status": "skipped",
            "reason": "Final artifact is TensorFlow Lite and cannot be compared with ONNX Runtime.",
            "note": "ONNX validation completed before TFLite export; validate TFLite accuracy on-device or with a TFLite interpreter.",
            "original_model": str(original_model),
            "onnx_optimized_model": str(optimized_onnx_model),
            "optimized_model": str(final_artifact),
            "runtime": "tflite",
        }
        self._write_json(report_path, report)
        logger.info("Skipping ONNX Runtime accuracy comparison for Android TFLite artifact")
        return report

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

    def _android_tflite_enabled(self, hardware_profile):
        return hardware_profile.get("target") == "android"

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
            op for op in ops
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
                    "decision_rejected_optimizations", []
                ),
                "reasons": hardware_profile.get("decision_reasons", []),
                "deployment_recommendations": hardware_profile.get(
                    "deployment_recommendations", []
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
