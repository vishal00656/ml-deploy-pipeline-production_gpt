from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json

from core.logger import logger


class ModelSizeAnalysisError(RuntimeError):
    """Raised when ONNX model size analysis cannot be completed."""


class OnnxModelSizeAnalyzer:
    """Analyze ONNX artifact size, tensor storage, and serialization overhead."""

    def analyze(self, model_paths, report_path=None):
        try:
            analyses = {}
            for label, model_path in model_paths.items():
                if model_path is None:
                    continue
                path = Path(model_path)
                if not path.exists():
                    continue
                analyses[label] = self.analyze_model(path)

            comparisons = self._build_comparisons(analyses)
            report = {
                "status": "analyzed",
                "models": analyses,
                "comparisons": comparisons,
                "notes": [
                    "initializer_total_bytes is the sum of raw tensor storage.",
                    "serialization_overhead_bytes estimates protobuf graph, "
                    "metadata, node definitions, tensor names, and any duplicated "
                    "non-initializer data.",
                ],
            }

            if report_path is not None:
                report_path = Path(report_path)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, "w") as file:
                    json.dump(report, file, indent=2)
                logger.info("Model size analysis report written: %s", report_path)

            self._log_findings(report)
            return report
        except Exception as exc:
            logger.exception("ONNX model size analysis failed: %s", exc)
            if isinstance(exc, ModelSizeAnalysisError):
                raise
            raise ModelSizeAnalysisError("Failed to analyze ONNX model sizes") from exc

    def analyze_model(self, model_path):
        onnx = self._import_onnx()
        model = onnx.load(str(model_path), load_external_data=True)

        file_size = model_path.stat().st_size
        initializers = self._analyze_initializers(onnx, model)
        tensor_counts = self._tensor_counts(onnx, model)
        node_count = len(model.graph.node)
        tensor_count = (
            len(model.graph.initializer)
            + len(model.graph.input)
            + len(model.graph.output)
            + len(model.graph.value_info)
        )
        initializer_bytes = initializers["initializer_total_bytes"]
        overhead = file_size - initializer_bytes

        return {
            "path": str(model_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 4),
            "initializer_total_bytes": initializer_bytes,
            "initializer_total_mb": round(initializer_bytes / (1024 * 1024), 4),
            "graph_serialization_overhead_bytes": overhead,
            "graph_serialization_overhead_mb": round(overhead / (1024 * 1024), 4),
            "node_count": node_count,
            "tensor_count": tensor_count,
            **tensor_counts,
            **initializers,
            "external_data_initializer_count": self._external_data_count(model),
            "metadata_props_count": len(model.metadata_props),
            "opset_imports": [
                {"domain": item.domain, "version": item.version}
                for item in model.opset_import
            ],
        }

    def _analyze_initializers(self, onnx, model):
        total_bytes = 0
        duplicates = []
        hashes = defaultdict(list)
        fp32_initializer_count = 0
        fp16_initializer_count = 0
        int8_initializer_count = 0

        for initializer in model.graph.initializer:
            tensor_bytes = self._initializer_nbytes(onnx, initializer)
            total_bytes += tensor_bytes

            if initializer.data_type == onnx.TensorProto.FLOAT:
                fp32_initializer_count += 1
            elif initializer.data_type == onnx.TensorProto.FLOAT16:
                fp16_initializer_count += 1
            elif initializer.data_type in (
                onnx.TensorProto.INT8,
                onnx.TensorProto.UINT8,
            ):
                int8_initializer_count += 1

            digest = self._initializer_hash(initializer)
            if digest is not None:
                hashes[digest].append(
                    {
                        "name": initializer.name,
                        "bytes": tensor_bytes,
                        "data_type": onnx.TensorProto.DataType.Name(
                            initializer.data_type
                        ),
                        "dims": list(initializer.dims),
                    }
                )

        duplicate_bytes = 0
        for entries in hashes.values():
            if len(entries) <= 1:
                continue
            duplicated_group_bytes = sum(item["bytes"] for item in entries[1:])
            duplicate_bytes += duplicated_group_bytes
            duplicates.append(
                {
                    "count": len(entries),
                    "duplicated_bytes": duplicated_group_bytes,
                    "initializers": entries,
                }
            )

        return {
            "initializer_total_bytes": total_bytes,
            "fp32_initializer_count": fp32_initializer_count,
            "fp16_initializer_count": fp16_initializer_count,
            "int8_initializer_count": int8_initializer_count,
            "duplicated_initializer_group_count": len(duplicates),
            "duplicated_initializer_bytes": duplicate_bytes,
            "duplicated_initializers": duplicates[:25],
        }

    def _tensor_counts(self, onnx, model):
        counts = {
            "fp32_tensor_count": 0,
            "fp16_tensor_count": 0,
            "int8_tensor_count": 0,
        }

        for value_info in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        ):
            elem_type = value_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT:
                counts["fp32_tensor_count"] += 1
            elif elem_type == onnx.TensorProto.FLOAT16:
                counts["fp16_tensor_count"] += 1
            elif elem_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8):
                counts["int8_tensor_count"] += 1

        for initializer in model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                counts["fp32_tensor_count"] += 1
            elif initializer.data_type == onnx.TensorProto.FLOAT16:
                counts["fp16_tensor_count"] += 1
            elif initializer.data_type in (
                onnx.TensorProto.INT8,
                onnx.TensorProto.UINT8,
            ):
                counts["int8_tensor_count"] += 1

        return counts

    def _initializer_nbytes(self, onnx, initializer):
        element_size = {
            onnx.TensorProto.FLOAT: 4,
            onnx.TensorProto.FLOAT16: 2,
            onnx.TensorProto.DOUBLE: 8,
            onnx.TensorProto.INT64: 8,
            onnx.TensorProto.UINT64: 8,
            onnx.TensorProto.INT32: 4,
            onnx.TensorProto.UINT32: 4,
            onnx.TensorProto.INT16: 2,
            onnx.TensorProto.UINT16: 2,
            onnx.TensorProto.INT8: 1,
            onnx.TensorProto.UINT8: 1,
            onnx.TensorProto.BOOL: 1,
        }.get(initializer.data_type, 0)

        element_count = 1
        for dim in initializer.dims:
            element_count *= dim
        return element_count * element_size

    def _initializer_hash(self, initializer):
        payload = initializer.raw_data
        if not payload:
            payload = str(
                (
                    initializer.data_type,
                    tuple(initializer.dims),
                    tuple(initializer.float_data),
                    tuple(initializer.int32_data),
                    tuple(initializer.int64_data),
                )
            ).encode("utf-8")
        if not payload:
            return None
        return sha256(payload).hexdigest()

    def _external_data_count(self, model):
        count = 0
        for initializer in model.graph.initializer:
            if initializer.external_data:
                count += 1
        return count

    def _build_comparisons(self, analyses):
        comparisons = {}
        original = analyses.get("original")
        if not original:
            return comparisons

        for label, analysis in analyses.items():
            if label == "original":
                continue
            size_delta = analysis["file_size_bytes"] - original["file_size_bytes"]
            size_delta_percent = 0.0
            if original["file_size_bytes"] > 0:
                size_delta_percent = (
                    size_delta / original["file_size_bytes"]
                ) * 100
            comparisons[f"original_vs_{label}"] = {
                "size_delta_bytes": size_delta,
                "size_delta_mb": round(size_delta / (1024 * 1024), 4),
                "size_delta_percent": round(size_delta_percent, 2),
                "initializer_delta_bytes": analysis["initializer_total_bytes"]
                - original["initializer_total_bytes"],
                "node_count_delta": analysis["node_count"]
                - original["node_count"],
                "serialization_overhead_delta_bytes": analysis[
                    "graph_serialization_overhead_bytes"
                ]
                - original["graph_serialization_overhead_bytes"],
            }
        return comparisons

    def _log_findings(self, report):
        for label, analysis in report["models"].items():
            if analysis["duplicated_initializer_group_count"] > 0:
                logger.warning(
                    "Detected duplicated initializers in %s: %s groups, %.4f MB",
                    label,
                    analysis["duplicated_initializer_group_count"],
                    analysis["duplicated_initializer_bytes"] / (1024 * 1024),
                )
            if (
                analysis["fp32_initializer_count"] > 0
                and analysis["fp16_initializer_count"] > 0
            ):
                logger.warning(
                    "Model contains both FP32 and FP16 initializers in %s: "
                    "fp32=%s fp16=%s",
                    label,
                    analysis["fp32_initializer_count"],
                    analysis["fp16_initializer_count"],
                )
            logger.info(
                "Serialized ONNX metadata overhead for %s: %.4f MB",
                label,
                analysis["graph_serialization_overhead_mb"],
            )

        for name, comparison in report["comparisons"].items():
            if comparison["size_delta_percent"] > 25:
                logger.warning(
                    "%s increased graph size by %.2f%%",
                    name,
                    comparison["size_delta_percent"],
                )

    def _import_onnx(self):
        try:
            import onnx

            return onnx
        except ImportError as exc:
            raise ModelSizeAnalysisError("onnx is required for size analysis") from exc
