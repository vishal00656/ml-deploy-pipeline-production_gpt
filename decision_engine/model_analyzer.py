from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from core.logger import logger


@dataclass
class TinyMLSuitability:
    estimated_memory_footprint_bytes: int
    estimated_memory_footprint_mb: float
    microcontroller_suitable: bool
    huge_model_rejection: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelAnalysis:
    model_path: str
    source_format: str
    architecture_category: str
    graph_characteristics: List[str]
    precision: str
    parameter_count: int
    initializer_count: int
    model_size_bytes: int
    model_size_mb: float
    node_count: int
    operator_distribution: Dict[str, int]
    tinyml_suitability: TinyMLSuitability
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        payload = asdict(self)
        payload["tinyml_suitability"] = self.tinyml_suitability.to_dict()
        return payload


class ModelAnalyzer:
    """Analyze model architecture and deployment suitability.

    Detailed graph analysis is available for ONNX models. Non-ONNX artifacts get
    a lightweight source analysis so the pipeline can log early intent before
    conversion, then the converted ONNX graph should be analyzed again.
    """

    ATTENTION_OPS = {
        "Attention",
        "MultiHeadAttention",
        "ScaledDotProductAttention",
        "LayerNormalization",
        "SkipLayerNormalization",
        "Softmax",
    }
    CONV_OPS = {"Conv", "ConvTranspose", "DepthwiseConv2dNative"}
    LINEAR_OPS = {"Gemm", "Linear"}
    MATMUL_OPS = {"MatMul", "BatchMatMul"}
    RNN_OPS = {"RNN", "LSTM", "GRU"}
    TINYML_UNFRIENDLY_OPS = {
        "Attention",
        "MultiHeadAttention",
        "NonMaxSuppression",
        "RoiAlign",
        "GridSample",
        "Loop",
        "Scan",
    }

    def analyze(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_path.suffix.lower() == ".onnx":
            return self._analyze_onnx(model_path)
        return self._analyze_source_artifact(model_path)

    def _analyze_onnx(self, model_path):
        onnx = self._import_onnx()
        model = onnx.load(str(model_path), load_external_data=True)
        ops = [node.op_type for node in model.graph.node]
        op_counts = Counter(ops)
        parameter_count = self._parameter_count(model)
        initializer_count = len(model.graph.initializer)
        precision = self._detect_precision(onnx, model)
        model_size = model_path.stat().st_size
        initializer_bytes = self._initializer_bytes(onnx, model)
        characteristics = self._graph_characteristics(op_counts, len(ops))
        architecture = self._architecture_category(
            op_counts,
            characteristics,
            parameter_count,
            model_size,
        )
        tinyml = self._tinyml_suitability(
            model_size,
            initializer_bytes,
            parameter_count,
            op_counts,
        )

        analysis = ModelAnalysis(
            model_path=str(model_path),
            source_format="onnx",
            architecture_category=architecture,
            graph_characteristics=characteristics,
            precision=precision,
            parameter_count=int(parameter_count),
            initializer_count=initializer_count,
            model_size_bytes=model_size,
            model_size_mb=round(model_size / (1024 * 1024), 4),
            node_count=len(ops),
            operator_distribution=dict(sorted(op_counts.items())),
            tinyml_suitability=tinyml,
        )
        self._log_analysis(analysis)
        return analysis

    def _analyze_source_artifact(self, model_path):
        name = model_path.name.lower()
        size = model_path.stat().st_size
        architecture = "Unknown"
        characteristics = []
        notes = [
            "Detailed graph analysis requires ONNX; source artifact analysis is heuristic."
        ]

        if "yolo" in name or "cnn" in name or "conv" in name:
            architecture = "CNN"
            characteristics.append("Conv-heavy")
        elif "transformer" in name or "bert" in name or "gpt" in name:
            architecture = "Transformer"
            characteristics.extend(["MatMul-heavy", "Attention-heavy"])
        elif "lstm" in name or "gru" in name or "rnn" in name:
            architecture = "RNN"

        tinyml = self._tinyml_suitability(
            model_size=size,
            initializer_bytes=size,
            parameter_count=0,
            op_counts=Counter(),
        )
        analysis = ModelAnalysis(
            model_path=str(model_path),
            source_format=model_path.suffix.lower().lstrip(".") or "unknown",
            architecture_category=architecture,
            graph_characteristics=characteristics,
            precision="unknown",
            parameter_count=0,
            initializer_count=0,
            model_size_bytes=size,
            model_size_mb=round(size / (1024 * 1024), 4),
            node_count=0,
            operator_distribution={},
            tinyml_suitability=tinyml,
            notes=notes,
        )
        self._log_analysis(analysis)
        return analysis

    def _graph_characteristics(self, op_counts, node_count):
        if node_count == 0:
            return []

        characteristics = []
        conv_count = sum(op_counts[op] for op in self.CONV_OPS)
        matmul_count = sum(op_counts[op] for op in self.MATMUL_OPS)
        linear_count = sum(op_counts[op] for op in self.LINEAR_OPS)
        attention_count = sum(op_counts[op] for op in self.ATTENTION_OPS)

        if conv_count >= 5 or conv_count / node_count >= 0.20:
            characteristics.append("Conv-heavy")
        if matmul_count >= 3 or matmul_count / node_count >= 0.15:
            characteristics.append("MatMul-heavy")
        if linear_count >= 3 or linear_count / node_count >= 0.15:
            characteristics.append("Linear-heavy")
        if (
            attention_count >= 3
            or op_counts.get("Attention", 0) > 0
            or (matmul_count >= 4 and op_counts.get("Softmax", 0) > 0)
        ):
            characteristics.append("Attention-heavy")

        return characteristics or ["Generic"]

    def _architecture_category(
        self,
        op_counts,
        characteristics,
        parameter_count,
        model_size,
    ):
        if any(op_counts[op] > 0 for op in self.RNN_OPS):
            return "RNN"
        if "Attention-heavy" in characteristics:
            return "Transformer"
        if "Conv-heavy" in characteristics:
            return "CNN"
        if self._is_tiny_shape(parameter_count, model_size, op_counts):
            return "TinyML-compatible"
        return "Unknown"

    def _detect_precision(self, onnx, model):
        fp32_count = 0
        fp16_count = 0

        for tensor in model.graph.initializer:
            if tensor.data_type == onnx.TensorProto.FLOAT:
                fp32_count += 1
            elif tensor.data_type == onnx.TensorProto.FLOAT16:
                fp16_count += 1

        for value_info in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        ):
            elem_type = value_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT:
                fp32_count += 1
            elif elem_type == onnx.TensorProto.FLOAT16:
                fp16_count += 1

        if fp32_count > 0 and fp16_count > 0:
            return "mixed precision"
        if fp16_count > 0:
            return "FP16"
        if fp32_count > 0:
            return "FP32"
        return "unknown"

    def _parameter_count(self, model):
        total = 0
        for initializer in model.graph.initializer:
            count = 1
            for dim in initializer.dims:
                count *= int(dim)
            total += count
        return total

    def _initializer_bytes(self, onnx, model):
        total = 0
        for initializer in model.graph.initializer:
            total += self._tensor_nbytes(onnx, initializer)
        return total

    def _tensor_nbytes(self, onnx, tensor):
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
        }.get(tensor.data_type, 0)
        count = 1
        for dim in tensor.dims:
            count *= int(dim)
        return count * element_size

    def _tinyml_suitability(
        self,
        model_size,
        initializer_bytes,
        parameter_count,
        op_counts,
    ):
        footprint = max(model_size, initializer_bytes) + int(initializer_bytes * 0.25)
        reasons = []

        if footprint > 2 * 1024 * 1024:
            reasons.append("Estimated footprint exceeds typical TinyML memory budget")
        if parameter_count > 1_000_000:
            reasons.append("Parameter count is too high for microcontroller deployment")
        unsupported_ops = sorted(
            op for op in self.TINYML_UNFRIENDLY_OPS if op_counts.get(op, 0) > 0
        )
        if unsupported_ops:
            reasons.append(
                "Graph contains TinyML-unfriendly operators: "
                + ", ".join(unsupported_ops)
            )

        huge = footprint > 16 * 1024 * 1024 or parameter_count > 10_000_000
        if huge:
            reasons.append("Huge-model rejection threshold exceeded")

        suitable = not reasons and footprint <= 2 * 1024 * 1024
        return TinyMLSuitability(
            estimated_memory_footprint_bytes=footprint,
            estimated_memory_footprint_mb=round(footprint / (1024 * 1024), 4),
            microcontroller_suitable=suitable,
            huge_model_rejection=huge,
            reasons=reasons or ["Model fits initial TinyML footprint heuristics"],
        )

    def _is_tiny_shape(self, parameter_count, model_size, op_counts):
        if parameter_count <= 0:
            return False
        if parameter_count > 250_000 or model_size > 1024 * 1024:
            return False
        return not any(op_counts[op] > 0 for op in self.TINYML_UNFRIENDLY_OPS)

    def _log_analysis(self, analysis):
        if "Conv-heavy" in analysis.graph_characteristics:
            logger.info("Detected Conv-heavy %s architecture", analysis.architecture_category)
        elif "Attention-heavy" in analysis.graph_characteristics:
            logger.info("Detected Attention-heavy %s architecture", analysis.architecture_category)
        else:
            logger.info("Detected %s architecture", analysis.architecture_category)
        logger.info(
            "Model statistics: params=%s, initializers=%s, nodes=%s, size=%.4f MB",
            analysis.parameter_count,
            analysis.initializer_count,
            analysis.node_count,
            analysis.model_size_mb,
        )
        if analysis.tinyml_suitability.huge_model_rejection:
            logger.warning(
                "Huge-model rejection detected: estimated footprint %.4f MB",
                analysis.tinyml_suitability.estimated_memory_footprint_mb,
            )

    def _import_onnx(self):
        try:
            import onnx

            return onnx
        except ImportError as exc:
            raise RuntimeError("onnx is required for ONNX model analysis") from exc

