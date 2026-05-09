from collections import Counter
from pathlib import Path
import json

import numpy as np
import yaml

from calibration.calibration_reader import OnnxImageCalibrationDataReader
from core.logger import logger


class StaticInt8OptimizationError(RuntimeError):
    """Raised when ONNX static INT8 quantization cannot be completed."""


class StaticInt8Optimizer:
    """Run calibration-backed ONNX Runtime static INT8 quantization for CNNs."""

    technique = "onnxruntime_static_int8_qdq"
    quantization = "static_int8"

    def __init__(self, calibration_config=None, calibration_config_path=None):
        self.calibration_config_path = Path(
            calibration_config_path or "configs/calibration.yaml"
        )
        self.calibration_config = self._load_calibration_config()
        self.calibration_config.update(calibration_config or {})

    def optimize(self, model_path, output_path):
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running ONNX Runtime static INT8 quantization: %s -> %s",
            model_path,
            output_path,
        )

        try:
            quantize_static = self._import_quantize_static()
            quantization_input, normalization_stats = self._prepare_quantization_input(
                model_path,
                output_path,
            )
            reader = OnnxImageCalibrationDataReader(
                quantization_input,
                self.calibration_config,
            )

            logger.info("Starting static INT8 quantization")
            quantize_static(
                model_input=str(quantization_input),
                model_output=str(output_path),
                calibration_data_reader=reader,
                quant_format=self._quant_format(),
                op_types_to_quantize=self.calibration_config.get(
                    "op_types_to_quantize",
                    ["Conv", "MatMul", "Gemm"],
                ),
                per_channel=bool(self.calibration_config.get("per_channel", True)),
                reduce_range=bool(self.calibration_config.get("reduce_range", False)),
                activation_type=self._quant_type(
                    self.calibration_config.get("activation_type", "QUInt8")
                ),
                weight_type=self._quant_type(
                    self.calibration_config.get("weight_type", "QInt8")
                ),
                use_external_data_format=False,
                calibrate_method=self._calibration_method(),
                extra_options=self._extra_options(),
            )

            if not output_path.exists():
                raise StaticInt8OptimizationError(
                    f"Static INT8 artifact was not created: {output_path}"
                )

            stats = self.validate(quantization_input, output_path)
            stats["calibration"] = reader.statistics()
            stats["fp32_normalization"] = normalization_stats
            self._write_report(stats)
            logger.info(
                "Static INT8 quantization completed: quantized_ops=%s, conv_quantized=%s",
                stats["quantized_operator_count"],
                stats["conv_quantization_count"],
            )
            return stats
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            logger.exception("Static INT8 quantization failed: %s", exc)
            if isinstance(exc, StaticInt8OptimizationError):
                raise
            raise StaticInt8OptimizationError(
                f"Failed to statically quantize ONNX model '{model_path}'"
            ) from exc

    def validate(self, original_model_path, quantized_model_path):
        onnx = self._import_onnx()
        original_model = onnx.load(str(original_model_path), load_external_data=True)
        quantized_model = onnx.load(str(quantized_model_path), load_external_data=True)
        onnx.checker.check_model(quantized_model)
        runtime_validation = self._validate_onnxruntime_session(quantized_model_path)

        original_ops = [node.op_type for node in original_model.graph.node]
        quantized_ops = [node.op_type for node in quantized_model.graph.node]
        op_counts = Counter(quantized_ops)
        original_conv_count = original_ops.count("Conv")

        qdq_activation_count = self._activation_quantization_count(quantized_model)
        conv_quantization_count = self._conv_quantization_count(quantized_model)
        quantized_operator_count = sum(
            count
            for op, count in op_counts.items()
            if op in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "ConvInteger"}
            or "Integer" in op
        )

        if original_conv_count > 0 and conv_quantization_count == 0:
            logger.warning(
                "Static INT8 artifact validated, but no Conv quantization pattern was detected"
            )

        return {
            "node_count": len(quantized_ops),
            "unique_ops": sorted(set(quantized_ops)),
            "operator_distribution": dict(sorted(op_counts.items())),
            "quantized_operator_count": quantized_operator_count,
            "quantized_ops": sorted(
                op
                for op in set(quantized_ops)
                if op in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "ConvInteger"}
                or "Integer" in op
            ),
            "original_conv_count": original_conv_count,
            "conv_quantization_count": conv_quantization_count,
            "activation_quantization_count": qdq_activation_count,
            "weight_quantization_count": op_counts.get("DequantizeLinear", 0),
            "quantization_format": self.calibration_config.get("quant_format", "QDQ"),
            "op_types_to_quantize": self.calibration_config.get(
                "op_types_to_quantize",
                ["Conv", "MatMul", "Gemm"],
            ),
            "validation": {
                "onnx_checker_valid": True,
                "onnxruntime_session_valid": True,
                "onnxruntime_providers": runtime_validation["providers"],
            },
        }

    def _prepare_quantization_input(self, model_path, output_path):
        precision = self._detect_precision(model_path)
        stats = {
            "source_model": str(model_path),
            "normalized_model": str(model_path),
            "input_precision": precision,
            "normalization_applied": False,
            "fp16_tensor_count_before": precision["fp16_tensor_count"],
            "fp32_tensor_count_before": precision["fp32_tensor_count"],
        }

        if precision["precision"] == "FP32":
            logger.info("Detected FP32 graph before static quantization")
            stats["source_validation"] = self._validate_model_artifact(model_path)
            return model_path, stats

        if precision["fp16_tensor_count"] <= 0:
            logger.info(
                "Detected %s graph before static quantization",
                precision["precision"],
            )
            stats["source_validation"] = self._validate_model_artifact(model_path)
            return model_path, stats

        logger.info("Detected FP16 graph before static quantization")
        logger.info("Normalizing graph to FP32")
        normalized_model = output_path.with_name(
            f"{model_path.stem}_fp32_normalized.onnx"
        )
        self._convert_fp16_graph_to_fp32(model_path, normalized_model)
        normalized_validation = self._validate_model_artifact(normalized_model)
        output_precision = self._detect_precision(normalized_model)
        if output_precision["fp16_tensor_count"] > 0:
            raise StaticInt8OptimizationError(
                "FP32 normalization did not remove all FP16 tensors"
            )

        logger.info("FP32 normalization complete")
        stats.update(
            {
                "normalized_model": str(normalized_model),
                "normalization_applied": True,
                "output_precision": output_precision,
                "fp16_tensor_count_after": output_precision["fp16_tensor_count"],
                "fp32_tensor_count_after": output_precision["fp32_tensor_count"],
                "normalized_validation": normalized_validation,
            }
        )
        return normalized_model, stats

    def _convert_fp16_graph_to_fp32(self, model_path, output_path):
        onnx = self._import_onnx()
        model = onnx.load(str(model_path), load_external_data=True)
        self._convert_graph_to_fp32(onnx, model.graph)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            logger.warning("Shape inference after FP32 normalization skipped: %s", exc)
        onnx.save(model, str(output_path))

    def _convert_graph_to_fp32(self, onnx, graph):
        for value_info in (
            list(graph.input)
            + list(graph.output)
            + list(graph.value_info)
        ):
            self._convert_type_proto(onnx, value_info.type)
        self._convert_initializers(onnx, graph)
        self._convert_graph_attributes(onnx, graph)

    def _convert_type_proto(self, onnx, type_proto):
        if type_proto.HasField("tensor_type"):
            tensor_type = type_proto.tensor_type
            if tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                tensor_type.elem_type = onnx.TensorProto.FLOAT
            return
        if type_proto.HasField("sequence_type"):
            self._convert_type_proto(onnx, type_proto.sequence_type.elem_type)
            return
        if type_proto.HasField("optional_type"):
            self._convert_type_proto(onnx, type_proto.optional_type.elem_type)

    def _convert_initializers(self, onnx, graph):
        for index, initializer in enumerate(graph.initializer):
            if initializer.data_type != onnx.TensorProto.FLOAT16:
                continue
            replacement = self._fp16_tensor_to_fp32(onnx, initializer)
            graph.initializer[index].CopyFrom(replacement)

    def _convert_graph_attributes(self, onnx, graph):
        for node in graph.node:
            if node.op_type == "Cast":
                for attribute in node.attribute:
                    if (
                        attribute.name == "to"
                        and attribute.i == onnx.TensorProto.FLOAT16
                    ):
                        attribute.i = onnx.TensorProto.FLOAT
            for attribute in node.attribute:
                self._convert_attribute(onnx, attribute)

    def _convert_attribute(self, onnx, attribute):
        if attribute.type == onnx.AttributeProto.TENSOR:
            if attribute.t.data_type == onnx.TensorProto.FLOAT16:
                attribute.t.CopyFrom(self._fp16_tensor_to_fp32(onnx, attribute.t))
        elif attribute.type == onnx.AttributeProto.TENSORS:
            for index, tensor in enumerate(attribute.tensors):
                if tensor.data_type == onnx.TensorProto.FLOAT16:
                    attribute.tensors[index].CopyFrom(
                        self._fp16_tensor_to_fp32(onnx, tensor)
                    )
        elif attribute.type == onnx.AttributeProto.GRAPH:
            self._convert_graph_to_fp32(onnx, attribute.g)
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for graph in attribute.graphs:
                self._convert_graph_to_fp32(onnx, graph)

    def _fp16_tensor_to_fp32(self, onnx, tensor):
        from onnx import numpy_helper

        array = numpy_helper.to_array(tensor).astype(np.float32)
        replacement = numpy_helper.from_array(array, name=tensor.name)
        replacement.doc_string = tensor.doc_string
        return replacement

    def _detect_precision(self, model_path):
        onnx = self._import_onnx()
        model = onnx.load(str(model_path), load_external_data=True)
        fp16_tensor_count = 0
        fp32_tensor_count = 0
        fp16_initializer_count = 0
        fp32_initializer_count = 0

        for initializer in model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT16:
                fp16_initializer_count += 1
                fp16_tensor_count += 1
            elif initializer.data_type == onnx.TensorProto.FLOAT:
                fp32_initializer_count += 1
                fp32_tensor_count += 1

        for value_info in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        ):
            elem_type = value_info.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.FLOAT16:
                fp16_tensor_count += 1
            elif elem_type == onnx.TensorProto.FLOAT:
                fp32_tensor_count += 1

        if fp16_tensor_count > 0 and fp32_tensor_count > 0:
            precision = "mixed precision"
        elif fp16_tensor_count > 0:
            precision = "FP16"
        elif fp32_tensor_count > 0:
            precision = "FP32"
        else:
            precision = "unknown"

        return {
            "precision": precision,
            "fp16_tensor_count": fp16_tensor_count,
            "fp32_tensor_count": fp32_tensor_count,
            "fp16_initializer_count": fp16_initializer_count,
            "fp32_initializer_count": fp32_initializer_count,
        }

    def _validate_model_artifact(self, model_path):
        onnx = self._import_onnx()
        model = onnx.load(str(model_path), load_external_data=True)
        onnx.checker.check_model(model)
        runtime_validation = self._validate_onnxruntime_session(model_path)
        return {
            "onnx_checker_valid": True,
            "onnxruntime_session_valid": True,
            "onnxruntime_providers": runtime_validation["providers"],
        }

    def _validate_onnxruntime_session(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise StaticInt8OptimizationError(
                "onnxruntime is required to validate quantization artifacts"
            ) from exc

        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            return {"providers": session.get_providers()}
        except Exception as exc:
            raise StaticInt8OptimizationError(
                f"ONNX Runtime could not load model artifact '{model_path}'"
            ) from exc

    def _conv_quantization_count(self, model):
        qlinear_conv_count = sum(1 for node in model.graph.node if node.op_type == "QLinearConv")
        if qlinear_conv_count:
            return qlinear_conv_count

        producer_by_output = {}
        for node in model.graph.node:
            for output in node.output:
                producer_by_output[output] = node

        count = 0
        for node in model.graph.node:
            if node.op_type != "Conv":
                continue
            if any(
                producer_by_output.get(input_name, None) is not None
                and producer_by_output[input_name].op_type == "DequantizeLinear"
                for input_name in node.input
            ):
                count += 1
        return count

    def _activation_quantization_count(self, model):
        initializer_names = {item.name for item in model.graph.initializer}
        count = 0
        for node in model.graph.node:
            if node.op_type != "QuantizeLinear":
                continue
            if node.input and node.input[0] not in initializer_names:
                count += 1
        return count

    def _extra_options(self):
        options = {
            "ActivationSymmetric": bool(
                self.calibration_config.get("activation_symmetric", False)
            ),
            "WeightSymmetric": bool(
                self.calibration_config.get("weight_symmetric", True)
            ),
        }
        if self.calibration_config.get("dedicated_qdq_pair") is not None:
            options["DedicatedQDQPair"] = bool(
                self.calibration_config["dedicated_qdq_pair"]
            )
        return options

    def _quant_format(self):
        from onnxruntime.quantization import QuantFormat

        value = str(self.calibration_config.get("quant_format", "QDQ")).upper()
        if value == "QOPERATOR":
            return QuantFormat.QOperator
        return QuantFormat.QDQ

    def _quant_type(self, value):
        from onnxruntime.quantization import QuantType

        normalized = str(value).upper()
        if normalized == "QUINT8":
            return QuantType.QUInt8
        if normalized == "QINT8":
            return QuantType.QInt8
        raise StaticInt8OptimizationError(f"Unsupported quantization type: {value}")

    def _calibration_method(self):
        from onnxruntime.quantization import CalibrationMethod

        value = str(self.calibration_config.get("calibrate_method", "MinMax")).lower()
        if value == "entropy":
            return CalibrationMethod.Entropy
        if value == "percentile":
            return CalibrationMethod.Percentile
        if value == "distribution":
            return CalibrationMethod.Distribution
        return CalibrationMethod.MinMax

    def _load_calibration_config(self):
        if not self.calibration_config_path.exists():
            return {}
        with open(self.calibration_config_path, "r") as file:
            return yaml.safe_load(file) or {}

    def _write_report(self, stats):
        report_path = Path("reports/static_int8_quantization.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as file:
            json.dump(stats, file, indent=2)
        logger.info("Static INT8 quantization report written: %s", report_path)

    def _import_quantize_static(self):
        try:
            from onnxruntime.quantization import quantize_static

            return quantize_static
        except ImportError as exc:
            raise StaticInt8OptimizationError(
                "onnxruntime quantization APIs are required for static INT8"
            ) from exc

    def _import_onnx(self):
        try:
            import onnx

            return onnx
        except ImportError as exc:
            raise StaticInt8OptimizationError(
                "onnx is required to validate static INT8 ONNX artifacts"
            ) from exc
